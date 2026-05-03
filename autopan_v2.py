"""
AutoPan Node — v2.0  (Production-Ready)
========================================
Intelligent Autonomous Robot | Graduation Project
Subsystem: AutoPan (Computer Vision & Tracking)
Partner subsystem: Artemis (Mobility, PID, Arm)

Architecture
------------
  CameraStream   — daemon thread, zero-copy double-buffer, kills V4L queue lag
  FSM            — enum-based with entry/exit hooks and transition guards
  ArtemisLink    — serial with exponential-back-off auto-reconnect + TX queue
  FilterEMA      — smooths centroid & distance for Artemis PID
  AutoPanTracker — orchestrates inference, HSV gate, HUD, telemetry

YOLO Optimisations (edge-device pre-MNN)
-----------------------------------------
  * Half-precision (FP16) on CUDA / MPS — falls back to FP32 on CPU
  * Model warm-up pass eliminates first-frame JIT spike
  * imgsz=416 instead of 640 — ~40 % faster, negligible accuracy loss on close targets
  * conf=0.45, iou=0.4  — prunes weak detections before NMS
  * Frame-skip governor: runs inference every N frames when FPS drops below threshold
  * agnostic_nms=True   — faster NMS across single class

Omar — check SERIAL_PORT and run `python autopan_v2.py --calibrate` to set FOCAL_LENGTH.
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np
import serial
import serial.tools.list_ports
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AutoPan")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration  (single source of truth — edit here, nowhere else)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Camera
    cam_src: int = 0
    cam_w: int = 640
    cam_h: int = 480

    # YOLO
    model_path: str = "yolov8n.pt"
    infer_imgsz: int = 416          # smaller = faster on edge
    infer_conf: float = 0.45
    infer_iou: float = 0.40
    infer_half: bool = True         # FP16 — auto-disabled on CPU
    yolo_class_id: int = 32         # COCO: sports ball

    # HSV gate — tennis ball yellow-green
    hsv_low: Tuple[int, int, int] = (25, 50, 50)
    hsv_high: Tuple[int, int, int] = (45, 255, 255)
    hsv_ratio_min: float = 0.15     # min green-pixel ratio inside bbox

    # Detection
    min_box_area: int = 500

    # Distance estimation
    ball_diameter_cm: float = 6.7
    focal_length_px: float = 700.0  # calibrate with --calibrate flag

    # FSM thresholds
    dist_lock_cm: float = 15.0      # enter LOCK below this
    dist_track_cm: float = 20.0     # hysteresis band above lock
    search_timeout_s: float = 3.0   # TRACK → SEARCH if no detection
    lost_frames_limit: int = 15     # consecutive misses before going SEARCH

    # EMA alphas  (lower = smoother but more lag)
    ema_alpha_xy: float = 0.20
    ema_alpha_dist: float = 0.10

    # Serial
    serial_port: str = "COM3"
    serial_baud: int = 115200
    serial_timeout: float = 1.0
    serial_reconnect_base_s: float = 1.0   # base back-off
    serial_reconnect_max_s: float = 30.0   # cap back-off
    serial_tx_queue_size: int = 10         # drop old cmds if full

    # Performance governor
    fps_target: int = 30
    infer_skip_threshold_fps: float = 15.0  # start skipping if below this
    infer_skip_frames: int = 2              # run inference every N frames

CFG = Config()

# ──────────────────────────────────────────────────────────────────────────────
# Finite State Machine
# ──────────────────────────────────────────────────────────────────────────────
class State(Enum):
    SEARCH = auto()   # no target visible
    TRACK  = auto()   # target detected, closing in
    LOCK   = auto()   # target within grasp range


@dataclass
class FSMContext:
    """Carries all mutable data associated with the current FSM state."""
    state: State = State.SEARCH
    prev_state: State = State.SEARCH
    entered_at: float = field(default_factory=time.monotonic)
    lost_frames: int = 0
    cx: float = 0.0
    cy: float = 0.0
    dist: float = 0.0


class TrackingFSM:
    """
    Enum-based FSM with guarded transitions and entry/exit hooks.

    Transition table
    ────────────────
    SEARCH ──[detection found]──────────────────► TRACK
    TRACK  ──[dist < lock_cm]───────────────────► LOCK
    TRACK  ──[lost_frames > limit | timeout]────► SEARCH
    LOCK   ──[dist > track_cm (hysteresis)]─────► TRACK
    LOCK   ──[lost_frames > limit]──────────────► SEARCH
    """

    _ENTRY_HOOKS = {}   # populated by decorator below
    _EXIT_HOOKS  = {}

    def __init__(self, ctx: FSMContext):
        self.ctx = ctx
        self._lock = threading.Lock()

    # ── decorator helpers ──────────────────────────────────────────────────
    @classmethod
    def on_enter(cls, state: State):
        def decorator(fn):
            cls._ENTRY_HOOKS[state] = fn
            return fn
        return decorator

    @classmethod
    def on_exit(cls, state: State):
        def decorator(fn):
            cls._EXIT_HOOKS[state] = fn
            return fn
        return decorator

    def _transition(self, new_state: State):
        ctx = self.ctx
        if ctx.state == new_state:
            return

        # exit hook
        exit_hook = self._EXIT_HOOKS.get(ctx.state)
        if exit_hook:
            exit_hook(self, ctx)

        log.info("FSM %s → %s", ctx.state.name, new_state.name)
        ctx.prev_state = ctx.state
        ctx.state = new_state
        ctx.entered_at = time.monotonic()
        ctx.lost_frames = 0

        # entry hook
        entry_hook = self._ENTRY_HOOKS.get(new_state)
        if entry_hook:
            entry_hook(self, ctx)

    def update(self, detection: Optional[Tuple[float, float, float]]):
        """
        Call once per frame.
        detection = (cx, cy, dist_cm) or None if no ball found.
        Returns current State.
        """
        with self._lock:
            ctx = self.ctx
            now = time.monotonic()

            if detection is not None:
                ctx.cx, ctx.cy, ctx.dist = detection
                ctx.lost_frames = 0

                if ctx.state == State.SEARCH:
                    self._transition(State.TRACK)

                elif ctx.state == State.TRACK:
                    if ctx.dist < CFG.dist_lock_cm:
                        self._transition(State.LOCK)

                elif ctx.state == State.LOCK:
                    if ctx.dist > CFG.dist_track_cm:   # hysteresis
                        self._transition(State.TRACK)

            else:  # no detection this frame
                ctx.lost_frames += 1
                elapsed = now - ctx.entered_at

                if ctx.state in (State.TRACK, State.LOCK):
                    if (ctx.lost_frames > CFG.lost_frames_limit or
                            elapsed > CFG.search_timeout_s):
                        self._transition(State.SEARCH)

            return ctx.state


# ── Entry/Exit hooks (defined after class so decorator resolves) ──────────────
@TrackingFSM.on_enter(State.SEARCH)
def _enter_search(fsm, ctx):
    log.info("FSM SEARCH entered — scanning for target.")

@TrackingFSM.on_enter(State.TRACK)
def _enter_track(fsm, ctx):
    log.info("FSM TRACK entered — target acquired at %.1f cm.", ctx.dist)

@TrackingFSM.on_enter(State.LOCK)
def _enter_lock(fsm, ctx):
    log.info("FSM LOCK entered — target within grasp range (%.1f cm).", ctx.dist)

@TrackingFSM.on_exit(State.LOCK)
def _exit_lock(fsm, ctx):
    log.info("FSM LOCK exited — target moved out of range.")

# ──────────────────────────────────────────────────────────────────────────────
# Serial Communication with Auto-Reconnect
# ──────────────────────────────────────────────────────────────────────────────
class ArtemisLink:
    """
    Non-blocking serial link to the Artemis Arduino.

    * Dedicated TX thread drains a bounded queue → main loop never blocks on I/O
    * Exponential back-off reconnect (base=1 s, cap=30 s)
    * Graceful shutdown via stop()
    Protocol: "<cx_int>,<dist_int>,<state_int>\\n"
      state: 0=SEARCH, 1=TRACK, 2=LOCK
    """

    STATE_MAP = {State.SEARCH: 0, State.TRACK: 1, State.LOCK: 2}

    def __init__(self):
        self._ser: Optional[serial.Serial] = None
        self._q: queue.Queue = queue.Queue(maxsize=CFG.serial_tx_queue_size)
        self._running = True
        self._lock = threading.Lock()
        self._log = logging.getLogger("ArtemisLink")

        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True, name="SerialTX")
        self._tx_thread.start()

    # ── public ────────────────────────────────────────────────────────────────
    def send(self, cx: float, dist: float, state: State):
        cmd = f"{int(cx)},{int(dist)},{self.STATE_MAP[state]}\n"
        try:
            self._q.put_nowait(cmd.encode("utf-8"))
        except queue.Full:
            # Drop oldest, insert newest — real-time data: freshness > completeness
            try:
                self._q.get_nowait()
                self._q.put_nowait(cmd.encode("utf-8"))
            except queue.Empty:
                pass

    def stop(self):
        self._running = False
        self._tx_thread.join(timeout=3)
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
                self._log.info("Serial port closed.")

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._ser is not None and self._ser.is_open

    # ── private ───────────────────────────────────────────────────────────────
    def _connect(self) -> bool:
        try:
            ser = serial.Serial(
                CFG.serial_port,
                CFG.serial_baud,
                timeout=CFG.serial_timeout,
            )
            with self._lock:
                self._ser = ser
            self._log.info("Connected to %s @ %d baud.", CFG.serial_port, CFG.serial_baud)
            return True
        except serial.SerialException as e:
            self._log.warning("Serial connect failed: %s", e)
            return False

    def _tx_loop(self):
        """Dedicated writer thread with exponential back-off reconnect."""
        backoff = CFG.serial_reconnect_base_s

        while self._running:
            if not self.connected:
                if self._connect():
                    backoff = CFG.serial_reconnect_base_s  # reset on success
                else:
                    self._log.info("Retrying serial in %.1f s…", backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, CFG.serial_reconnect_max_s)
                    continue

            try:
                payload = self._q.get(timeout=0.5)
                with self._lock:
                    if self._ser and self._ser.is_open:
                        self._ser.write(payload)
            except queue.Empty:
                pass
            except serial.SerialException as e:
                self._log.error("Serial write error: %s — will reconnect.", e)
                with self._lock:
                    if self._ser:
                        try:
                            self._ser.close()
                        except Exception:
                            pass
                        self._ser = None

# ──────────────────────────────────────────────────────────────────────────────
# Camera Stream  (double-buffer, daemon thread)
# ──────────────────────────────────────────────────────────────────────────────
class CameraStream:
    def __init__(self):
        self._cap = cv2.VideoCapture(CFG.cam_src)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.cam_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_h)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # kill V4L queue

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera src={CFG.cam_src}")

        self._ret, self._frame = self._cap.read()
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._update, daemon=True, name="CamStream")
        self._thread.start()
        log.info("CameraStream started (src=%d, %dx%d).", CFG.cam_src, CFG.cam_w, CFG.cam_h)

    def _update(self):
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret, self._frame = ret, frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self._ret:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()
        log.info("CameraStream released.")

# ──────────────────────────────────────────────────────────────────────────────
# EMA Filter
# ──────────────────────────────────────────────────────────────────────────────
class FilterEMA:
    def __init__(self, alpha: float, initial: float = 0.0):
        self.alpha = alpha
        self._val: Optional[float] = None

    def update(self, x: float) -> float:
        if self._val is None:
            self._val = x
        else:
            self._val = self.alpha * x + (1.0 - self.alpha) * self._val
        return self._val

    def reset(self):
        self._val = None

# ──────────────────────────────────────────────────────────────────────────────
# HUD Renderer
# ──────────────────────────────────────────────────────────────────────────────
class HUD:
    _STATE_COLOR = {
        State.SEARCH: (0,   0,   255),
        State.TRACK:  (0,   165, 255),
        State.LOCK:   (0,   255, 0),
    }
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    @classmethod
    def draw(cls, frame: np.ndarray, ctx: FSMContext, fps: float, serial_ok: bool):
        h, w = frame.shape[:2]
        color = cls._STATE_COLOR[ctx.state]

        # crosshair at frame centre
        cx_f, cy_f = w // 2, h // 2
        cv2.line(frame, (cx_f - 20, cy_f), (cx_f + 20, cy_f), (80, 80, 80), 1)
        cv2.line(frame, (cx_f, cy_f - 20), (cx_f, cy_f + 20), (80, 80, 80), 1)

        if ctx.state != State.SEARCH:
            bx, by = int(ctx.cx), int(ctx.cy)

            # error vector from centre to target
            cv2.arrowedLine(frame, (cx_f, cy_f), (bx, by), (200, 200, 0), 1, tipLength=0.15)
            cv2.drawMarker(frame, (bx, by), color, cv2.MARKER_CROSS, 20, 2)

            cv2.putText(frame, f"DIST: {ctx.dist:.1f} cm",
                        (bx + 10, by - 10), cls._FONT, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"ERR X: {bx - cx_f:+d}px",
                        (bx + 10, by + 14), cls._FONT, 0.4, (200, 200, 200), 1)

        # state banner
        banner = ctx.state.name
        cv2.rectangle(frame, (0, 0), (200, 30), (20, 20, 20), -1)
        cv2.putText(frame, banner, (8, 22), cls._FONT, 0.7, color, 2)

        # telemetry strip (top-right)
        serial_color = (0, 220, 0) if serial_ok else (0, 0, 200)
        cv2.putText(frame, f"FPS: {fps:4.1f}", (w - 120, 20), cls._FONT, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"SER: {'OK' if serial_ok else 'NO'}",
                    (w - 120, 40), cls._FONT, 0.5, serial_color, 1)

# ──────────────────────────────────────────────────────────────────────────────
# Main Tracker
# ──────────────────────────────────────────────────────────────────────────────
class AutoPanTracker:
    def __init__(self):
        self._log = logging.getLogger("Tracker")
        self._shutdown = threading.Event()
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # ── YOLO ──────────────────────────────────────────────────────────────
        self._log.info("Loading YOLO model: %s", CFG.model_path)
        self.model = YOLO(CFG.model_path)

        # Determine device and half-precision capability
        import torch
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
            CFG.infer_half = False   # MPS doesn't support FP16 reliably yet
        else:
            self._device = "cpu"
            CFG.infer_half = False

        self._log.info("Inference device: %s  |  FP16: %s", self._device, CFG.infer_half)

        # Warm-up pass (eliminates first-frame JIT spike)
        dummy = np.zeros((CFG.infer_imgsz, CFG.infer_imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=CFG.infer_imgsz, device=self._device,
                           half=CFG.infer_half, verbose=False)
        self._log.info("YOLO warm-up done.")

        # ── HSV kernel ────────────────────────────────────────────────────────
        self._hsv_low  = np.array(CFG.hsv_low,  dtype=np.uint8)
        self._hsv_high = np.array(CFG.hsv_high, dtype=np.uint8)

        # ── Filters ───────────────────────────────────────────────────────────
        self._fx   = FilterEMA(CFG.ema_alpha_xy)
        self._fy   = FilterEMA(CFG.ema_alpha_xy)
        self._fdist = FilterEMA(CFG.ema_alpha_dist)

        # ── FSM ───────────────────────────────────────────────────────────────
        self._ctx = FSMContext()
        self._fsm = TrackingFSM(self._ctx)

        # ── Comms ─────────────────────────────────────────────────────────────
        self._comm = ArtemisLink()

        # ── FPS governor ──────────────────────────────────────────────────────
        self._fps: float = 0.0
        self._skip_counter: int = 0
        self._last_result = None   # cached result for skipped frames

    # ── Signal handling ───────────────────────────────────────────────────────
    def _handle_signal(self, signum, frame):
        self._log.info("Shutdown signal received.")
        self._shutdown.set()

    # ── Distance estimation ───────────────────────────────────────────────────
    def _estimate_distance(self, bbox_width_px: int) -> float:
        if bbox_width_px <= 0:
            return 0.0
        return (CFG.ball_diameter_cm * CFG.focal_length_px) / bbox_width_px

    # ── HSV gate ─────────────────────────────────────────────────────────────
    def _passes_hsv_gate(self, roi: np.ndarray, area: int) -> bool:
        if roi.size == 0 or area == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._hsv_low, self._hsv_high)
        return (cv2.countNonZero(mask) / area) >= CFG.hsv_ratio_min

    # ── Inference ─────────────────────────────────────────────────────────────
    def _run_inference(self, frame: np.ndarray):
        return self.model.predict(
            frame,
            classes=[CFG.yolo_class_id],
            imgsz=CFG.infer_imgsz,
            conf=CFG.infer_conf,
            iou=CFG.infer_iou,
            half=CFG.infer_half,
            device=self._device,
            agnostic_nms=True,   # faster single-class NMS
            verbose=False,
        )

    # ── Best detection picker ─────────────────────────────────────────────────
    def _pick_best(self, results, frame: np.ndarray):
        best_box  = None
        max_area  = 0

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area < CFG.min_box_area:
                    continue
                roi = frame[y1:y2, x1:x2]
                if self._passes_hsv_gate(roi, area) and area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2, x2 - x1)

        return best_box

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cam = CameraStream()
        t_prev = time.monotonic()

        self._log.info("AutoPan running. Press Q to quit.")

        try:
            while not self._shutdown.is_set():
                ret, frame = cam.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                # ── FPS calculation ───────────────────────────────────────────
                now = time.monotonic()
                dt = now - t_prev
                t_prev = now
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt if dt > 0 else 0)

                # ── Frame-skip governor ───────────────────────────────────────
                run_infer = True
                if self._fps < CFG.infer_skip_threshold_fps:
                    self._skip_counter += 1
                    if self._skip_counter % CFG.infer_skip_frames != 0:
                        run_infer = False   # reuse cached result

                if run_infer:
                    results = self._run_inference(frame)
                    self._last_result = results
                else:
                    results = self._last_result

                # ── Pick best detection ───────────────────────────────────────
                best = self._pick_best(results, frame) if results else None
                detection = None

                if best:
                    x1, y1, x2, y2, w = best
                    raw_cx = (x1 + x2) / 2.0
                    raw_cy = (y1 + y2) / 2.0
                    raw_d  = self._estimate_distance(w)

                    cx   = self._fx.update(raw_cx)
                    cy   = self._fy.update(raw_cy)
                    dist = self._fdist.update(raw_d)

                    detection = (cx, cy, dist)

                    # draw bbox
                    color = HUD._STATE_COLOR[self._ctx.state]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                else:
                    # Reset EMA so stale values don't poison next acquisition
                    self._fx.reset()
                    self._fy.reset()
                    self._fdist.reset()

                # ── FSM update ────────────────────────────────────────────────
                state = self._fsm.update(detection)

                # ── Send to Artemis ───────────────────────────────────────────
                if detection:
                    self._comm.send(self._ctx.cx, self._ctx.dist, state)
                else:
                    self._comm.send(0, 0, state)

                # ── HUD ───────────────────────────────────────────────────────
                HUD.draw(frame, self._ctx, self._fps, self._comm.connected)
                cv2.imshow("AutoPan Node v2", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._shutdown.set()

        except Exception as e:
            self._log.exception("Fatal error in main loop: %s", e)
        finally:
            self._log.info("Shutting down…")
            cam.release()
            self._comm.stop()
            cv2.destroyAllWindows()
            self._log.info("AutoPan stopped cleanly.")

# ──────────────────────────────────────────────────────────────────────────────
# Focal-length calibration helper
# ──────────────────────────────────────────────────────────────────────────────
def calibrate_focal_length():
    """
    Place the tennis ball at a KNOWN distance (enter it below),
    then press SPACE when the ball is fully visible.
    Prints the focal_length value to paste into Config.
    """
    KNOWN_DIST_CM = float(input("Enter known distance from camera to ball centre (cm): "))
    cap = cv2.VideoCapture(CFG.cam_src)
    model = YOLO(CFG.model_path)
    print("Press SPACE when ball is clearly visible, Q to abort.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        results = model.predict(frame, classes=[CFG.yolo_class_id], verbose=False)
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                w = x2 - x1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"w={w}px", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Calibration", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord(" "):
            # compute from last detected width
            for r in results:
                for b in r.boxes:
                    w = int(b.xyxy[0][2]) - int(b.xyxy[0][0])
                    fl = (KNOWN_DIST_CM * w) / CFG.ball_diameter_cm  # rearranged
                    # correct formula: FL = (W_px * D) / D_real
                    fl = (w * KNOWN_DIST_CM) / CFG.ball_diameter_cm
                    print(f"\n✓  FOCAL_LENGTH = {fl:.1f}  ← paste into Config.focal_length_px\n")
            break

        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoPan Node v2 — Intelligent Autonomous Robot")
    parser.add_argument("--calibrate", action="store_true", help="Run focal-length calibration")
    parser.add_argument("--port",  default=CFG.serial_port, help="Serial port (default: COM3)")
    parser.add_argument("--cam",   type=int, default=CFG.cam_src, help="Camera index (default: 0)")
    parser.add_argument("--imgsz", type=int, default=CFG.infer_imgsz, help="YOLO input size")
    args = parser.parse_args()

    CFG.serial_port  = args.port
    CFG.cam_src      = args.cam
    CFG.infer_imgsz  = args.imgsz

    if args.calibrate:
        calibrate_focal_length()
    else:
        tracker = AutoPanTracker()
        tracker.run()
