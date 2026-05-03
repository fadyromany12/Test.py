import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import serial 

# Omar, check the COM port for the Arduino. It defaults to COM3 here.
class ArtemisLink:
    def __init__(self, port='COM3', baudrate=115200):
        self.connected = False
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self.connected = True
            print(f"[ArtemisLink] Connected to {port}")
        except Exception as e:
            print(f"[ArtemisLink] Serial error: {e}. Running vision only.")

    def send_command(self, cx, dist, state):
        # Format: X_Centroid, Distance, Current_State
        if self.connected:
            cmd = f"{int(cx)},{int(dist)},{state}\n"
            self.ser.write(cmd.encode('utf-8'))

class CameraStream:
    # Multithreaded camera to kill buffer delay.
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class FilterEMA:
    # Smooths out the sensor noise for Artemis PID.
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.val = None

    def update(self, new_val):
        if self.val is None:
            self.val = new_val
        else:
            self.val = (self.alpha * new_val) + ((1.0 - self.alpha) * self.val)
        return self.val

class AutoPanTracker:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        
        # FSM States
        self.STATE_SEARCH = 0
        self.STATE_TRACK = 1
        self.STATE_LOCK = 2
        self.current_state = self.STATE_SEARCH

        # Calibration vars (Omar, adjust FOCAL_LENGTH after testing)
        self.BALL_DIAM = 6.7 
        self.FOCAL_LENGTH = 700.0  
        
        # HSV threshold for tennis ball
        self.HSV_L = np.array([25, 50, 50])
        self.HSV_H = np.array([45, 255, 255])
        
        self.fx = FilterEMA(0.2)
        self.fy = FilterEMA(0.2)
        self.fdist = FilterEMA(0.1)

        # Initialize Arduino comms
        self.comm = ArtemisLink('COM3') 

    def get_distance(self, w):
        if w == 0: return 0
        return (self.BALL_DIAM * self.FOCAL_LENGTH) / w

    def draw_hud(self, frame, box, cx, cy, dist):
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if self.current_state == self.STATE_LOCK else (0, 165, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.drawMarker(frame, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
        
        state_str = "LOCKED" if self.current_state == self.STATE_LOCK else "TRACKING"
        cv2.putText(frame, f"STATE: {state_str}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"DIST: {dist:.1f}cm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        cam = CameraStream(0)
        time.sleep(1)
        print("[AutoPan] System started.")
        
        while True:
            ret, frame = cam.read()
            if not ret: continue

            results = self.model.predict(frame, classes=[32], verbose=False)
            
            best_box = None
            max_area = 0
            
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    area = (x2-x1) * (y2-y1)
                    
                    if area < 500: continue
                        
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), self.HSV_L, self.HSV_H)
                        if (cv2.countNonZero(mask) / area) > 0.15 and area > max_area:
                            max_area = area
                            best_box = (x1, y1, x2, y2, x2-x1)

            if best_box:
                x1, y1, x2, y2, w = best_box
                cx = self.fx.update((x1 + x2) / 2)
                cy = self.fy.update((y1 + y2) / 2)
                dist = self.fdist.update(self.get_distance(w))

                # Update State Machine
                if dist < 15.0:
                    self.current_state = self.STATE_LOCK
                else:
                    self.current_state = self.STATE_TRACK

                self.draw_hud(frame, best_box[:4], cx, cy, dist)
                
                # Send data to Artemis
                self.comm.send_command(cx, dist, self.current_state)
            else:
                self.current_state = self.STATE_SEARCH
                cv2.putText(frame, "SEARCHING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.comm.send_command(0, 0, self.current_state)

            cv2.imshow("AutoPan Node", frame)
            if cv2.waitKey(1) == ord('q'): break

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AutoPanTracker()
    app.run()
