import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [(4, 8), (8, 12)]

LANDMARKS = [4, 8, 12]

MERGE_DIST = 65

class HandTracker:
    def __init__(self, model_path='./hand_landmarker.task'):
        self.latest_result = None
        
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=self._result_callback,
            num_hands=2
        )
        
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.start_time = time.time() * 1000

        self.state = {"Left": 0, "Right": 0} # 0: default, 1: spin, 2: pinch

    def _result_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def get_latest_result(self):
        return self.latest_result

    def detect_async(self, frame):
        now = time.time() * 1000
        timestamp = int(now - self.start_time)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, timestamp)

    def close(self):
        self.landmarker.close()

def draw_overlay(frame, tracker, result):
    """
    Draws the DJ-style skeleton and UI elements.
    """
    height, width, _ = frame.shape
    
    cv2.rectangle(frame, (width // 3, 2 * height // 3), (width // 3 + 100, 2 * height // 3 + 100), (0, 0, 255), 3)
    cv2.rectangle(frame, (2 * width // 3, 2 * height // 3), (2 * width // 3 + 100, 2 * height // 3 + 100), (0, 0, 255), 3)

    if not result or not result.hand_landmarks:
        return frame
    
    for h, hand in enumerate(result.hand_landmarks):

        handedness = result.handedness[h][0].category_name

        dots = []

        state = 0
        i = 0
        while i < len(LANDMARKS):
            if i + 1 < len(LANDMARKS):
                start_idx, end_idx = LANDMARKS[i], LANDMARKS[i + 1]
                start, end = hand[start_idx], hand[end_idx]

                x1, y1 = int(start.x * width), int(start.y * height)
                x2, y2 = int(end.x * width), int(end.y * height)

                dist = abs(x2 - x1) + abs(y2 - y1)

                if dist <= MERGE_DIST:
                    if (start_idx, end_idx) == (4, 8):
                        state = 1
                    else:
                        state = 2
                    x_mid, y_mid = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.circle(frame, (x_mid, y_mid), 6, (255, 255, 255), -1)
                    dots.append((x_mid, y_mid))
                    i += 2
                else:
                    cv2.circle(frame, (x1, y1), 4, (255, 255, 255), -1)
                    i += 1
                    dots.append((x1, y1))
            else:
                idx = LANDMARKS[i]
                landmark = hand[idx]
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
                dots.append((x, y))
                i += 1
        
        tracker.state[handedness] = state

        for i in range(len(dots) - 1):
            start, end = dots[i], dots[i + 1]

            cv2.line(frame, start, end, (255, 255, 255), 1)

        
    return frame

def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    print("DJ Hand Tracking Started. Press 'q' to exit.")

    try:
    
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracker.detect_async(rgb_frame)

            result = tracker.get_latest_result()
            frame = draw_overlay(frame, tracker, result)
            
            reversed_frame = cv2.flip(frame, 1)
            cv2.imshow('CV DJ Set', reversed_frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()