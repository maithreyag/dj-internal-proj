import cv2
from hand_tracking import HandTracker, draw_hand_skeleton
from ui import Button
import pygame

def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(16)
    bass_left = pygame.mixer.Sound("songs/oouuh/bass.mp3")
    drums_left = pygame.mixer.Sound("songs/oouuh/drums.mp3")
    instrumental_left = pygame.mixer.Sound("songs/oouuh/instrumental.mp3")
    other_left= pygame.mixer.Sound("songs/oouuh/other.mp3")
    vocals_left = pygame.mixer.Sound("songs/oouuh/vocals.mp3")

    bass_right = pygame.mixer.Sound("songs/ghetto_angels/bass.mp3")
    drums_right = pygame.mixer.Sound("songs/ghetto_angels/drums.mp3")
    instrumental_right = pygame.mixer.Sound("songs/ghetto_angels/instrumental.mp3")
    other_right = pygame.mixer.Sound("songs/ghetto_angels/other.mp3")
    vocals_right = pygame.mixer.Sound("songs/ghetto_angels/vocals.mp3")
    
    left_sounds = [bass_left, drums_left, instrumental_left, other_left, vocals_left]
    right_sounds = [bass_right, drums_right, instrumental_right, other_right, vocals_right]

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        return

    height, width, _ = frame.shape

    left_button = Button(2 * width // 3, 2 * height // 3, 100, 100, sounds=left_sounds)
    right_button = Button(width // 3, 2 * height // 3, 100, 100, sounds=right_sounds)
    buttons = [left_button, right_button]

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

            # Draw hand skeleton
            frame = draw_hand_skeleton(frame, tracker, result)

            for hand in ["Left", "Right"]:
                pinch_pos = tracker.pinch_pos[hand]
                for button in buttons:
                    if tracker.state[hand] == 1:
                        button.update(hand, pinch_pos)
                    else:
                        button.pinched[hand] = False
                    
                    if tracker.state[hand] == 2:
                        pass
                    else:
                        pass

            for button in buttons:
                button.draw(frame)

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
