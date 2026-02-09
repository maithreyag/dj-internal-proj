import cv2
from hand_tracker import HandTracker, draw_hand_skeleton
from song_selector import SongSelector
from ui import PlayButton, StemButton, Deck, Waveform

def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        return

    height, width, _ = frame.shape

    def_left = "ctmn"
    def_right = "crew_love"

    song_selector = SongSelector()
    song_selector.select("left", def_left)
    song_selector.select("right", def_right)

    # Play buttons (display coords: left on left, right on right)
    left_button = PlayButton(width // 4 - 30, 2 * height // 3, 100, 100, selector=song_selector, side="left")
    right_button = PlayButton(3 * width // 4 - 70, 2 * height // 3, 100, 100, selector=song_selector, side="right")

    buttons = [left_button, right_button]

    stem_labels = ["bass", "drm", "oth", "vox"]
    stem_size = 70
    gap = 40

    # Left stems (to the left of left play button)
    lx = width // 4 - 30 - (2 * stem_size + gap) - gap
    ly = 2 * height // 3
    for i, label in enumerate(stem_labels):
        row, col = divmod(i, 2)
        buttons.append(StemButton(
            lx + col * (stem_size + gap), ly + row * (stem_size + gap),
            stem_size, stem_size,
            selector=song_selector, side="left", stem_index=i, label=label))

    # Right stems (to the right of right play button)
    rx = 3 * width // 4 + 30 + gap
    ry = 2 * height // 3
    for i, label in enumerate(stem_labels):
        row, col = divmod(i, 2)
        buttons.append(StemButton(
            rx + col * (stem_size + gap), ry + row * (stem_size + gap),
            stem_size, stem_size,
            selector=song_selector, side="right", stem_index=i, label=label))

    # Decks (top corners)
    deck_radius = height // 4
    left_deck = Deck(deck_radius + 20, deck_radius + 20, deck_radius, selector=song_selector, side="left", label="L")
    right_deck = Deck(width - deck_radius - 20, deck_radius + 20, deck_radius, selector=song_selector, side="right", label="R")
    decks = [left_deck, right_deck]

    wf_height = 60
    wf_y = 2 * deck_radius + 40
    left_wf = Waveform(left_deck.cx - deck_radius, wf_y, 2 * deck_radius, wf_height, selector=song_selector, side="left")
    right_wf = Waveform(right_deck.cx - deck_radius, wf_y, 2 * deck_radius, wf_height, selector=song_selector, side="right")
    waveforms = [left_wf, right_wf]

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

            frame = draw_hand_skeleton(frame, tracker, result)

            for hand in ["Left", "Right"]:
                pinch_pos = tracker.pinch_pos[hand]
                # Flip to display coords for hit detection
                if pinch_pos:
                    pinch_pos = (width - 1 - pinch_pos[0], pinch_pos[1])

                for button in buttons:
                    if tracker.state[hand] == 1:
                        button.update(hand, pinch_pos)
                    else:
                        button.pinched[hand] = False

                press_pos = tracker.press_pos[hand]
                if press_pos:
                    press_pos = (width - 1 - press_pos[0], press_pos[1])
                for deck in decks:
                    if tracker.state[hand] == 2:
                        deck.update(hand, press_pos)
                    else:
                        deck.prev_angle[hand] = None

            reversed_frame = cv2.flip(frame, 1)

            for button in buttons:
                button.draw(reversed_frame)
                if hasattr(button, 'draw_label'):
                    button.draw_label(reversed_frame)

            for deck in decks:
                deck.draw(reversed_frame)

            for wf in waveforms:
                wf.draw(reversed_frame)

            cv2.imshow('CV DJ Set', reversed_frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        song_selector.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
