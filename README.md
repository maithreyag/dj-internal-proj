# Mini DJ 🎛️

Real-time gesture-controlled DJ built in Python. MediaPipe tracks two hands, a custom PyTorch classifier recognizes gestures, and a stem-based audio engine mixes music — all running in a single OpenCV loop over your webcam at interactive framerates. **Pinch** to press buttons, spin vinyl decks, and drag sliders. **Gesture** to trigger deck actions like pause or stem isolation.

The hardest problems were getting the audio callback to stay glitch-free under variable playback rates (solved with vectorized NumPy interpolation in the `sounddevice` callback), and making pinch detection reliable (hysteresis thresholds with separate merge/unmerge distances to avoid flickering).

### [▶ Watch the Demo](https://drive.google.com/file/d/1HLQVPp4IvFh-YlwH3QAcSvwzPhU1dOjl/view?usp=sharing)

---

## Tech Stack

Python 3.10+ · MediaPipe (hand landmark async streaming) · PyTorch (gesture MLP) · OpenCV (rendering + camera) · sounddevice/soundfile (real-time audio) · NumPy

---

## Playback

**Audio engine** (`playback/selector.py`) — Dual-deck mixer where each track is pre-split into 4 stems (bass, drums, other, vocals). A `sounddevice` output stream callback mixes all active stems per frame using vectorized linear interpolation at arbitrary playback rates. BPM sync works by resampling both decks to their average BPM — simple `np.interp` time-stretching that trades pitch accuracy for zero-latency, zero-dependency sync. Per-stem mute/unmute, seeking, and memory cue points are all supported.

**UI** (`playback/ui.py`) — Entirely OpenCV-drawn: two spinning vinyl decks (rotation driven by pinch-drag angle), live waveform displays with playhead, play/stem toggle buttons, BPM sliders, and memory cue controls. All interaction is pinch-based — the tracker provides a pinch coordinate each frame, and each UI element does its own hit-testing.

**Hand tracking** (`hand_tracking/tracker.py`) — MediaPipe's `HandLandmarker` runs in `LIVE_STREAM` mode with an async result callback, so landmark detection doesn't block the render loop. A state machine on landmarks 4 (thumb tip), 8 (index tip), and 12 (middle tip) detects pinch via Manhattan distance with hysteresis (merge at 60px, unmerge at 80px) to prevent flicker.

---

## Gestures

The gesture system is fully trainable. A PyTorch MLP (60→64→64→N) classifies wrist-centered, scale-normalized landmark vectors at >0.8 confidence (`hand_tracking/classifier.py`). All inference runs on CPU — no GPU needed.

Gesture names **must** end with `-l` or `-r` to indicate which hand (e.g. `fist-r`, `peace-l`). The suffix is parsed to route the action to the correct deck.

**1. Collect** — `python tools/collect.py` — Webcam HUD; type a gesture name, press **R** to record yourself performing it (~12fps landmark capture → `data/gesture_data.csv`). **N** to switch gestures, **Q** to save.

**2. Train** — `python tools/train.py` — Trains the MLP with inverse-frequency class weighting. Outputs `models/gesture_model.pt` + `models/gesture_encoder.joblib`. Prints accuracy and confusion matrix.

**3. Test** — `python tools/test.py` — Live webcam showing predicted gesture + confidence, no DJ UI. Verify accuracy before wiring into the app.

**4. Audit** (optional) — `python tools/audit.py` — Flags `none` samples that look like real gestures. Prompts to remove, then retrain.

**5. Wire up** — Define handlers in `hand_tracking/gesture_actions.py`, register them in `main.py`:

```python
ONE_SHOT = {"fist": gesture_actions.on_fist}          # fires once on gesture change
CONTINUOUS = {"peace": gesture_actions.hold_peace}     # fires every frame while held
```

---

## Setup

Requires **Python 3.10+** and a webcam.

```bash
git clone https://github.com/maithreyag/mini-dj.git && cd mini-dj
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Prepare Songs

Place each song in `songs/<name>/` with pre-separated stems:

```
songs/my-track/
├── bass.mp3
├── drums.mp3
├── other.mp3
├── vocals.mp3
└── bpm.txt       # single number, e.g. "128" — look up manually on songbpm.com
```

**Stem separation workflow:**
1. Download MP3 from Spotify via [SpotDown](https://spotdown.org/)
2. Upload to [Fadr](https://fadr.com/stems) to split stems (or use [Demucs](https://github.com/facebookresearch/demucs) locally)
3. Download all stems except instrumental, rename to `bass.mp3`, `drums.mp3`, `other.mp3`, `vocals.mp3`

### Run

```bash
python main.py
```

Select songs for each deck via terminal menu → DJ interface launches over webcam.
