"""
Gesture-to-DJ-action handlers.

This file ships with stub (pass) implementations. To add your own behavior:
1. Override any function below (or add new ones).
2. Register it in main.py's ONE_SHOT or CONTINUOUS dispatch tables so the
   app loop calls it, e.g.:
       ONE_SHOT = {"fist": gesture_actions.on_fist}
       CONTINUOUS = {"peace": gesture_actions.hold_peace}

All handlers receive:
    action        – gesture name (e.g. "fist", "peace", "thumb")
    side          – "left" or "right"
    song_selector – the SongSelector instance
    ui            – dict of UI references (see below)

ui keys:
    "left_button", "right_button"   – PlayButton instances
    "left_stems",  "right_stems"    – lists of StemButton instances

One-shot handlers fire once when a gesture is first detected.
Continuous handlers fire every frame while the gesture is held.
"""

def on_fist(action, side, song_selector, ui):
    """Fired once when a fist gesture is first detected."""
    pass


# ── Continuous (called every frame while gesture is held) ────────────────

def hold_peace(action, side, song_selector, ui):
    """Called every frame while peace gesture is held."""
    pass


def hold_thumb(action, side, song_selector, ui):
    """Called every frame while thumbs-up gesture is held."""
    pass
