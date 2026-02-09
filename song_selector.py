import sounddevice as sd
import soundfile as sf
import numpy as np

STEMS = ["bass", "drums", "other", "vocals"]

class SongSelector:
    def __init__(self, sr=44100):
        self.sr = sr
        self.stems = {"left": [], "right": []}
        self.playing = {"left": False, "right": False}
        self.volumes = {"left": [1.0] * 4, "right": [1.0] * 4}
        self.position = {"left": 0, "right": 0}

        self.stream = sd.OutputStream(
            samplerate=sr,
            channels=2,
            dtype='float32',
            callback=self._callback
        )
        self.stream.start()

    def _callback(self, outdata, frames, time, status):
        outdata[:] = 0
        for side in ["left", "right"]:
            if not self.playing[side] or not self.stems[side]:
                continue
            pos = self.position[side]
            max_len = max(len(s) for s in self.stems[side])
            if pos >= max_len:
                self.playing[side] = False
                continue
            for i, stem_data in enumerate(self.stems[side]):
                if pos >= len(stem_data):
                    continue
                end = min(pos + frames, len(stem_data))
                outdata[:end - pos] += stem_data[pos:end] * self.volumes[side][i]
            self.position[side] = pos + frames

    def play(self, side):
        self.playing[side] = True

    def pause(self, side):
        self.playing[side] = False

    def mute(self, side, stem_index):
        self.volumes[side][stem_index] = 0.0

    def unmute(self, side, stem_index):
        self.volumes[side][stem_index] = 1.0

    def select(self, side, song):
        self.playing[side] = False
        loaded = []
        for stem in STEMS:
            data, sr = sf.read(f"songs/{song}/{stem}.mp3", dtype='float32')
            if data.ndim == 1:
                data = np.column_stack([data, data])
            loaded.append(data)
        self.stems[side] = loaded
        self.position[side] = 0

    def seek(self, side, ds):
        self.position[side] += int(ds * self.sr)
        max_len = max(len(s) for s in self.stems[side]) if self.stems[side] else 0
        self.position[side] = max(0, min(self.position[side], max_len))

    def get_position(self, side):
        if not self.stems[side]:
            return 0.0
        return self.position[side] / self.sr

    def get_duration(self, side):
        if not self.stems[side]:
            return 0.0
        return max(len(s) for s in self.stems[side]) / self.sr

    def close(self):
        self.stream.stop()
        self.stream.close()
