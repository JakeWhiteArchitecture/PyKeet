"""Embedded text-to-speech for PyKeet's proofreader: Kokoro first, Piper as the fallback.

Both give us the audio samples, so PyKeet plays them itself, can pause/rewind exactly, can
synthesise the next phrase while the current one plays, and knows where each word falls.
Models download on first use into ./tts-models/. Optional installs:
    pip install kokoro-onnx          (Kokoro: livelier, ~120 MB of model files)
    pip install piper-tts            (Piper: lighter, ~60 MB voice)
"""
from __future__ import annotations

import logging
import re
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger("pykeet")

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_MODELS = {"int8": "kokoro-v1.0.int8.onnx", "fp16": "kokoro-v1.0.fp16.onnx",
                 "full": "kokoro-v1.0.onnx"}
KOKORO_VOICES = "voices-v1.0.bin"


class TTSUnavailable(Exception):
    pass


def download(url: str, dest: Path, note=None) -> None:
    """Download url to dest (atomically), telling note(msg) what is happening."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if note:
        note(f"Downloading {dest.name} (first use only)…")
    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
    tmp.replace(dest)


class KokoroSpeaker:
    name = "Kokoro"

    def __init__(self, models_dir: Path, cfg: dict, note=None):
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise TTSUnavailable("pip install kokoro-onnx") from exc
        model = models_dir / KOKORO_MODELS.get(str(cfg["kokoro_model"]), KOKORO_MODELS["int8"])
        voices = models_dir / KOKORO_VOICES
        for path in (model, voices):
            if not path.exists():
                download(KOKORO_BASE + path.name, path, note)
        self.k = Kokoro(str(model), str(voices))
        self.voice = str(cfg["kokoro_voice"])
        self.speed = float(cfg["kokoro_speed"])
        self.lang = str(cfg["kokoro_lang"])

    def synth(self, text: str):
        audio, sr = self.k.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        return np.asarray(audio, dtype=np.float32), int(sr)


class PiperSpeaker:
    name = "Piper"

    def __init__(self, models_dir: Path, cfg: dict, note=None):
        try:
            from piper import PiperVoice
            from piper.config import SynthesisConfig
            from piper.download_voices import download_voice
        except ImportError as exc:
            raise TTSUnavailable("pip install piper-tts") from exc
        voice = str(cfg["piper_voice"])
        path = models_dir / f"{voice}.onnx"
        if not path.exists():
            if note:
                note(f"Downloading Piper voice {voice} (first use only)…")
            models_dir.mkdir(parents=True, exist_ok=True)
            download_voice(voice, models_dir)
        self.voice = PiperVoice.load(path)
        self.syn = SynthesisConfig(length_scale=float(cfg["piper_speed"]),
                                   noise_scale=float(cfg["piper_noise_scale"]),
                                   noise_w_scale=float(cfg["piper_noise_w"]))

    def synth(self, text: str):
        parts, sr = [], 22050
        for chunk in self.voice.synthesize(text, self.syn):
            parts.append(np.asarray(chunk.audio_float_array, dtype=np.float32))
            sr = int(chunk.sample_rate)
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), sr


_SPEAKER = None
_SPEAKER_KEY = None
_SPEAKER_LOCK = threading.Lock()


def get_speaker(cfg: dict, models_dir: Path, note=None):
    """The best available speaker for cfg['tts_engine'] ("auto", "kokoro", "piper"),
    or None if neither is installed/usable. Cached after the first call."""
    global _SPEAKER, _SPEAKER_KEY
    engine = str(cfg["tts_engine"]).lower()
    key = (engine, cfg["kokoro_model"], cfg["kokoro_voice"], cfg["piper_voice"])
    with _SPEAKER_LOCK:
        if _SPEAKER_KEY == key:
            return _SPEAKER
        order = {"auto": [KokoroSpeaker, PiperSpeaker], "kokoro": [KokoroSpeaker, PiperSpeaker],
                 "piper": [PiperSpeaker]}.get(engine, [KokoroSpeaker, PiperSpeaker])
        speaker = None
        for cls in order:
            try:
                speaker = cls(models_dir, cfg, note)
                log.info("text-to-speech engine: %s", speaker.name)
                break
            except TTSUnavailable as exc:
                log.warning("%s not available (%s)", cls.name, exc)
            except Exception:
                log.exception("%s failed to start", cls.name)
        _SPEAKER, _SPEAKER_KEY = speaker, key
        return speaker


class AudioCache:
    """Synthesises phrases, and the next one in the background while this one plays."""

    def __init__(self, speaker):
        self.speaker = speaker
        self._done: dict[str, tuple] = {}
        self._busy: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._synth_lock = threading.Lock()

    def prefetch(self, text: str) -> None:
        with self._lock:
            if text in self._done or text in self._busy:
                return
            self._busy[text] = threading.Event()
        threading.Thread(target=self._work, args=(text,), daemon=True).start()

    def _work(self, text: str) -> None:
        try:
            with self._synth_lock:
                result = self.speaker.synth(text)
            with self._lock:
                self._done[text] = result
                if len(self._done) > 6:  # keep memory small
                    self._done.pop(next(iter(self._done)))
        except Exception:
            log.exception("speech synthesis failed")
        finally:
            with self._lock:
                ev = self._busy.pop(text, None)
            if ev:
                ev.set()

    def get(self, text: str):
        """(samples, sample_rate); raises if synthesis failed."""
        self.prefetch(text)
        while True:
            with self._lock:
                if text in self._done:
                    return self._done[text]
                ev = self._busy.get(text)
            if ev is None:
                raise TTSUnavailable("speech synthesis failed (see the log)")
            ev.wait()


class Playback:
    """Plays samples through sounddevice; looks like a subprocess (poll/terminate/wait) so the
    proofreader can stop it the same way as a subprocess."""

    def __init__(self, samples: np.ndarray, sr: int):
        import sounddevice as sd

        self.sd, self.samples, self.sr = sd, samples, sr
        self.duration = len(samples) / sr
        self._stopped = threading.Event()
        self.started_at = 0.0

    def start(self) -> None:
        self.sd.play(self.samples, self.sr)
        self.started_at = time.monotonic()

    def poll(self):
        if self._stopped.is_set() or time.monotonic() - self.started_at >= self.duration + 0.05:
            return 0
        return None

    def terminate(self) -> None:
        self._stopped.set()
        try:
            self.sd.stop()
        except Exception:
            pass

    def wait(self) -> None:
        end = self.started_at + self.duration + 0.05
        while not self._stopped.is_set() and time.monotonic() < end:
            time.sleep(0.02)
        if not self._stopped.is_set():
            try:
                self.sd.wait()
            except Exception:
                pass


def word_spans(phrase: str, offset: int, duration: float) -> list[tuple[int, int, float, float]]:
    """Estimate when each word is spoken: (char_start, char_end, t_start, t_end), characters
    relative to the whole text (offset = where the phrase starts). Time is shared out by word
    length, with extra weight after commas and full stops (where speech pauses)."""
    toks = list(re.finditer(r"\S+", phrase))
    if not toks:
        return []
    weights = []
    for m in toks:
        w = float(max(1, len(re.sub(r"\W", "", m.group(0)))) + 1.0)
        tail = m.group(0)[-1]
        if tail in ",;:":
            w += 3
        elif tail in ".!?":
            w += 6
        weights.append(w)
    total, t, out = sum(weights), 0.0, []
    for m, w in zip(toks, weights):
        dt = duration * w / total
        out.append((offset + m.start(), offset + m.end(), t, t + dt))
        t += dt
    return out
