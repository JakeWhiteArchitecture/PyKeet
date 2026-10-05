"""PyKeet: local, offline dictation + call transcription on Moondream Parakeet.

Run with:  python main.py
See README.md for setup, shortcuts and limitations.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.toml"
LOG_PATH = HERE / "pykeet.log"
TMP_DIR = HERE / "tmp"  # temporary call WAVs live here

SAMPLE_RATE = 16_000
MIN_PRESS_SECONDS = 0.3
MAX_DICTATION_SECONDS = 5 * 60
MAX_CALL_SECONDS = 2 * 60 * 60
PASTE_RESTORE_SECONDS = 0.3

log = logging.getLogger("pykeet")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

CONFIG_TEMPLATE = '''\
# PyKeet config. Edit, then restart the app.

# --- Shortcuts -------------------------------------------------------------
dictation_shortcut = "ctrl+space"
recall_shortcut = "f8"
cancel_shortcut = "esc"
call_shortcut = "ctrl+alt+r"

# "push" = hold to talk, "toggle" = press to start, press to stop
dictation_mode = "push"

# --- Speech engine ---------------------------------------------------------
model = "moondream/parakeet-redux"
# Model used for calls. Empty = same as `model`.
# e.g. "moondream/parakeet-ultra" if you have a GPU.
call_model = ""
# "cpu", "cuda" (NVIDIA) or "mps" (Apple Silicon)
device = "cpu"
# Device for call_model. Empty = same as `device`.
call_device = ""
# Microphone: empty = system default, or a device number, or part of its name.
input_device = ""

# --- Dictation -------------------------------------------------------------
cleanup = true            # strip um/uh, collapse repeats, apply replacements
insert_method = "paste"   # "paste" (clipboard + Ctrl+V) or "type"
dictation_beep = false

# --- Calls -----------------------------------------------------------------
call_beep = true
transcript_folder = "~/CallTranscripts"
keep_audio = false        # keep the call WAV next to the transcript
label_speakers = true
expected_speakers = 2     # 0 = detect automatically

# --- Misc ------------------------------------------------------------------
# [x, y] of the floating widget; empty = bottom centre. Updated when you drag it.
widget_position = []
debug = false             # true = transcript text is allowed in the log

# Spoken -> written. Case-insensitive, applied in dictation and call mode.
[replacements]
"stair smith" = "StairSmith"
"sun form" = "SunForm"
"IFC four x three" = "IFC4X3"
"part L" = "Part L"
'''

DEFAULTS = {
    "dictation_shortcut": "ctrl+space",
    "recall_shortcut": "f8",
    "cancel_shortcut": "esc",
    "call_shortcut": "ctrl+alt+r",
    "dictation_mode": "push",
    "model": "moondream/parakeet-redux",
    "call_model": "",
    "device": "cpu",
    "call_device": "",
    "input_device": "",
    "cleanup": True,
    "insert_method": "paste",
    "dictation_beep": False,
    "call_beep": True,
    "transcript_folder": "~/CallTranscripts",
    "keep_audio": False,
    "label_speakers": True,
    "expected_speakers": 2,
    "widget_position": [],
    "debug": False,
    "replacements": {},
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    cfg = dict(DEFAULTS)
    try:
        data = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg.update(data)
    except Exception:
        print(f"Could not read {CONFIG_PATH}; using defaults.", file=sys.stderr)
        logging.getLogger("pykeet").exception("config read failed")
        cfg["replacements"] = {}
    return cfg


def save_widget_position(x: int, y: int) -> None:
    """Rewrite just the widget_position line of config.toml."""
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
        line = f"widget_position = [{x}, {y}]"
        if re.search(r"^widget_position\s*=.*$", text, flags=re.M):
            text = re.sub(r"^widget_position\s*=.*$", line, text, count=1, flags=re.M)
        else:
            text = line + "\n" + text
        CONFIG_PATH.write_text(text, encoding="utf-8")
    except Exception:
        log.exception("could not save widget position")


def setup_logging(debug: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger("pykeet")
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.handlers.clear()
    fh = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

_FILLER = re.compile(r"\b(?:um+|uh+|er+m?|erm|ah+)\b[,.]?\s*", re.I)
_REPEAT = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.I)


def compile_replacements(table: dict) -> list[tuple[re.Pattern, str]]:
    out = []
    for spoken, written in sorted(table.items(), key=lambda kv: -len(kv[0])):
        words = [re.escape(w) for w in spoken.split()]
        if not words:
            continue
        pat = r"(?<!\w)" + r"[,.]?\s+".join(words) + r"(?!\w)"
        out.append((re.compile(pat, re.I), str(written)))
    return out


def apply_replacements(text: str, rules) -> str:
    for pat, written in rules:
        text = pat.sub(lambda _m, w=written: w, text)
    return text


def clean_dictation(text: str, rules, cleanup: bool) -> str:
    text = text.strip()
    if cleanup:
        text = _FILLER.sub("", text)
        text = _REPEAT.sub(r"\1", text)
    text = apply_replacements(text, rules)
    if cleanup:
        text = re.sub(r"^[\s,.;:]+", "", text).strip()
        if text:
            text = text[0].upper() + text[1:]
    text = text.strip()
    return text + " " if text else ""


def fmt_ts(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_duration(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m} min {s} s"
    if m:
        return f"{m} min {s} s"
    return f"{s} s"


# --------------------------------------------------------------------------
# Transcript building (pure functions, no hardware)
# --------------------------------------------------------------------------

def extract_units(result: dict) -> tuple[list[tuple[float, float, str]], bool]:
    """Pull (start, end, text) units out of a transcribe() result.

    Returns (units, word_level). Tolerant about the exact result layout: it
    looks for words at the top level, then words inside segments, then the
    segments themselves, then falls back to the plain text.
    """

    def to_units(items):
        out = []
        for w in items or []:
            if not isinstance(w, dict):
                continue
            text = w.get("word", w.get("text"))
            start, end = w.get("start"), w.get("end")
            if text is None or start is None:
                continue
            text = str(text).strip()
            if text:
                out.append((float(start), float(end if end is not None else start), text))
        return out

    words = to_units(result.get("words"))
    if words:
        return words, True
    segments = result.get("segments") or []
    nested: list = []
    for seg in segments:
        if isinstance(seg, dict):
            nested.extend(to_units(seg.get("words")))
    if nested:
        return nested, True
    segs = to_units(segments)
    if segs:
        return segs, False
    text = str(result.get("text", "")).strip()
    return ([(0.0, 0.0, text)] if text else []), False


def assign_speakers(units, segments):
    """Give each word the speaker of the segment holding its midpoint
    (or the nearest segment). Returns [(speaker_label, start, text)] per word."""
    starts = np.array([s[0] for s in segments], dtype=float)
    ends = np.array([s[1] for s in segments], dtype=float)
    out = []
    for start, end, text in units:
        mid = (start + end) / 2
        inside = np.nonzero((starts <= mid) & (mid <= ends))[0]
        if len(inside):
            idx = int(inside[0])
        else:
            dist = np.maximum(np.maximum(starts - mid, mid - ends), 0)
            idx = int(np.argmin(dist))
        out.append((segments[idx][2], start, text))
    return out


def build_turns(word_speakers):
    """Merge consecutive words from one speaker into turns.

    Speakers are numbered 1.. in order of first appearance.
    Returns [(start_seconds, speaker_number, text)].
    """
    numbering: dict[str, int] = {}
    turns: list[list] = []
    for speaker, start, text in word_speakers:
        num = numbering.setdefault(speaker, len(numbering) + 1)
        if turns and turns[-1][1] == num:
            turns[-1][2].append(text)
        else:
            turns.append([start, num, [text]])
    return [(t[0], t[1], " ".join(t[2])) for t in turns]


def group_lines(units, word_level: bool):
    """Unlabelled transcript: [(start, None, text)] lines."""
    if not word_level:
        return [(s, None, t) for s, _e, t in units]
    lines: list[list] = []
    prev_end = None
    for start, end, text in units:
        gap = 0 if prev_end is None else start - prev_end
        cur = lines[-1] if lines else None
        sentence_end = cur is not None and cur[2][-1].rstrip().endswith((".", "?", "!"))
        if cur is None or gap > 2.5 or (gap > 1.0 and sentence_end) or len(cur[2]) >= 60:
            lines.append([start, None, [text]])
        else:
            cur[2].append(text)
        prev_end = end
    return [(l[0], None, " ".join(l[2])) for l in lines]


def build_markdown(started: datetime, duration: float, turns, labelled: bool, rules,
                   recovered: bool = False, source: str | None = None) -> str:
    out = [
        "# Audio transcript" if source else "# Call transcript",
        f"- Date: {started:%Y-%m-%d %H:%M}",
        f"- Duration: {fmt_duration(duration)}",
        "- Contact:",
        "- Project:",
    ]
    if source:
        out.insert(1, f"- Source: {source}")
    if recovered:
        out.append("- Note: recovered after the app stopped mid-call")
    if labelled:
        out.append("")
        for n in sorted({t[1] for t in turns}):
            out.append(f"- Speaker {n}:")
    else:
        out.append("- Speakers: not labelled")
    out.append("")
    for start, speaker, text in turns:
        text = apply_replacements(text, rules)
        label = f"**Speaker {speaker}:** " if labelled else ""
        out.append(f"[{fmt_ts(start)}] {label}{text}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# WAV helpers (crash-safe: header is patched on close and repairable)
# --------------------------------------------------------------------------

def _wav_header(data_bytes: int) -> bytes:
    return (
        b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
        + b"data" + struct.pack("<I", data_bytes)
    )


def repair_wav(path: Path) -> float:
    """Fix the header of a WAV that was never closed. Returns duration in s."""
    size = path.stat().st_size
    data_bytes = max(0, size - 44) // 2 * 2
    with open(path, "r+b") as fh:
        fh.write(_wav_header(data_bytes))
    return data_bytes / 2 / SAMPLE_RATE


class WavWriter:
    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "wb")
        self._fh.write(_wav_header(0))
        self._bytes = 0
        self._last_flush = time.monotonic()

    def write(self, samples: np.ndarray) -> None:
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        self._fh.write(pcm)
        self._bytes += len(pcm)
        if time.monotonic() - self._last_flush > 2:
            self._fh.flush()
            self._last_flush = time.monotonic()

    def close(self) -> float:
        self._fh.seek(0)
        self._fh.write(_wav_header(self._bytes))
        self._fh.close()
        return self._bytes / 2 / SAMPLE_RATE


# --------------------------------------------------------------------------
# Audio capture
# --------------------------------------------------------------------------

AUDIO_TYPES = [("Audio files", "*.mp3 *.ogg *.flac *.wav"), ("All files", "*.*")]


def decode_to_wav(src: Path, dest: Path) -> float:
    """Decode mp3/ogg/flac/wav to 16 kHz mono 16-bit WAV. Returns duration in s."""
    import soundfile as sf
    from math import gcd

    from scipy.signal import resample_poly

    data, rate = sf.read(str(src), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, int(rate))
        mono = resample_poly(mono, SAMPLE_RATE // g, int(rate) // g).astype("float32")
    w = WavWriter(dest)
    try:
        for i in range(0, len(mono), SAMPLE_RATE * 60):
            w.write(mono[i:i + SAMPLE_RATE * 60])
    finally:
        duration = w.close()
    return duration


def parse_input_device(value):
    if value in ("", None):
        return None
    s = str(value).strip()
    return int(s) if s.isdigit() else s


class Recorder:
    """16 kHz mono capture. Dictation: memory only. Call: streamed to a WAV."""

    def __init__(self, input_device):
        self.input_device = parse_input_device(input_device)
        self.level = 0.0
        self.started = 0.0
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._writer: WavWriter | None = None
        self._wq: queue.Queue | None = None
        self._wthread: threading.Thread | None = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started if self._stream else 0.0

    def start(self, wav_path: Path | None = None) -> None:
        import sounddevice as sd

        self._chunks = []
        self.level = 0.0
        if wav_path is not None:
            self._writer = WavWriter(wav_path)
            self._wq = queue.Queue()
            self._wthread = threading.Thread(target=self._drain, daemon=True)
            self._wthread.start()

        def callback(indata, frames, time_info, status):
            mono = indata[:, 0].copy()
            self.level = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
            if self._wq is not None:
                self._wq.put(mono)
            else:
                self._chunks.append(mono)

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                device=self.input_device, callback=callback,
            )
            self._stream.start()
        except Exception:
            self._stream = None
            self._close_writer()
            raise
        self.started = time.monotonic()

    def _drain(self) -> None:
        assert self._wq is not None and self._writer is not None
        while True:
            chunk = self._wq.get()
            if chunk is None:
                return
            try:
                self._writer.write(chunk)
            except Exception:
                log.exception("writing call audio failed")

    def _close_writer(self) -> float:
        duration = 0.0
        if self._wq is not None and self._wthread is not None:
            self._wq.put(None)
            self._wthread.join()
        if self._writer is not None:
            duration = self._writer.close()
        self._writer = self._wq = self._wthread = None
        return duration

    def stop(self):
        """Stop capture. Dictation: returns the samples. Call: returns duration."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.exception("closing input stream failed")
        self.level = 0.0
        if self._writer is not None:
            return self._close_writer()
        chunks, self._chunks = self._chunks, []
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def beep(high: bool) -> None:
    try:
        import sounddevice as sd

        freq, n = (880, 0.12) if high else (560, 0.12)
        t = np.arange(int(44100 * n)) / 44100
        tone = (0.25 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))).astype("float32")
        sd.play(tone, 44100)
    except Exception:
        log.debug("beep failed", exc_info=True)


# --------------------------------------------------------------------------
# Speech engine (loaded once, shared by both modes)
# --------------------------------------------------------------------------

class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._stack = ExitStack()
        self._lock = threading.Lock()  # one transcription at a time
        self._open_lock = threading.Lock()
        self.main = None
        self.call = None

    def _open(self, model: str, device: str):
        import moondream as md

        log.info("loading model %s on %s", model, device)
        t = time.monotonic()
        speech = self._stack.enter_context(md.photon(model, device=device))
        log.info("model loaded in %.1fs", time.monotonic() - t)
        return speech

    def load(self) -> None:
        self.main = self._open(self.cfg["model"], self.cfg["device"])

    def for_call(self):
        model = self.cfg["call_model"] or self.cfg["model"]
        device = self.cfg["call_device"] or self.cfg["device"]
        if model == self.cfg["model"] and device == self.cfg["device"]:
            return self.main
        with self._open_lock:
            if self.call is None:
                try:
                    self.call = self._open(model, device)
                except Exception:
                    log.exception("call_model failed to load; using main model")
                    self.call = self.main
            return self.call

    def transcribe(self, speech, **kwargs) -> dict:
        with self._lock:
            return speech.transcribe(**kwargs)

    def close(self) -> None:
        try:
            self._stack.close()
        except Exception:
            log.exception("closing model failed")


# --------------------------------------------------------------------------
# OS helpers: opening files, notifications, inserting text
# --------------------------------------------------------------------------

def open_path(path) -> None:
    path = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        log.exception("could not open %s", path)


def insert_text(text: str, method: str) -> None:
    from pynput.keyboard import Controller, Key

    kb = Controller()
    if method == "type":
        kb.type(text)
        return
    import pyperclip

    try:
        old = pyperclip.paste()
    except Exception:
        old = None
    pyperclip.copy(text)
    time.sleep(0.05)
    mod = Key.cmd if sys.platform == "darwin" else Key.ctrl
    with kb.pressed(mod):
        kb.press("v")
        kb.release("v")

    if old is not None:
        def restore():
            try:
                pyperclip.copy(old)
            except Exception:
                log.exception("clipboard restore failed")

        threading.Timer(PASTE_RESTORE_SECONDS, restore).start()


# --------------------------------------------------------------------------
# Hotkeys
# --------------------------------------------------------------------------

_MOD_ALIASES = {"control": "ctrl", "option": "alt", "win": "cmd", "super": "cmd",
                "meta": "cmd", "escape": "esc"}


def parse_shortcut(spec: str) -> tuple[frozenset, str]:
    parts = [_MOD_ALIASES.get(p.strip().lower(), p.strip().lower()) for p in spec.split("+")]
    mods = frozenset(p for p in parts if p in ("ctrl", "alt", "shift", "cmd"))
    mains = [p for p in parts if p not in mods]
    if len(mains) != 1:
        raise ValueError(f"bad shortcut {spec!r}")
    return mods, mains[0]


def key_name(key) -> str | None:
    """Normalise a pynput key to 'ctrl', 'space', 'r', 'f8', ..."""
    name = getattr(key, "name", None)
    if name:
        for base in ("ctrl", "alt", "shift", "cmd"):
            if name.startswith(base):
                return base
        return name
    ch = getattr(key, "char", None)
    if ch and ch.isprintable():
        return ch.lower()
    vk = getattr(key, "vk", None)
    if vk and 65 <= vk <= 90:
        return chr(vk).lower()
    return None


def name_to_vk(name: str) -> int | None:
    fixed = {"space": 0x20, "esc": 0x1B, "enter": 0x0D, "tab": 0x09}
    if name in fixed:
        return fixed[name]
    if re.fullmatch(r"f([1-9]|1\d|2[0-4])", name):
        return 0x6F + int(name[1:])
    if len(name) == 1 and name.isalnum():
        return ord(name.upper())
    return None


class Hotkeys:
    def __init__(self, app: "App"):
        self.app = app
        cfg = app.cfg
        self.dictation = parse_shortcut(cfg["dictation_shortcut"])
        self.recall = parse_shortcut(cfg["recall_shortcut"])
        self.cancel = parse_shortcut(cfg["cancel_shortcut"])
        self.call = parse_shortcut(cfg["call_shortcut"])
        self.down: set[str] = set()
        self.last_press: dict[str, float] = {}
        self.ptt_active = False
        self.listener = None
        self._suppressed: set[int] = set()

    # -- state tracking (idempotent, so it is safe if both the Windows filter
    #    and the normal callbacks deliver the same event) ---------------------
    def press(self, name: str) -> None:
        now = time.monotonic()
        if name in self.down and now - self.last_press.get(name, 0) < 1.0:
            self.last_press[name] = now
            return  # key auto-repeat
        self.down.add(name)
        self.last_press[name] = now
        others = self.down - {name}

        mods, main = self.dictation
        if name == main and mods <= others and self.app.dictation_enabled:
            self.ptt_active = True
            self.app.post(self.app.on_dictation_press)
            return
        mods, main = self.call
        if name == main and mods <= others:
            self.app.post(self.app.toggle_call)
            return
        mods, main = self.recall
        if name == main and mods <= others:
            self.app.post(self.app.recall)
            return
        mods, main = self.cancel
        if name == main and mods <= others:
            self.app.post(self.app.cancel_dictation)

    def release(self, name: str) -> None:
        if name not in self.down:
            return
        self.down.discard(name)
        mods, main = self.dictation
        if self.ptt_active and (name == main or name in mods):
            self.ptt_active = False
            self.app.post(self.app.on_dictation_release)

    def _on_press(self, key):
        n = key_name(key)
        if n:
            self.press(n)

    def _on_release(self, key):
        n = key_name(key)
        if n:
            self.release(n)

    # -- Windows: swallow the hotkeys so the focused app never sees them ------
    def _win32_filter(self, msg, data):
        try:
            if data.flags & 0x10:  # injected by us (paste) - leave alone
                return True
            vk = data.vkCode
            is_down = msg in (0x100, 0x104)
            is_up = msg in (0x101, 0x105)
            if is_up and vk in self._suppressed:
                self._suppressed.discard(vk)
                self.release(self._vk_name(vk))
                self.listener.suppress_event()
            if not is_down:
                return True
            import ctypes

            get = ctypes.windll.user32.GetAsyncKeyState  # type: ignore[attr-defined]
            held = {m for m, code in (("ctrl", 0x11), ("alt", 0x12),
                                      ("shift", 0x10), ("cmd", 0x5B)) if get(code) & 0x8000}
            for spec, enabled in ((self.dictation, self.app.dictation_enabled),
                                  (self.call, True)):
                mods, main = spec
                if enabled and name_to_vk(main) == vk and mods <= held:
                    for m in ("ctrl", "alt", "shift", "cmd"):  # resync stale modifiers
                        (self.down.add if m in held else self.down.discard)(m)
                    self._suppressed.add(vk)
                    self.press(main)
                    self.listener.suppress_event()
        except Exception as exc:
            if type(exc).__name__ == "SuppressException":
                raise  # pynput's way of swallowing the key system-wide
            log.exception("hotkey filter error")
        return True

    @staticmethod
    def _vk_name(vk: int) -> str:
        for spec in (("space", 0x20), ("esc", 0x1B)):
            if spec[1] == vk:
                return spec[0]
        if 0x70 <= vk <= 0x87:
            return f"f{vk - 0x6F}"
        return chr(vk).lower()

    def start(self) -> None:
        if os.environ.get("XDG_SESSION_TYPE") == "wayland":
            log.warning("Wayland session: global hotkeys will not work. Use an X11 session.")
        try:
            from pynput import keyboard

            kwargs = {}
            if sys.platform.startswith("win"):
                kwargs["win32_event_filter"] = self._win32_filter
            else:
                log.warning("This platform cannot consume hotkeys: the focused app will "
                            "also see %s. Pick a different dictation_shortcut if it clashes.",
                            self.app.cfg["dictation_shortcut"])
            self.listener = keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release, **kwargs)
            self.listener.daemon = True
            self.listener.start()
            log.info("hotkeys registered")
        except Exception:
            log.exception("WARNING: could not register hotkeys")

    def stop(self) -> None:
        if self.listener:
            self.listener.stop()


# --------------------------------------------------------------------------
# Floating widget (tkinter, main thread)
# --------------------------------------------------------------------------

class Widget:
    W, H = 290, 46
    KEY = "#010101"  # transparent colour on Windows

    def __init__(self, app: "App"):
        import tkinter as tk

        self.app = app
        self.tk = tk
        self.q: queue.Queue = queue.Queue()
        self.root = tk.Tk()
        self.root.title("PyKeet")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.9)
        except tk.TclError:
            pass
        bg = "#1e1e24"
        self.canvas = tk.Canvas(self.root, width=self.W, height=self.H,
                                highlightthickness=0, bg=bg)
        if sys.platform.startswith("win"):
            self.canvas.configure(bg=self.KEY)
            self.root.attributes("-transparentcolor", self.KEY)
        self.canvas.pack()
        self.bg = bg
        self.state = "hidden"
        self.t0 = 0.0
        self.state_since = 0.0
        self.levels: deque = deque([0.0] * 10, maxlen=10)
        self._place()
        self.root.withdraw()
        self._no_focus()
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", self._right_click)
        self._drag_origin = None
        self.root.after(40, self._poll)

    # -- window plumbing -------------------------------------------------------
    def _place(self) -> None:
        pos = self.app.cfg.get("widget_position") or []
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        if len(pos) == 2:
            x, y = int(pos[0]), int(pos[1])
        else:
            x, y = (sw - self.W) // 2, sh - self.H - 90
        self.root.geometry(f"{self.W}x{self.H}+{x}+{y}")

    def _no_focus(self) -> None:
        if not sys.platform.startswith("win"):
            return  # unmanaged (override-redirect) windows do not take focus on X11
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            self.root.update_idletasks()
            hwnd = user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            GWL_EXSTYLE, NOACTIVATE, TOOLWINDOW = -20, 0x08000000, 0x00000080
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | NOACTIVATE | TOOLWINDOW)
        except Exception:
            log.exception("could not set no-activate window style")

    # -- input -------------------------------------------------------------------
    def _press(self, e):
        self._drag_origin = (e.x_root, e.y_root, self.root.winfo_x(), self.root.winfo_y())
        self._moved = False

    def _drag(self, e):
        if not self._drag_origin:
            return
        ox, oy, wx, wy = self._drag_origin
        dx, dy = e.x_root - ox, e.y_root - oy
        if abs(dx) + abs(dy) > 4:
            self._moved = True
        if self._moved:
            self.root.geometry(f"+{wx + dx}+{wy + dy}")

    def _release(self, e):
        if self._moved:
            save_widget_position(self.root.winfo_x(), self.root.winfo_y())
        elif self.state == "rec":
            self.app.post(self.app.stop_dictation)
        self._drag_origin = None

    def _right_click(self, _e):
        if self.state == "rec":
            self.app.post(self.app.cancel_dictation)

    # -- state ---------------------------------------------------------------------
    def send(self, state: str) -> None:
        """Thread-safe: queue a state change."""
        self.q.put(state)

    def _poll(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple):  # ("text", title, body, path)
                    self._show_text(*item[1:])
                else:
                    self._set(item)
        except queue.Empty:
            pass
        now = time.monotonic()
        if self.state == "done" and now - self.state_since > 0.5:
            self._set("hidden")
        elif self.state == "nospeech" and now - self.state_since > 1.0:
            self._set("hidden")
        if self.state != "hidden":
            self._draw()
        self.root.after(50, self._poll)

    def _set(self, state: str) -> None:
        if state == "quit":
            self.root.quit()
            return
        if state == "pick_file":
            self._pick_file()
            return
        self.state = state
        self.state_since = time.monotonic()
        if state in ("rec", "call"):
            self.t0 = time.monotonic()
        if state == "hidden":
            self.root.withdraw()
        else:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self._draw()

    def _show_text(self, title: str, body: str, path) -> None:
        """Result window: scrollable transcript with Copy / Open file buttons."""
        tk = self.tk
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("760x560")
        bar = tk.Frame(win)
        bar.pack(side="bottom", fill="x", padx=8, pady=6)
        frame = tk.Frame(win)
        frame.pack(side="top", fill="both", expand=True, padx=8, pady=(8, 0))
        text = tk.Text(frame, wrap="word", font=("Consolas" if sys.platform.startswith("win")
                                                 else "TkFixedFont", 11), undo=False)
        scroll = tk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.insert("1.0", body)

        def copy():
            win.clipboard_clear()
            win.clipboard_append(text.get("1.0", "end-1c"))

        tk.Button(bar, text="Copy all", command=copy).pack(side="left")
        if path:
            tk.Button(bar, text="Open file", command=lambda: open_path(path)).pack(
                side="left", padx=6)
            tk.Label(bar, text=str(path), anchor="w", fg="#777").pack(side="left", padx=6)
        tk.Button(bar, text="Close", command=win.destroy).pack(side="right")
        win.lift()
        win.focus_force()

    def _pick_file(self) -> None:
        from tkinter import filedialog

        paths = filedialog.askopenfilenames(title="Choose audio files to transcribe",
                                            filetypes=AUDIO_TYPES)
        if paths:
            self.app.post(lambda: self.app.import_files(paths))

    def _pill(self, fill: str) -> None:
        c, w, h = self.canvas, self.W, self.H
        c.create_oval(0, 0, h, h, fill=fill, outline=fill)
        c.create_oval(w - h, 0, w, h, fill=fill, outline=fill)
        c.create_rectangle(h / 2, 0, w - h / 2, h, fill=fill, outline=fill)

    def _bars(self, x0: int) -> None:
        self.levels.append(min(1.0, (self.app.recorder.level * 9) ** 0.6))
        for i, lv in enumerate(self.levels):
            bh = 4 + lv * 26
            x = x0 + i * 9
            self.canvas.create_rectangle(x, self.H / 2 - bh / 2, x + 5, self.H / 2 + bh / 2,
                                         fill="#e8e8ee", outline="")

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        self._pill(self.bg)
        now = time.monotonic()
        st = self.state
        font = ("Segoe UI", 11, "bold") if sys.platform.startswith("win") else ("Helvetica", 11, "bold")
        mid = self.H / 2

        def dot(color):
            c.create_oval(18, mid - 7, 32, mid + 7, fill=color, outline=color)

        def text(x, s, color="#f2f2f5"):
            c.create_text(x, mid, text=s, fill=color, font=font, anchor="w")

        if st == "rec":
            dot("#ff4040")
            self._bars(44)
            text(150, fmt_ts(now - self.t0))
        elif st == "call":
            if int(now * 2) % 2 == 0:
                dot("#ff4040")
            text(40, "CALL")
            self._bars(100)
            text(200, fmt_ts(now - self.t0))
        elif st == "transcribing":
            dot("#ffb020")
            text(44, "Transcribing…")
        elif st in ("call_transcribing", "call_labelling"):
            frames = "|/-\\"
            spin = frames[int(now * 8) % 4]
            dot("#ffb020")
            text(44, ("Transcribing call… " if st == "call_transcribing"
                      else "Labelling speakers… ") + spin)
        elif st == "done":
            dot("#40d070")
            text(44, "Done", "#40d070")
        elif st == "nospeech":
            dot("#8a8a92")
            text(44, "No speech", "#b0b0b8")

    def run(self) -> None:
        self.root.mainloop()


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

class App:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rules = compile_replacements(cfg.get("replacements") or {})
        self.engine = Engine(cfg)
        self.recorder = Recorder(cfg["input_device"])
        self.widget: Widget | None = None
        self.tray = None
        self.hotkeys: Hotkeys | None = None
        self.lock = threading.RLock()
        self.mode = "idle"       # "idle" | "dictation" | "call"
        self.phase = "none"      # "recording" | "processing" | "none"
        self.paused = False
        self.ready = False
        self.last_text = ""
        self.last_transcript: Path | None = None
        self._timer: threading.Timer | None = None
        self._actions: queue.Queue = queue.Queue()
        self._call_started: datetime | None = None
        self._call_wav: Path | None = None
        self._quitting = False
        threading.Thread(target=self._action_loop, daemon=True).start()

    # -- plumbing -------------------------------------------------------------
    def post(self, fn) -> None:
        self._actions.put(fn)

    def _action_loop(self) -> None:
        while True:
            fn = self._actions.get()
            try:
                fn()
            except Exception:
                log.exception("action failed")

    def ui(self, state: str) -> None:
        if self.widget:
            self.widget.send(state)
        self._refresh_tray()

    @property
    def dictation_enabled(self) -> bool:
        return self.ready and not self.paused and self.mode != "call"

    def _set_idle(self) -> None:
        with self.lock:
            self.mode, self.phase = "idle", "none"
        self._refresh_tray()

    def _arm_timer(self, seconds: float, fn) -> None:
        self._cancel_timer()
        self._timer = threading.Timer(seconds, lambda: self.post(fn))
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    # -- startup ------------------------------------------------------------------
    def load_in_background(self) -> None:
        def work():
            try:
                self.engine.load()
            except Exception:
                log.exception("could not load speech model")
                self.notify("PyKeet", "Could not load the speech model. See pykeet.log.")
                return
            try:
                self.recover_calls()
            except Exception:
                log.exception("crash recovery failed")
            self.ready = True
            log.info("ready")
            self.notify("PyKeet", "Ready.")

        threading.Thread(target=work, daemon=True).start()

    def recover_calls(self) -> None:
        TMP_DIR.mkdir(exist_ok=True)
        for wav in sorted(TMP_DIR.glob("call_*.wav")):
            if wav == self._call_wav:
                continue
            log.info("recovering leftover call audio %s", wav.name)
            try:
                started = datetime.strptime(wav.stem[len("call_"):], "%Y%m%d_%H%M%S")
            except ValueError:
                started = datetime.fromtimestamp(wav.stat().st_mtime)
            duration = repair_wav(wav)
            if duration < 1:
                wav.unlink(missing_ok=True)
                continue
            with self.lock:
                self.mode, self.phase = "call", "processing"
            self.process_call(wav, started, duration, recovered=True)

    # -- dictation --------------------------------------------------------------
    def on_dictation_press(self) -> None:
        if self.cfg["dictation_mode"] == "toggle":
            if self.mode == "dictation" and self.phase == "recording":
                self.stop_dictation()
            else:
                self.start_dictation()
        else:
            self.start_dictation()

    def on_dictation_release(self) -> None:
        if self.cfg["dictation_mode"] == "push":
            self.stop_dictation()

    def start_dictation(self) -> None:
        with self.lock:
            if not self.dictation_enabled or self.mode != "idle":
                return
            try:
                self.recorder.start()
            except Exception:
                log.exception("could not start microphone")
                self.ui("nospeech")
                return
            self.mode, self.phase = "dictation", "recording"
        if self.cfg["dictation_beep"]:
            beep(True)
        self.ui("rec")
        self._arm_timer(MAX_DICTATION_SECONDS, self.stop_dictation)
        log.info("dictation: recording started")

    def cancel_dictation(self) -> None:
        self._end_dictation_recording(commit=False)

    def stop_dictation(self) -> None:
        self._end_dictation_recording(commit=True)

    def _end_dictation_recording(self, commit: bool) -> None:
        with self.lock:
            if self.mode != "dictation" or self.phase != "recording":
                return
            self.phase = "processing"
        self._cancel_timer()
        elapsed = self.recorder.elapsed
        audio = self.recorder.stop()
        if self.cfg["dictation_beep"]:
            beep(False)
        if not commit or elapsed < MIN_PRESS_SECONDS:
            log.info("dictation: discarded (%.2fs, commit=%s)", elapsed, commit)
            self._set_idle()
            self.ui("hidden")
            return
        self.ui("transcribing")
        threading.Thread(target=self._dictation_worker, args=(audio, elapsed),
                         daemon=True).start()

    def _dictation_worker(self, audio: np.ndarray, elapsed: float) -> None:
        t = time.monotonic()
        try:
            result = self.engine.transcribe(self.engine.main, audio=audio,
                                            sample_rate=SAMPLE_RATE, timestamps="none")
            raw = str(result.get("text", ""))
            text = clean_dictation(raw, self.rules, self.cfg["cleanup"])
            log.info("dictation: %.1fs audio transcribed in %.2fs (%d chars)",
                     elapsed, time.monotonic() - t, len(text))
            if self.cfg["debug"]:
                log.debug("dictation text: %r", text)
            if not text.strip():
                self.ui("nospeech")
                return
            self.last_text = text
            insert_text(text, self.cfg["insert_method"])
            self.ui("done")
        except Exception:
            log.exception("dictation failed")
            self.ui("nospeech")
        finally:
            self._set_idle()

    def recall(self) -> None:
        if self.last_text and self.mode == "idle":
            try:
                insert_text(self.last_text, self.cfg["insert_method"])
            except Exception:
                log.exception("recall failed")

    # -- call mode ----------------------------------------------------------------
    def toggle_call(self) -> None:
        if self.mode == "call" and self.phase == "recording":
            self.stop_call()
        elif self.mode == "idle":
            self.start_call()

    def start_call(self) -> None:
        with self.lock:
            if not self.ready or self.mode != "idle":
                return
            TMP_DIR.mkdir(exist_ok=True)
            self._call_started = datetime.now()
            self._call_wav = TMP_DIR / f"call_{self._call_started:%Y%m%d_%H%M%S}.wav"
            try:
                self.recorder.start(self._call_wav)
            except Exception:
                log.exception("could not start microphone")
                self._call_wav.unlink(missing_ok=True)
                self._call_wav = None
                self.ui("nospeech")
                return
            self.mode, self.phase = "call", "recording"
        if self.cfg["call_beep"]:
            beep(True)
        self.ui("call")
        self._arm_timer(MAX_CALL_SECONDS, self.stop_call)
        log.info("call: recording started")

    def stop_call(self) -> None:
        with self.lock:
            if self.mode != "call" or self.phase != "recording":
                return
            self.phase = "processing"
        self._cancel_timer()
        duration = self.recorder.stop()
        if self.cfg["call_beep"]:
            beep(False)
        log.info("call: recording stopped after %.0fs", duration)
        wav, started = self._call_wav, self._call_started
        assert wav is not None and started is not None
        threading.Thread(target=self.process_call, args=(wav, started, duration),
                         daemon=True).start()

    def run_diarize(self, wav: Path, duration: float):
        import diarize  # lazy: only loaded on first call-mode use

        n = int(self.cfg["expected_speakers"] or 0)
        box: dict = {}

        def work():
            try:
                box["r"] = diarize.diarize(str(wav), num_speakers=n or None)
            except BaseException as exc:  # noqa: BLE001
                box["e"] = exc

        th = threading.Thread(target=work, daemon=True)
        th.start()
        # 2x call length, with a floor so short calls have time to load the models
        th.join(max(2 * duration, 120))
        if th.is_alive():
            raise TimeoutError("diarisation timed out")
        if "e" in box:
            raise box["e"]
        return [(s.start, s.end, s.speaker) for s in box["r"].segments]

    def process_call(self, wav: Path, started: datetime, duration: float,
                     recovered: bool = False, source: str | None = None,
                     keep_busy: bool = False) -> None:
        t0 = time.monotonic()
        try:
            self.ui("call_transcribing")
            want_labels = bool(self.cfg["label_speakers"])
            speech = self.engine.for_call()
            result = self.engine.transcribe(
                speech, audio=str(wav), timestamps="word" if want_labels else "segment")
            units, word_level = extract_units(result)
            log.info("call: transcribed %.0fs in %.1fs (%d units)", duration,
                     time.monotonic() - t0, len(units))

            turns = None
            if want_labels and word_level and units:
                try:
                    self.ui("call_labelling")
                    t1 = time.monotonic()
                    segments = self.run_diarize(wav, duration)
                    if not segments:
                        raise RuntimeError("diarisation returned no segments")
                    turns = build_turns(assign_speakers(units, segments))
                    if len({t[1] for t in turns}) < 2 and duration > 60:
                        log.warning("diarisation found one speaker in a %.0fs call; "
                                    "saving unlabelled", duration)
                        turns = None
                    log.info("call: diarisation took %.1fs", time.monotonic() - t1)
                except Exception:
                    log.exception("speaker labelling failed; saving unlabelled transcript")
                    turns = None
            labelled = turns is not None
            if turns is None:
                turns = group_lines(units, word_level)

            md = build_markdown(started, duration, turns, labelled, self.rules, recovered, source)
            path = self._save_transcript(started, md, source)
            self._dispose_wav(wav, path, source)
            self.last_transcript = path
            log.info("call: saved %s (labelled=%s) total %.1fs", path.name, labelled,
                     time.monotonic() - t0)
            self.notify("Transcript saved", path.name, path)
            if source and self.widget:
                self.widget.q.put(("text", f"Transcript: {source}", md, path))
        except Exception:
            log.exception("call processing failed; audio kept at %s", wav)
            if source:  # the original file is untouched; drop our decoded copy
                wav.unlink(missing_ok=True)
                self.notify("PyKeet", f"Could not transcribe {source}. See pykeet.log.")
            else:
                self.notify("PyKeet", f"Call transcription failed. Audio kept: {wav.name}")
        finally:
            if not keep_busy:
                self._set_idle()
            self.ui("hidden")

    # -- audio file import ----------------------------------------------------------
    def request_file_picker(self) -> None:
        if self.ready and self.mode == "idle" and self.widget:
            self.widget.send("pick_file")  # the dialog must open on the tkinter thread
        else:
            self.notify("PyKeet", "Busy or still loading. Try again in a moment.")

    def import_files(self, paths) -> None:
        with self.lock:
            if not self.ready or self.mode != "idle":
                return
            self.mode, self.phase = "call", "processing"  # blocks recording meanwhile
        threading.Thread(target=self._import_worker, args=(list(paths),), daemon=True).start()

    def _import_worker(self, paths) -> None:
        try:
            TMP_DIR.mkdir(exist_ok=True)
            for i, raw in enumerate(paths):
                src = Path(raw)
                wav = TMP_DIR / f"import_{datetime.now():%Y%m%d_%H%M%S}_{i}.wav"
                try:
                    self.ui("call_transcribing")
                    duration = decode_to_wav(src, wav)
                except Exception:
                    log.exception("could not read audio file %s", src.name)
                    wav.unlink(missing_ok=True)
                    self.notify("PyKeet", f"Could not read {src.name}. Is it a valid audio file?")
                    continue
                if duration < 1:
                    wav.unlink(missing_ok=True)
                    self.notify("PyKeet", f"{src.name} has no audio.")
                    continue
                try:
                    started = datetime.fromtimestamp(src.stat().st_mtime)
                except OSError:
                    started = datetime.now()
                log.info("importing %s (%.0fs)", src.name, duration)
                self.process_call(wav, started, duration, source=src.name, keep_busy=True)
        finally:
            self._set_idle()
            self.ui("hidden")

    def _save_transcript(self, started: datetime, md: str, source: str | None = None) -> Path:
        folder = Path(os.path.expanduser(self.cfg["transcript_folder"]))
        folder.mkdir(parents=True, exist_ok=True)
        tag = re.sub(r"[^\w.-]+", "_", Path(source).stem)[:40] if source else "call"
        path = folder / f"{started:%Y-%m-%d_%H%M}_{tag}.md"
        n = 2
        while path.exists():
            path = folder / f"{started:%Y-%m-%d_%H%M}_{tag}_{n}.md"
            n += 1
        path.write_text(md, encoding="utf-8")
        return path

    def _dispose_wav(self, wav: Path, transcript: Path, source: str | None = None) -> None:
        try:
            if self.cfg["keep_audio"] and not source:
                shutil.move(str(wav), str(transcript.with_suffix(".wav")))
            else:
                wav.unlink(missing_ok=True)
        except Exception:
            log.exception("could not remove/move temp WAV")

    # -- tray / notifications -------------------------------------------------------
    def notify(self, title: str, message: str, open_file: Path | None = None) -> None:
        def work():
            try:
                if sys.platform.startswith("linux") and shutil.which("notify-send"):
                    cmd = ["notify-send", title, message]
                    if open_file:
                        cmd = ["notify-send", "--action=open=Open", "--wait", "-t", "10000",
                               title, message]
                    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if open_file and out.stdout.strip() == "open":
                        open_path(open_file)
                elif self.tray is not None:
                    self.tray.notify(message, title)
            except Exception:
                log.debug("notification failed", exc_info=True)

        threading.Thread(target=work, daemon=True).start()

    def _refresh_tray(self) -> None:
        if self.tray is not None:
            try:
                self.tray.update_menu()
            except Exception:
                pass

    def start_tray(self) -> None:
        try:
            import pystray
            from PIL import Image, ImageDraw
        except Exception:
            log.warning("pystray/Pillow not available; running without a tray icon")
            return

        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((4, 4, 60, 60), fill=(30, 30, 36, 255))
        d.rounded_rectangle((24, 12, 40, 38), radius=8, fill=(240, 240, 245, 255))
        d.arc((16, 24, 48, 50), 0, 180, fill=(240, 240, 245, 255), width=3)
        d.line((32, 50, 32, 56), fill=(240, 240, 245, 255), width=3)

        def call_label(_item):
            return "Stop call" if self.mode == "call" and self.phase == "recording" \
                else "Start call"

        menu = pystray.Menu(
            pystray.MenuItem(call_label, lambda: self.post(self.toggle_call)),
            pystray.MenuItem("Transcribe audio file…",
                             lambda: self.post(self.request_file_picker)),
            pystray.MenuItem("Pause dictation", self._toggle_pause,
                             checked=lambda _i: self.paused),
            pystray.MenuItem("Show last transcript", lambda: self.post(self.show_last)),
            pystray.MenuItem("Open call transcripts", lambda: self.post(self.open_folder)),
            pystray.MenuItem("Open config", lambda: open_path(CONFIG_PATH)),
            pystray.MenuItem("Quit", lambda: self.post(self.quit)),
        )
        self.tray = pystray.Icon("pykeet", img, "PyKeet", menu)
        self.tray.run_detached()

    def _toggle_pause(self, *_a) -> None:
        self.paused = not self.paused
        log.info("dictation %s", "paused" if self.paused else "resumed")

    def show_last(self) -> None:
        if self.last_transcript and self.last_transcript.exists():
            open_path(self.last_transcript)
            return
        folder = Path(os.path.expanduser(self.cfg["transcript_folder"]))
        files = sorted(folder.glob("*_call*.md")) if folder.exists() else []
        if files:
            open_path(files[-1])
        else:
            self.notify("PyKeet", "No call transcripts yet.")

    def open_folder(self) -> None:
        folder = Path(os.path.expanduser(self.cfg["transcript_folder"]))
        folder.mkdir(parents=True, exist_ok=True)
        open_path(folder)

    def quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        log.info("quitting")
        self._cancel_timer()
        if self.mode == "call" and self.phase == "recording":
            self.recorder.stop()  # temp WAV stays; it is transcribed on next start
        elif self.mode == "dictation" and self.phase == "recording":
            self.recorder.stop()
        if self.hotkeys:
            self.hotkeys.stop()
        if self.tray:
            self.tray.stop()
        self.engine.close()
        self.ui("quit")


def main() -> None:
    cfg = load_config()
    setup_logging(bool(cfg["debug"]))
    log.info("PyKeet starting (python %s)", sys.version.split()[0])
    try:
        import tkinter  # noqa: F401
    except ImportError:
        sys.exit("tkinter is missing. On Debian/Ubuntu: sudo apt install python3-tk")

    app = App(cfg)
    app.widget = Widget(app)
    app.hotkeys = Hotkeys(app)
    app.hotkeys.start()
    app.start_tray()
    app.load_in_background()
    try:
        app.widget.run()  # tkinter must own the main thread
    except KeyboardInterrupt:
        app.quit()


if __name__ == "__main__":
    main()
