# PyKeet

Local, offline dictation and call transcription, built on Moondream Parakeet.
No cloud, no account, no word limits.

- **Dictation:** hold `Ctrl+Space`, speak, release. Cleaned-up text appears at the cursor in whatever app has focus.
- **Call mode:** press `Ctrl+Alt+R` at the start of a speakerphone call and again at the end. A timestamped, speaker-labelled Markdown transcript is saved to `~/CallTranscripts/`.

> **Status:** the pure logic (text cleanup, replacements, speaker assignment, transcript format,
> crash-recovery of WAVs) is tested. The hotkeys, widget, tray, microphone and model calls need a real
> desktop, so run the acceptance tests in the spec on your own machine before relying on it.
> One assumption to check first: the exact layout of Parakeet's `timestamps="word"` result
> (`extract_units()` in `main.py` accepts words at top level, words inside segments, or plain segments).

## Install

Python 3.10+ (moondream needs 3.10 or newer).

```
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate

# CPU only (recommended unless you have an NVIDIA GPU): install the small CPU build of PyTorch first.
pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

pip install --no-cache-dir -r requirements.txt            # core: dictation + calls
pip install --no-cache-dir -r requirements-optional.txt   # tray icon, file import, speaker labels
python main.py
```

**Disk space:** with the CPU-only PyTorch the whole install is roughly 2-3 GB. Skipping the first
`pip install` line (the default PyTorch) pulls in about 3 GB of NVIDIA GPU libraries and needs
8-10 GB free. Only do that if you have an NVIDIA GPU and want `device = "cuda"`. Check space with
`df -h .`, delete old copies of this folder (each has its own `.venv`), and run `pip cache purge`
if it is tight.

The two requirements files are separate on purpose: if an optional package (usually `diarize`)
fails to install, the core app still works. At startup PyKeet logs which packages are
missing and the exact `pip` command for the Python it is running. Always run `python main.py` from
the same terminal/venv you installed into.

Linux also needs system packages. Debian/Ubuntu: `sudo apt install python3-tk xclip libportaudio2 libnotify-bin`.
Fedora: `sudo dnf install python3-tkinter xclip portaudio libnotify`.
`config.toml` is created next to `main.py` on first run. The first start loads the model (a download
the first time, then fully offline); the tray shows "Ready." when it is done.

**Quit Vibe Typer first.** Both apps want `Ctrl+Space`.

## Using it

| Shortcut | What it does |
| --- | --- |
| hold `Ctrl+Space` | Record; release to transcribe and insert (taps under 0.3 s are ignored) |
| `F8` | Re-insert the last dictation |
| `Esc` while recording | Discard |
| `Ctrl+Alt+R` | Start / stop call recording |
| `Ctrl+Alt+O` | Open the audio file picker (same as the tray item) |

Widget: click while dictating = stop and insert, right-click = cancel. It is draggable and remembers its
position. During a call it shows "CALL" and ignores clicks; stop with the shortcut or the tray menu.
While a call is being recorded the dictation shortcut is disabled.

Tray menu: Start/Stop call, Transcribe audio file…, Pause dictation, Show last transcript (opens the latest call transcript),
Open call transcripts, Open config, Quit.

## Config

Everything is in `config.toml` (commented). Highlights: shortcuts, `dictation_mode = "push" | "toggle"`,
`model` / `call_model` / `device` (`cpu`, `cuda`, `mps`), `input_device`, `cleanup`, `insert_method`
(`paste` or `type`), `transcript_folder`, `keep_audio`, `label_speakers`, `expected_speakers`, and the
`[replacements]` table (case-insensitive, both modes).

For a GPU, set `call_model = "moondream/parakeet-ultra"` and `call_device = "cuda"` (it is loaded the
first time you record a call).

## Transcribing an audio file

Tray menu > **Transcribe audio file…** (or `Ctrl+Alt+O`, handy if your desktop has no tray) opens a file picker for `.mp3`, `.ogg`, `.flac` and `.wav` (pick
several to batch them). Each file is decoded to 16 kHz mono, transcribed with the call settings
(speaker labelling included, same fallbacks), and saved in your transcript folder as
`YYYY-MM-DD_HHMM_<filename>.md`. When it finishes, a **text window** shows the transcript with
**Copy all** and **Open file** buttons. Your original file is never modified or deleted. It is
unavailable while a recording or another job is running. MP3 needs `soundfile` 0.12+ (included in
`requirements.txt`).

## Call transcripts

`YYYY-MM-DD_HHMM_call.md`, with blank Contact / Project / speaker-name lines to fill in by hand.
During a call the audio is streamed to `tmp/call_*.wav` so a crash loses nothing; after the transcript
is saved the WAV is deleted (or moved next to the transcript if `keep_audio = true`). If the app dies
mid-call, the leftover WAV is transcribed and saved at next start.

Speaker labels use the `diarize` package, loaded on first call. If it fails, times out
(2x the call length, minimum 2 minutes) or finds one speaker in a call over a minute, you get the plain
transcript with `- Speakers: not labelled`. Labels are a best guess; short interjections and people
talking over each other are most likely to be wrong. Dictation never writes audio to disk or runs labelling.

For better call audio: phone right next to the mic, quiet room, ideally a USB conference mic, and no
typing near the mic.

Notifications: on Linux (with `notify-send`) clicking "Open" opens the transcript. On Windows/macOS the
tray notification cannot be clicked through, so use the tray's "Show last transcript".

## Autostart on login

- **Windows:** `Win+R`, type `shell:startup`, add a shortcut to
  `C:\path\to\.venv\Scripts\pythonw.exe C:\path\to\PyKeet\main.py` (start in the PyKeet folder).
- **Linux:** create `~/.config/autostart/pykeet.desktop`:
  ```
  [Desktop Entry]
  Type=Application
  Name=PyKeet
  Path=/path/to/PyKeet
  Exec=/path/to/PyKeet/.venv/bin/python main.py
  ```
- **macOS:** System Settings > General > Login Items, add a small launcher script that runs the same command.

## Known limitations

- **Hotkey consumption:** only on Windows does PyKeet swallow `Ctrl+Space` so the focused app never
  sees it. On Linux/macOS pynput cannot do that, so the app also receives the keys (a warning is
  logged). If that clashes (VS Code autocomplete, input-language switching), change `dictation_shortcut`.
- **Linux Wayland:** pynput hotkeys and key injection do not work. Use an X11 session. (Wayland would need
  `evdev` for hotkeys and `ydotool`/`wtype` for typing.)
- **macOS:** grant Accessibility and Microphone permission to the terminal/Python. Widget focus behaviour
  on macOS is untested.
- **Windows:** elevated (admin) apps ignore simulated keys from a non-elevated script.
- **Clipboard restore** handles text only; if you had an image on the clipboard it is not put back.
- **Widget corners** are truly transparent only on Windows; elsewhere it is a dark pill on a rectangle.
- **Licences:** Parakeet weights are CC-BY-4.0, `diarize` is Apache 2.0. Check the moondream / Photon
  engine licence separately before publishing this as open source.
- Logs go to `pykeet.log` (rotating). Transcript text is only logged when `debug = true`.
