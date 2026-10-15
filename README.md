# PyKeet

Local, offline dictation and call transcription, built on Moondream Parakeet.
No cloud, no account, no word limits.

- **Dictation:** press `Ctrl+Space`, speak, press `Ctrl+Space` again. Cleaned-up text appears at the cursor in whatever app has focus.
- **Call mode:** press `Ctrl+Alt+R` at the start of a speakerphone call and again at the end. A timestamped, speaker-labelled Markdown transcript is saved to `~/CallTranscripts/`.

> **Status:** the pure logic (text cleanup, replacements, speaker assignment, transcript format,
> crash-recovery of WAVs) is tested. The hotkeys, widget, microphone and model calls need a real
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
pip install --no-cache-dir -r requirements-optional.txt   # file import, speaker labels, name highlighting, Wayland portals, voices
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

Linux also needs system packages. Debian/Ubuntu: `sudo apt install xclip libportaudio2 libnotify-bin libxcb-cursor0`.
Fedora: `sudo dnf install xclip portaudio libnotify xcb-util-cursor`. The windows use Qt 6
(`PySide6-Essentials`, installed by `requirements.txt`, about 250 MB), which talks to Wayland natively.
`config.toml` is created next to `main.py` on first run. The first start loads the model (a download
the first time, then fully offline); the floating pill and a notification say when it is ready.

**Quit Vibe Typer first.** Both apps want `Ctrl+Space`.

## Using it

| Shortcut | What it does |
| --- | --- |
| `Ctrl+Space` | Press once to start recording, press again to stop, transcribe and insert (presses under 0.3 s are ignored). Set `dictation_mode = "push"` in `config.toml` if you prefer hold-to-talk. |
| `F8` | Re-insert the last dictation |
| X on the pill, or right-click it, while recording | Discard |
| `Ctrl+Alt+R` | Start / stop call recording |
| `Ctrl+Alt+O` | Open the audio file picker  |

Widget: click while dictating = stop and insert, right-click = cancel. It is draggable and remembers its
position. During a call it shows "CALL" and ignores clicks; stop with the shortcut.
While a call is being recorded the dictation shortcut is disabled.

There is no tray icon. Other actions are commands you can bind to a shortcut or run in a terminal:
`python main.py --show-last` (open the latest call transcript), `--pause` (pause/resume dictation),
`--quit`, `--import`, `--toggle-call`, `--toggle-dictation`.

## Config

Everything is in `config.toml` (commented). Highlights: shortcuts, `dictation_mode = "push" | "toggle"`,
`model` / `call_model` / `device` (`cpu`, `cuda`, `mps`), `input_device`, `cleanup`, `insert_method`
(`paste` or `type`), `transcript_folder`, `keep_audio`, `label_speakers`, `expected_speakers`, and the
`[replacements]` table (case-insensitive, both modes).

For a GPU, set `call_model = "moondream/parakeet-ultra"` and `call_device = "cuda"` (it is loaded the
first time you record a call).

## Transcribing an audio file

**Ctrl+Alt+O** (or `python main.py --import`) opens your desktop's own file dialog (via `zenity` on GNOME, `kdialog` on KDE, so you get your usual sidebar
and bookmarks; on Fedora `sudo dnf install zenity` if it is missing; set `native_file_picker = false` to use the
basic built-in dialog) for `.mp3`, `.ogg`, `.flac` and `.wav` (pick
several to batch them). Each file is decoded to 16 kHz mono, transcribed with the call settings
(speaker labelling included, same fallbacks), and saved in your transcript folder as
`YYYY-MM-DD_HHMM_<filename>.md`. When it finishes, a **text window** shows the transcript with
**Copy all** and **Open file** buttons. Your original file is never modified or deleted. It is
unavailable while a recording or another job is running. Needs `soundfile` (in `requirements-optional.txt`); if a
file can't be read, a window shows the exact reason.

## Progress, parallel work and the transcript viewer

- **Percent complete.** While a call or audio file is transcribed, the pill shows
  `Transcribing ~37%` with a thin progress bar (then `Labelling speakers ~..%`). It is an
  *estimate* from the audio length and how fast your machine has actually been (it learns from each
  job and stores that in `speeds.json`), so the first job may be a little off. It holds at 99% until
  the work really finishes.
- **Several call jobs at once, dictation always one at a time.** `call_workers = 2` (default) lets two
  call / audio-file transcriptions run at the same time, each on its own copy of the model; more jobs
  wait in a queue (the pill says "Waiting for a free worker…", and "(3 jobs)" when several are going).
  You can start the next call recording while earlier ones are still being transcribed, and
  picking several files starts one job per file. Dictation has its own model and only ever runs one
  dictation at a time, and it keeps working while call jobs run. The cost is RAM (about one more
  model per worker) and your CPU is shared, so each job is slower while they overlap. Set
  `call_workers = 1` for a single background job, or `0` to share the dictation model (dictation then
  waits while a call job runs).
- **Silence trimming (dictation only).** When you dictate, long silent pauses (e.g. you stop to think) are
  cut out before transcribing (`trim_dictation_silence = true`). Only silences of
  `dictation_silence_min_seconds` (2 s) or more are shortened, and `dictation_silence_keep_seconds`
  (0.4 s) of quiet is always kept before and after your speech, so the first sound of a returning voice
  is never clipped; shorter pauses are untouched. A silence is only cut once speech has clearly resumed
  after it, and the level adapts to your room noise (set `dictation_silence_threshold_db` to a number like
  `-45` to override). Calls, meetings and imported audio files are **not** trimmed.
- **Transcript viewer.** Imported files, and `--show-last`, open a small Markdown viewer
  (headings, bold speaker names, grey timestamps). Names / proper nouns (blue), phone numbers (green),
  addresses and postcodes (orange), companies (purple) and emails (teal) are coloured. This is
  rule-based guesswork, not AI: it will miss some and mark some places or products as names. Add anyone
  or any firm you want always marked to `highlight_names` / `highlight_companies` in `config.toml`.
  "Show raw Markdown" and "Copy all" give you the plain text.
- **Smarter name detection (optional).** With a small name-recognition model (spaCy, about 40 MB, runs in
  milliseconds on CPU, no GPU or big LLM needed) the viewer finds people, companies and places much
  better than the built-in rules, including unusual names at the start of a sentence. Install it once:
  `pip install spacy` then `python -m spacy download en_core_web_md` (use `en_core_web_sm` for the
  12 MB version). PyKeet uses it automatically (`use_ner = true`) and falls back to the simple rules if
  it is missing. It is still a statistical guess and will occasionally mislabel something.
- **One copy at a time.** Starting a second PyKeet is refused with a message. Two copies both react
  to every shortcut (two dialogs, two pastes). To stop an old copy: `pkill -f main.py`.

**False starts.** With `remove_restarts = true` (default) a dictation like "I might start a
sentence and actually I might start a sentence" becomes "I might start a sentence". It is plain
rules, not an AI model: it fires only when at least `restart_min_words` (3) words at the start of
a phrase are repeated, either as a full repeat or after a correction word (actually, sorry,
I mean, no wait). Numbers said twice and repeated whole sentences you finished are left alone,
but it can occasionally cut something you repeated on purpose. It never touches call transcripts.

**Small-model pass (LFM2.5-350M, on by default).** `cleanup_backend = "llm"` also asks LiquidAI's
LFM2.5-350M (about 700 MB, downloads on first use, `pip install transformers` recent enough for
LFM2) to catch false starts where the second attempt uses slightly different words. It is cautious:
the model's answer is only trusted for deleting a repeated phrase and for punctuation/capitals; any
word it changes or adds is ignored and your original word is kept. Dictations under
`cleanup_min_words` (8) skip it. Try it without speaking: `python main.py --clean "your sentence"`
(it prints what the model said and how many changes were accepted or ignored). Untested on a real
model so far: check the speed and quality on your own dictation before relying on it.

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
notification cannot be clicked through, so run `python main.py --show-last`.

## Wayland / Fedora 43 (GNOME)

Fedora 43 removes the X11 session, so apps can no longer listen for keys globally or press keys
for you. PyKeet works around it (**written without access to a Wayland machine, so untested**):

1. Start PyKeet as normal, then once run `python main.py --install-gnome-shortcuts`. GNOME now
   owns the shortcuts (Settings > Keyboard > Custom Shortcuts) and runs
   `python main.py --toggle-dictation` etc., which message the running PyKeet.
   On other desktops, `python main.py --shortcut-commands` prints the commands to bind.
2. Shortcuts are **press to start, press again to stop**. Hold-to-talk is not available on
   Wayland.
3. Inserting text: PyKeet cannot press Ctrl+V for you, so it **copies the text and shows a
   notification - press Ctrl+V**. For automatic typing, install `ydotool`, run its daemon, and
   give yourself write access to `/dev/uinput` (udev rule); PyKeet then uses it automatically.
4. Mic, file picker and notifications work as before. **Native Wayland windows cannot choose their
   position or stay above other windows**, so by default PyKeet runs its windows through XWayland
   (`qt_platform = "auto"`, needs `sudo dnf install xcb-util-cursor`), which puts the pill at the
   bottom centre and keeps it on top. If the pill ever steals keyboard focus from the app you
   dictate into, set `show_pill = false`; notifications still tell you the state.

**Python on Fedora 43:** the system Python becomes 3.14, and your `.venv` (built on 3.13) will
probably break. Rebuild it:
`sudo dnf install python3.13 python3.13-devel`, then
`python3.13 -m venv .venv && source .venv/bin/activate`, then the CPU-only torch command and
`pip install -r requirements.txt` from the Install section.

### Wayland stages 2-4 (all optional, all untested on real GNOME)

Set `wayland_portals = true` in `config.toml` and `pip install jeepney`. Then:
- **Typing:** at start GNOME asks once to let PyKeet control the keyboard (the answer is
  remembered). Dictated text is then typed into the focused app, no Ctrl+V needed.
- **Hold-to-talk:** PyKeet registers the shortcuts itself through the GlobalShortcuts portal
  (GNOME 48+), which reports key press *and* release, so `dictation_mode = "push"` can work.
  GNOME shows a dialog to confirm the keys. If you ran `--install-gnome-shortcuts` earlier,
  remove those custom shortcuts first (Settings > Keyboard) or each key fires twice.
- **Feedback:** the floating pill is positioned by GNOME and may not stay on top. With
  `wayland_status_notifications = true` you get small "Recording / Transcribing / Done"
  notifications instead. PyKeet has no tray icon.
- Anything the desktop refuses falls back to the Stage 1 behaviour, and the reason is in the log.

## Proofreading: have the text read back to you

Select text in any app (email draft, dictation result) and press **Ctrl+Alt+P**. A window opens
and reads it aloud one short phrase at a time, highlighting the phrase being spoken.

**Long dictations are read back first.** When a dictation is 3+ sentences (`review_min_sentences`)
or 30+ words (`review_min_words`) it is not pasted straight away. A window opens instead. No new
dictation can start until you close it. The window has two views:
- **Cleanup** shows what the cleanup changed, hover/restore/right-click as described below. It opens
  in this view when something was changed (`review_start_view = "auto"`).
- **Proofread** reads the text aloud and lets you flag errors.
If GNOME opens the window behind your other windows, PyKeet sends a notification. (`review_when_changed = true` also opens it for a short
dictation when the cleanup replaced or removed words. `review_long_dictation = false` turns the
window off.)

**Flagging mistakes (Proofread view).** **Tap Space** to flag the word you just heard. Or **hold Space**: the audio
stops and the highlight walks backwards word by word; let go to flag the word it stopped on and
hear it again from there. **P** pauses, **Left/Right** go back/skip, or use the buttons.

**Seeing what the cleanup changed (Cleanup view).** Words that were replaced (e.g. "sun form" to "SunForm"), and
the spot where a false start was cut, show in amber with a dotted underline. Hover over one to see
what Parakeet originally heard, and click **Restore** to put it back. (Capitals, punctuation and
um/uh removal are not marked.)

**Fixing text and teaching PyKeet.** **Edit text** lets you correct words in place (reading pauses,
then reads your version). Your fix shows as a change too. **Right-click** a changed word and choose
**Always replace this in future dictation** to remember it: e.g. "damp proofing coarse" becomes
"damp proof course" from then on. It is saved in `corrections.json` (not in your hand-edited
`config.toml`; lines in `config.toml` win if both exist). `python main.py --corrections` lists what
was learned and `python main.py --forget "phrase"` removes one. This fixes the text after Parakeet
has heard it; it does not retrain Parakeet, so it only helps when Parakeet makes the same mistake
each time.

**Finishing.** **Copy with highlights and close** puts rich text on the clipboard with flagged
words highlighted (it pastes straight into an email), plus a plain-text fallback. **Copy plain and
close** copies the clean text. Copying also closes the window. On Wayland, press Ctrl+V
afterwards.

**Voices** (`tts_engine = "auto"` tries Kokoro then Piper; install at least one; models download on first use into
`tts-models/`):
1. **Kokoro** - the most natural. `pip install kokoro-onnx`. About 120 MB of models and
   0.5-1 GB RAM while speaking (tested here: ~400 MB, int8 model, faster than real time).
   Change voice with `kokoro_voice` (UK: bf_emma, bf_isabella, bm_george, bm_lewis).
2. **Piper** - lighter, flatter. `pip install piper-tts`. `piper_noise_scale` makes it livelier.

Word-level flags use estimated word timings (the voice doesn't report them), so a flag can land
a word either side. Piper's download and voices were not tested here; Kokoro was.
- On Wayland, select the text and press the shortcut; if nothing is picked up, press Ctrl+C
  first (PyKeet falls back to the clipboard).
- It catches what *sounds* wrong (mis-heard words, garbled names), not mistakes that sound
  identical when spoken, like there/their.

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
- **Linux Wayland:** global hotkeys and key typing need the GNOME shortcuts / portals described in the
  Wayland section above.
- **macOS:** grant Accessibility and Microphone permission to the terminal/Python. Widget focus behaviour
  on macOS is untested.
- **Windows:** elevated (admin) apps ignore simulated keys from a non-elevated script.
- **Clipboard restore** handles text only; if you had an image on the clipboard it is not put back.
- **Pill position and always-on-top** work on X11, Windows and macOS, and on Wayland because PyKeet
  runs its windows through XWayland by default (`qt_platform = "auto"`; needs `xcb-util-cursor`).
  With `qt_platform = "wayland"` GNOME decides where windows go and nothing stays on top.
- **Licences:** Parakeet weights are CC-BY-4.0, `diarize` is Apache 2.0. Check the moondream / Photon
  engine licence separately before publishing this as open source.
- Logs go to `pykeet.log` (rotating). Transcript text is only logged when `debug = true`.
