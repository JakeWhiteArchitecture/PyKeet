# How PyKeet fits together

PyKeet is one Python program (`main.py`) plus a few helper modules. This page shows the parts,
how a dictation and a call travel through them, and where something else (for example an agent
or a project board) could attach. GitHub draws the diagrams below; if you read this somewhere
else, paste a block into <https://mermaid.live>.

**Built** = exists today. **Proposed** = an idea, not built.

| File | Job |
|---|---|
| `main.py` | The app: shortcuts, recording, speech engine, cleanup, call jobs, files |
| `ui_qt.py` | Every window (Qt 6): the floating pill, transcript viewer, read-back window |
| `tts.py` | Text-to-speech for the read-back (Kokoro, Piper fallback) |
| `llm_cleanup.py` | The small language model (LFM2.5-350M) that helps clean a dictation |
| `portal.py` | Optional Wayland helpers (typing and shortcuts through desktop portals) |
| `config.toml` | Your settings |
| `corrections.json` | Fixes you taught PyKeet from the read-back window |

## 1. The big picture

```mermaid
flowchart TB
    subgraph IN["Things you do"]
        KEYS["Shortcuts<br/>(GNOME shortcut, portal or hotkey)"]
        MIC["Microphone"]
        FILES["Audio files<br/>mp3 / ogg / flac / wav"]
        SEL["Selected text"]
    end

    subgraph APP["PyKeet (main.py)"]
        SOCK["Control socket<br/>main.py --toggle-dictation etc."]
        QUEUE["One action queue<br/>(App.post)"]
        REC["Recorder<br/>writes a crash-safe WAV"]
        ENG["Speech engine<br/>Parakeet via Moondream"]
        DIA["Speaker labels<br/>(diarize, calls only)"]
        CLEAN["Cleanup<br/>rules + LFM2.5 with a safety check"]
        VOICE["Voice<br/>Kokoro or Piper"]
    end

    subgraph WIN["Windows (ui_qt.py)"]
        PILL["Pill<br/>recording / transcribing / done"]
        READ["Read-back window<br/>Cleanup view + Proofread view"]
        VIEW["Transcript viewer"]
    end

    subgraph OUT["Results"]
        PASTE["Text at the cursor<br/>or on the clipboard"]
        MD["Call transcript (.md)<br/>in ~/CallTranscripts"]
    end

    KEYS --> SOCK --> QUEUE
    QUEUE --> REC
    MIC --> REC
    FILES --> ENG
    REC --> ENG
    ENG --> CLEAN
    ENG --> DIA
    DIA --> MD
    CLEAN --> PASTE
    CLEAN --> READ
    READ --> PASTE
    SEL --> READ
    READ --- VOICE
    MD --> VIEW
    QUEUE -.-> PILL
```

## 2. A dictation, step by step

```mermaid
sequenceDiagram
    actor You
    participant Key as Shortcut
    participant App as PyKeet
    participant Eng as Speech engine
    participant Cl as Cleanup
    participant Win as Read-back window
    participant Tgt as The app you are typing in

    You->>Key: Ctrl+Space
    Key->>App: toggle-dictation
    App->>App: start recording (pill shows the timer)
    You->>Key: Ctrl+Space again
    App->>App: stop, trim long silences
    App->>Eng: audio
    Eng-->>App: raw text (what Parakeet heard)
    App->>Cl: raw text
    Note over Cl: 1 fillers (um, uh) and doubled words<br/>2 false starts (rules)<br/>3 your replacements and corrections<br/>4 LFM2.5 suggestion, only safe edits kept
    Cl-->>App: cleaned text + list of word changes
    alt 3+ sentences, 30+ words, or changes and review_when_changed
        App->>Win: open (text, raw text)
        You->>Win: review, edit, flag, then Copy and close
        Win-->>App: final text, or rich text on the clipboard
    end
    App->>Tgt: paste (or copy, then you press Ctrl+V on Wayland)
```

## 3. A call

```mermaid
flowchart TD
    A["Ctrl+Alt+R<br/>start recording"] --> B["Recorder streams audio to a temp WAV<br/>(survives a crash)"]
    B --> C["Ctrl+Alt+R again<br/>stop"]
    C --> D["Job queue<br/>1 or 2 jobs at a time (call_workers)"]
    D --> E["Parakeet transcribes with timestamps"]
    E --> F["diarize works out who spoke when"]
    F --> G["build_markdown<br/>speaker turns, timestamps, your replacements"]
    G --> H["Saved to ~/CallTranscripts/..."]
    H --> I["Transcript viewer<br/>names, phones, emails, companies highlighted"]
    B -. "app crashed?" .-> R["Next start: PyKeet finds the leftover WAV<br/>and transcribes it (crash recovery)"]
    R --> D
```

Imported audio files follow the same path from the transcription step onwards.

## 4. The read-back window and how PyKeet learns

```mermaid
flowchart LR
    RAW["What Parakeet heard"] --> CL["Cleanup"]
    CL --> TXT["Cleaned text"]
    RAW --> DIFF["Compare word by word"]
    TXT --> DIFF
    DIFF --> CV["Cleanup view<br/>changed words in amber<br/>hover: the original<br/>Restore puts it back"]
    TXT --> PV["Proofread view<br/>reads aloud (Kokoro)<br/>tap Space: flag a word<br/>hold Space: rewind and flag"]
    CV -- "Edit text, then right-click:<br/>Always replace this" --> CORR["corrections.json"]
    CORR -- "applied to every future dictation" --> CL
    PV --> OUTC["Copy with highlights (rich text)<br/>or Copy plain"]
    CV --> OUTC
```

## 5. Where other tools could attach (proposed, not built)

PyKeet already has two seams that need no changes:

- **Commands in:** `python main.py --toggle-dictation`, `--toggle-call`, `--import`, `--recall`,
  `--pause`, `--quit`, `--show-last` talk to the running copy through a local socket. Anything
  that can run a command (a script, an agent) can start and stop a recording.
- **Files out:** every call ends as a Markdown transcript in `~/CallTranscripts`.

The natural next step is a second file next to each transcript with the same content as
structured data. This is the part that does not exist yet.

```mermaid
flowchart LR
    PK["PyKeet<br/>(built)"] --> MD["Transcript .md<br/>(built)"]
    PK -.-> JS["Transcript .json<br/>speaker, start, end, text<br/>(proposed)"]
    MD --> AG["Agent such as Hermes<br/>reads the transcript<br/>(proposed)"]
    JS -.-> AG
    AG -.-> TASKS["Tasks, decisions and notes<br/>(proposed)"]
    TASKS -.-> AK["ArcKanBan project JSON<br/>(proposed)"]
    AG -.->|"commands"| PK
```

Pulling tasks and decisions out of a call is deliberately **not** done inside PyKeet any more
(the old rule-based detector was removed). A language model that reads the whole transcript is a
better fit, and that is the agent's job in this picture.
