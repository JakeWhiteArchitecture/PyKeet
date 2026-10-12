"""PyKeet's user interface in Qt 6 (PySide6): the floating pill, the transcript viewer and the
read-back (proofreading) window. The app logic stays in main.py; call init(main_module) once.

Qt talks to Wayland natively. What Wayland does NOT allow any app to do (so Qt cannot either):
choose where its window appears, or stay above other windows. On Wayland the pill is therefore
placed by the compositor and may not float above everything; on X11 both work.
"""
from __future__ import annotations

import logging
import queue
import signal
import sys
import threading
import time
from collections import deque

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer

log = logging.getLogger("pykeet")
M = None  # main.py's module (set by init)


def init(main_module) -> None:
    global M
    M = main_module


def theme() -> dict:
    return M.THEME


def qss() -> str:
    t = theme()
    return f"""
    QWidget {{ background: {t['bg']}; color: {t['text']}; font-size: 13px; }}
    QTextEdit {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px;
                 padding: 10px; selection-background-color: {t['purple']}; }}
    QPushButton {{ background: {t['card']}; color: {t['cyan']}; border: 1px solid {t['border']};
                   border-radius: 9px; padding: 8px 16px; }}
    QPushButton:hover {{ background: {t['purple']}; color: {t['text']}; }}
    QPushButton:pressed {{ background: {t['cyan']}; color: {t['bg']}; }}
    QLabel#muted {{ color: {t['muted']}; }}
    QLabel#status {{ color: {t['text']}; padding: 2px 4px; }}
    QScrollBar:vertical {{ background: {t['card']}; width: 12px; margin: 0; border-radius: 6px; }}
    QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 6px; min-height: 30px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    """


def is_x11() -> bool:
    return QtGui.QGuiApplication.platformName() in ("xcb", "windows", "cocoa")


# --------------------------------------------------------------------------
# The floating pill
# --------------------------------------------------------------------------

class Pill(QtWidgets.QWidget):
    X_STATES = ("rec", "transcribing")
    MAX_W = 380

    def __init__(self, ui: "QtWidget"):
        flags = (Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                 | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        super().__init__(None, flags)
        self.ui = ui
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        f = QtGui.QFont()
        f.setPointSize(11)
        f.setBold(True)
        self.setFont(f)
        self.fm = QtGui.QFontMetrics(f)
        self.levels: deque = deque([0.0] * 10, maxlen=10)
        self._press_pos = None
        self._moved = False
        self._user_moved = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_position)
        self.resize(300, 46)

    # -- layout: what to show and how big -----------------------------------------
    def layout_for(self, st: str) -> dict:
        t = theme()
        lab = self.ui.label_for(st)
        if st == "rec":
            return {"w": 300, "h": 46}
        if st == "call":
            return {"w": 290, "h": 46}
        if not lab:
            return {"w": self.width(), "h": self.height()}
        text, colour, progress = lab
        xspace = 36 if st in self.X_STATES else 18
        adv = self.fm.horizontalAdvance(text)
        w = int(min(self.MAX_W, max(170, 44 + adv + xspace)))
        if progress:
            w = max(w, 240)
        text_w = w - 44 - xspace
        flags = int(Qt.TextFlag.TextWordWrap.value | Qt.AlignmentFlag.AlignLeft.value)
        rect = self.fm.boundingRect(QtCore.QRect(0, 0, text_w, 10000), flags, text)
        h = max(46, rect.height() + 28 + (10 if progress else 0))
        return {"w": w, "h": h, "text": text, "colour": colour, "progress": progress,
                "text_w": text_w, "text_h": rect.height(), "flags": flags,
                "dot": {"done": t["green"], "nospeech": t["muted"],
                        "call_queued": t["muted"]}.get(st, t["yellow"])}

    def fit(self, st: str) -> dict:
        lay = self.layout_for(st)
        w, h = lay["w"], lay["h"]
        if (w, h) != (self.width(), self.height()):
            old_h = self.height()
            self.resize(w, h)
            if not self.ui.user_placed and is_x11():
                self.move(self.x(), self.y() + old_h - h)  # grow upwards from the bottom edge
        return lay

    # -- painting ------------------------------------------------------------------
    def paintEvent(self, _e) -> None:
        t = theme()
        st = self.ui.state
        if st == "hidden":
            return
        lay = self.layout_for(st)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        radius = min(h / 2, 23)
        p.setPen(QtGui.QPen(QtGui.QColor(t["border"]), 2))
        p.setBrush(QtGui.QColor(t["card"]))
        p.drawRoundedRect(QtCore.QRectF(1, 1, w - 2, h - 2), radius, radius)
        p.setFont(self.font())
        now = time.monotonic()
        mid = h / 2

        def dot(colour, y=None):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(colour))
            p.drawEllipse(QtCore.QPointF(25, mid if y is None else y), 7, 7)

        def label(x, y_mid, s, colour):
            p.setPen(QtGui.QColor(colour))
            p.drawText(QtCore.QRectF(x, y_mid - 12, 120, 24),
                       int(Qt.AlignmentFlag.AlignVCenter.value | Qt.AlignmentFlag.AlignLeft.value),
                       s)

        def bars(x0):
            self.levels.append(min(1.0, (self.ui.app.recorder.level * 9) ** 0.6))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(t["cyan"]))
            for i, lv in enumerate(self.levels):
                bh = 4 + lv * 26
                p.drawRoundedRect(QtCore.QRectF(x0 + i * 9, mid - bh / 2, 5, bh), 2, 2)

        if st == "rec":
            dot(t["pink"])
            label(44, mid, M.fmt_ts(now - self.ui.t0), t["text"])
            bars((w - 10 * 9 + 4) / 2)  # waveform centred on the pill
        elif st == "call":
            if int(now * 2) % 2 == 0:
                dot(t["pink"])
            label(40, mid, "CALL", t["text"])
            bars(100)
            label(200, mid, M.fmt_ts(now - self.ui.t0), t["text"])
        elif "text" in lay:
            top = 14
            dot(lay["dot"], top + self.fm.height() / 2)
            p.setPen(QtGui.QColor(lay["colour"]))
            p.drawText(QtCore.QRect(44, top, lay["text_w"], lay["text_h"] + 2), lay["flags"],
                       lay["text"])
            if lay["progress"]:
                prog = self.ui.app.progress() or 0
                x0, x1, y = 44, w - 26, h - 14
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QtGui.QColor("#2a2a4a"))
                p.drawRoundedRect(QtCore.QRectF(x0, y, x1 - x0, 4), 2, 2)
                p.setBrush(QtGui.QColor(t["cyan"]))
                p.drawRoundedRect(QtCore.QRectF(x0, y, (x1 - x0) * prog, 4), 2, 2)
        if st in self.X_STATES:  # small cancel X, top right
            cx, cy, r = w - 20, 15, 4
            p.setPen(QtGui.QPen(QtGui.QColor(t["muted"]), 2, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap))
            p.drawLine(cx - r, cy - r, cx + r, cy + r)
            p.drawLine(cx - r, cy + r, cx + r, cy - r)
        p.end()

    # -- mouse: click to stop, X to cancel, drag to move -------------------------------
    def _x_hit(self, pos) -> bool:
        return (self.ui.state in self.X_STATES and pos.x() > self.width() - 34
                and pos.y() < 30)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.RightButton:
            if self.ui.state == "rec":
                self.ui.app.post(self.ui.app.cancel_dictation)
            return
        self._press_pos = e.globalPosition()
        self._moved = False

    def mouseMoveEvent(self, e) -> None:
        if self._press_pos is None or self._moved:
            return
        d = e.globalPosition() - self._press_pos
        if abs(d.x()) + abs(d.y()) > 4:
            self._moved = True
            self._user_moved = True
            handle = self.windowHandle()
            if handle is not None:
                handle.startSystemMove()  # works on X11 and Wayland

    def mouseReleaseEvent(self, e) -> None:
        if self._press_pos is None:
            return
        self._press_pos = None
        if self._moved:
            return
        app = self.ui.app
        if self._x_hit(e.position()):
            app.post(app.cancel_dictation if self.ui.state == "rec" else app.cancel_transcription)
        elif self.ui.state == "rec":
            app.post(app.stop_dictation)

    def moveEvent(self, e) -> None:
        super().moveEvent(e)
        if self._user_moved and is_x11():
            self._save_timer.start(600)

    def _save_position(self) -> None:
        self.ui.user_placed = True
        M.save_widget_position(self.x(), self.y())


# --------------------------------------------------------------------------
# The application shell: owns the Qt app, the pill and the windows
# --------------------------------------------------------------------------

class QtWidget:
    """Drop-in for the old Tk Widget: .q (queue of states/windows), .send(), .run()."""

    def __init__(self, app):
        self.app = app
        self.q: queue.Queue = queue.Queue()
        self.qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        self.qapp.setApplicationName("PyKeet")
        self.qapp.setQuitOnLastWindowClosed(False)
        self.qapp.setStyleSheet(qss())
        self.state = "hidden"   # what is on screen now
        self.fg = "hidden"      # dictation / call-recording state (takes priority)
        self.bgs = "none"       # background job state
        self.t0 = 0.0
        self.state_since = 0.0
        self.windows: list = []
        pos = app.cfg.get("widget_position") or []
        self.user_placed = len(pos) == 2
        self.pill = Pill(self)
        self.show_pill = bool(app.cfg.get("show_pill", True))
        if self.user_placed and is_x11():
            self.pill.move(int(pos[0]), int(pos[1]))
        elif is_x11():
            geo = self.qapp.primaryScreen().availableGeometry()
            self.pill.move(geo.center().x() - self.pill.width() // 2, geo.bottom() - 136)
        self.timer = QTimer()
        self.timer.timeout.connect(self._poll)
        self.timer.start(40)
        signal.signal(signal.SIGINT, lambda *_: app.post(app.quit))  # Ctrl+C in the terminal

    # -- called from any thread --------------------------------------------------
    def send(self, state: str) -> None:
        self.q.put(state)

    # -- GUI thread --------------------------------------------------------------
    def run(self) -> None:
        self.qapp.exec()

    def label_for(self, st: str):
        """(text, colour, show_progress) for the states that are a dot plus a line of text."""
        t = theme()
        note = self.app.bg_note
        if st == "transcribing":
            return "Transcribing…", t["text"], False
        if st == "call_preparing":
            return "Preparing…" + note, t["text"], False
        if st == "call_queued":
            return "Waiting for a free worker…" + note, t["muted"], False
        if st in ("call_transcribing", "call_labelling"):
            p = self.app.progress()
            pct = f" ~{int(p * 100)}%" if p is not None else ""
            return (("Transcribing" if st == "call_transcribing" else "Labelling speakers")
                    + note + pct), t["text"], p is not None
        if st == "done":
            return "Done", t["green"], False
        if st == "nospeech":
            return "No speech", t["muted"], False
        return None

    def _poll(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item[0] == "text":
                    self.open_viewer(*item[1:])
                elif isinstance(item, tuple) and item[0] == "proof":
                    try:
                        self.open_proof(*item[1:])
                    except Exception:
                        log.exception("could not open the read-back window")
                        if len(item) > 2 and item[2]:
                            item[2](item[1])  # don't lose the text or leave dictation blocked
                elif isinstance(item, tuple) and item[0] == "bg":
                    self.bgs = item[1]
                    self._apply()
                else:
                    self._set(item)
        except queue.Empty:
            pass
        now = time.monotonic()
        if self.fg == "done" and now - self.state_since > 0.5:
            self._set("hidden")
        elif self.fg == "nospeech" and now - self.state_since > 1.0:
            self._set("hidden")
        if self.state != "hidden":
            self.pill.fit(self.state)
            self.pill.update()

    def _set(self, state: str) -> None:
        if state == "quit":
            self.qapp.quit()
            return
        if state == "pick_file":
            self.pick_file()
            return
        self.fg = state
        self.state_since = time.monotonic()
        if state in ("rec", "call"):
            self.t0 = time.monotonic()
        log.info("widget state: %s", state)
        self._apply()

    def _apply(self) -> None:
        """Show the dictation/recording state if there is one, else the background job."""
        eff = self.fg if self.fg != "hidden" else (self.bgs if self.bgs != "none" else "hidden")
        self.state = eff
        if eff == "hidden" or not self.show_pill:
            self.pill.hide()
            return
        self.pill.fit(eff)
        if not self.pill.isVisible():
            self.pill.show()
        self.pill.raise_()
        self.pill.update()

    # -- windows -------------------------------------------------------------------
    def keep(self, win) -> None:
        self.windows.append(win)
        win.destroyed.connect(lambda *_: self.windows.remove(win) if win in self.windows else None)

    def open_viewer(self, title, body, path, markdown=False) -> None:
        win = ViewerWindow(self, title, body, path, markdown)
        self.keep(win)
        win.show()
        win.raise_()
        win.activateWindow()

    def open_proof(self, text, review_cb=None) -> None:
        win = ProofWindow(self, text, review_cb)
        self.keep(win)
        win.show()
        win.raise_()
        win.activateWindow()
        win.setFocus()

    def pick_file(self) -> None:
        filt = ";;".join(f"{name} ({pat})" if pat != "*.*" else f"{name} (*)"
                         for name, pat in M.AUDIO_TYPES)
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            None, "Choose audio files to transcribe", "", filt)
        if paths:
            self.app.post(lambda: self.app.import_files(paths))


# --------------------------------------------------------------------------
# Transcript viewer
# --------------------------------------------------------------------------

class ViewerWindow(QtWidgets.QWidget):
    def __init__(self, ui: QtWidget, title: str, body: str, path, markdown: bool):
        super().__init__()
        self.ui, self.body, self.markdown = ui, body, markdown
        self.mode = "formatted" if markdown else "raw"
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(860, 640)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 12)
        self.edit = QtWidgets.QTextEdit()
        self.edit.setReadOnly(True)
        lay.addWidget(self.edit, 1)

        if markdown:  # colour legend
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self._muted("Highlights (best guess):"))
            for kind, colour in M.ENTITY_COLOURS.items():
                lab = QtWidgets.QLabel(M.ENTITY_LABELS[kind])
                lab.setStyleSheet(f"background:{colour}; color:#111; padding:2px 8px; "
                                  f"border-radius:6px;")
                row.addWidget(lab)
            row.addStretch(1)
            lay.addLayout(row)
            if ui.app.cfg.get("highlight_actions"):
                row2 = QtWidgets.QHBoxLayout()
                row2.addWidget(self._muted("Underlined:"))
                for kind, colour in M.ACTION_COLOURS.items():
                    lab = QtWidgets.QLabel(M.ACTION_LABELS[kind])
                    lab.setStyleSheet(f"color:{colour}; text-decoration: underline; "
                                      f"padding:2px 6px; font-weight:bold;")
                    row2.addWidget(lab)
                row2.addStretch(1)
                lay.addLayout(row2)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.status.hide()
        lay.addWidget(self.status)

        bar = QtWidgets.QHBoxLayout()
        self.copy_btn = QtWidgets.QPushButton("Copy all")
        self.copy_btn.clicked.connect(self.copy_all)
        bar.addWidget(self.copy_btn)
        self.raw_btn = QtWidgets.QPushButton("Show raw Markdown")
        self.sum_btn = QtWidgets.QPushButton("Action summary")
        self.raw_btn.clicked.connect(lambda: self.set_mode("raw"))
        self.sum_btn.clicked.connect(lambda: self.set_mode("summary"))
        if markdown:
            bar.addWidget(self.raw_btn)
            bar.addWidget(self.sum_btn)
        if path:
            b = QtWidgets.QPushButton("Open file")
            b.clicked.connect(lambda: M.open_path(path))
            bar.addWidget(b)
            bar.addWidget(self._muted(str(path)))
        bar.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        bar.addWidget(close)
        lay.addLayout(bar)
        QtGui.QShortcut(QtGui.QKeySequence("Escape"), self, activated=self.close)

        self.render()
        self.start_analysis()

    def _muted(self, text: str) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel(text)
        lab.setObjectName("muted")
        return lab

    # -- formats for the tags md_segments produces ------------------------------------
    def fmt(self, tags) -> QtGui.QTextCharFormat:
        t = theme()
        f = QtGui.QTextCharFormat()
        for tag in tags:
            if tag == "h1":
                f.setFontPointSize(18)
                f.setFontWeight(QtGui.QFont.Weight.Bold)
            elif tag == "h2":
                f.setFontPointSize(14)
                f.setFontWeight(QtGui.QFont.Weight.Bold)
            elif tag == "bold":
                f.setFontWeight(QtGui.QFont.Weight.Bold)
            elif tag in ("ts", "bullet"):
                f.setForeground(QtGui.QColor(t["muted"]))
            elif tag in M.ENTITY_COLOURS:
                f.setBackground(QtGui.QColor(M.ENTITY_COLOURS[tag]))
                f.setForeground(QtGui.QColor("#111111"))
            elif tag in M.ACTION_COLOURS:
                f.setUnderlineStyle(QtGui.QTextCharFormat.UnderlineStyle.SingleUnderline)
                f.setUnderlineColor(QtGui.QColor(M.ACTION_COLOURS[tag]))
        return f

    def render(self) -> None:
        cfg = self.ui.app.cfg
        self.edit.clear()
        if self.mode == "raw":
            self.edit.setPlainText(self.body)
            return
        cur = self.edit.textCursor()
        if self.mode == "summary":
            segs = M.md_segments(M.action_summary(self.body), highlight=False)
        else:
            segs = M.md_segments(self.body, cfg.get("highlight_names") or (),
                                 cfg.get("highlight_companies") or (),
                                 ner=bool(cfg.get("use_ner")),
                                 actions=bool(cfg.get("highlight_actions")))
        for chunk, tags in segs:
            cur.insertText(chunk, self.fmt(tags))
        self.edit.moveCursor(QtGui.QTextCursor.MoveOperation.Start)

    def copy_all(self) -> None:
        text = M.action_summary(self.body) if self.mode == "summary" else self.body
        QtGui.QGuiApplication.clipboard().setText(text)

    def set_mode(self, mode: str) -> None:
        self.mode = "formatted" if self.mode == mode else mode
        self.raw_btn.setText("Show formatted" if self.mode == "raw" else "Show raw Markdown")
        self.sum_btn.setText("Show formatted" if self.mode == "summary" else "Action summary")
        self.render()

    # -- AI decision/task analysis in the background -------------------------------------
    def start_analysis(self) -> None:
        cfg = self.ui.app.cfg
        model_name = str(cfg.get("action_model") or "")
        backend = str(cfg.get("action_backend") or "zeroshot")
        if not (self.markdown and cfg.get("highlight_actions")
                and (model_name or backend in ("server", "needle", "phrases", "needle-phrases"))):
            return
        self.status.setText("Analysing decisions and tasks with the AI model "
                            "(the first time it downloads the model)…")
        self.status.show()
        self.results: queue.Queue = queue.Queue()

        def work():
            try:
                model = M.get_action_model(model_name, backend,
                                           str(cfg.get("action_server") or ""),
                                           float(cfg.get("phrase_min_similarity") or 0))
                if model is None:
                    raise RuntimeError("model could not be loaded; see pykeet.log")
                M.analyse_actions(self.body, model)
                self.results.put(None)
            except Exception as exc:
                log.exception("action analysis failed")
                self.results.put(exc)

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_analysis)
        self.poll_timer.start(300)
        threading.Thread(target=work, daemon=True).start()

    def _poll_analysis(self) -> None:
        try:
            res = self.results.get_nowait()
        except queue.Empty:
            return
        self.poll_timer.stop()
        if res is None:
            self.status.setText("AI analysis done.")
            self.render()
        else:
            self.status.setText(f"AI model unavailable ({res}); showing the phrase rules.")


# --------------------------------------------------------------------------
# Read-back window
# --------------------------------------------------------------------------

class ProofWindow(QtWidgets.QWidget):
    """Reads the text aloud from the start; SPACE flags the word being spoken. Two buttons
    copy and close in one step (with [CHECK] marks, or plain). For a dictation (review_cb)
    the chosen text is handed back to be pasted and dictation stays blocked until closed."""

    def __init__(self, ui: QtWidget, text: str, review_cb=None):
        super().__init__()
        self.ui, self.text, self.review_cb = ui, text, review_cb
        self.cfg = ui.app.cfg
        self.done = False
        self.events: queue.Queue = queue.Queue()
        self.tts_state: dict = {"tried": False, "cache": None, "fails": 0}
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("PyKeet - read back" if review_cb else "PyKeet proofreading")
        self.resize(780, 560)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.sess = M.ProofSession(text, self.speak, lambda **kw: self.events.put(kw),
                                   int(self.cfg["proofread_phrase_words"]),
                                   float(self.cfg["proofread_reaction_seconds"]))
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 12)
        self.status = QtWidgets.QLabel("Starting…")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.edit = QtWidgets.QTextEdit()
        self.edit.setReadOnly(True)
        self.edit.setPlainText(text)
        self.edit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        f = self.edit.font()
        f.setPointSize(14)
        self.edit.setFont(f)
        lay.addWidget(self.edit, 1)

        bar = QtWidgets.QHBoxLayout()
        self.rewind = QtWidgets.QPushButton("Rewind to start")
        self.rewind.clicked.connect(self.sess.restart)
        self.pause = QtWidgets.QPushButton("Pause")
        self.pause.clicked.connect(self.sess.toggle_pause)
        marked = QtWidgets.QPushButton("Copy with [CHECK] marks and close")
        marked.clicked.connect(lambda: self.finish(True))
        plain = QtWidgets.QPushButton("Copy plain and close")
        plain.clicked.connect(lambda: self.finish(False))
        for b in (self.rewind, self.pause, marked, plain):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        bar.addWidget(self.rewind)
        bar.addWidget(self.pause)
        bar.addStretch(1)
        bar.addWidget(marked)
        bar.addWidget(plain)
        lay.addLayout(bar)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.pump)
        self.timer.start(120)
        if self.sess.spans:
            self.sess.start()
        else:
            QTimer.singleShot(0, lambda: self.finish(False))

    # -- speech (runs in the session thread) ---------------------------------------------
    def speak(self, phrase: str, sess) -> None:
        st = self.tts_state
        if not st["tried"]:
            st["tried"] = True
            try:
                import tts

                def note(msg):
                    sess.note = msg
                    sess.on_change()

                speaker = tts.get_speaker(self.cfg, M.HERE / "tts-models", note)
                sess.note = ""
                if speaker:
                    st["cache"] = tts.AudioCache(speaker)
                    st["tts"] = tts
            except Exception:
                log.exception("embedded speech unavailable")
                sess.note = ""
        cache = st["cache"]
        if not cache:
            raise RuntimeError("no voice installed - run: pip install kokoro-onnx")
        tts = st["tts"]
        me = sess.cur
        try:
            samples, sr = cache.get(phrase)
            st["fails"] = 0
        except Exception:
            st["fails"] += 1
            log.exception("proofread: could not speak phrase %d/%d", me + 1, len(sess.spans))
            if st["fails"] >= 3:
                raise RuntimeError("the voice keeps failing (see the log)")
            return  # skip this phrase, carry on
        if me + 1 < len(sess.spans):
            cache.prefetch(sess.phrase(me + 1))
        if sess.next != me + 1 or sess._stop:
            return  # skipped or paused while this was being prepared
        pb = tts.Playback(samples, sr)
        sess.proc = pb
        sess.set_words(me, tts.word_spans(phrase, sess.spans[me][0], pb.duration))
        log.info("proofread: phrase %d/%d (%.1fs)", me + 1, len(sess.spans), pb.duration)
        pb.start()
        sess.began_playing()
        pb.wait()

    # -- display ---------------------------------------------------------------------------
    def pump(self) -> None:
        err = None
        try:
            while True:
                err = self.events.get_nowait().get("error") or err
        except queue.Empty:
            pass
        self.refresh(err)

    def _selection(self, a: int, b: int, fmt: QtGui.QTextCharFormat):
        sel = QtWidgets.QTextEdit.ExtraSelection()
        cur = QtGui.QTextCursor(self.edit.document())
        cur.setPosition(a)
        cur.setPosition(b, QtGui.QTextCursor.MoveMode.KeepAnchor)
        sel.cursor = cur
        sel.format = fmt
        return sel

    def refresh(self, error=None) -> None:
        t = theme()
        sess = self.sess
        sels = []
        now = QtGui.QTextCharFormat()
        now.setBackground(QtGui.QColor(t["purple"]))
        flag = QtGui.QTextCharFormat()
        flag.setBackground(QtGui.QColor(t["pink"]))
        flag.setUnderlineStyle(QtGui.QTextCharFormat.UnderlineStyle.SingleUnderline)
        if 0 <= sess.cur < len(sess.spans) and not sess.finished:
            a, b = sess.spans[sess.cur]
            sels.append(self._selection(a, b, now))
        for a, b in sess.flags:
            sels.append(self._selection(a, b, flag))
        self.edit.setExtraSelections(sels)
        if 0 <= sess.cur < len(sess.spans) and not sess.finished:
            cur = QtGui.QTextCursor(self.edit.document())
            cur.setPosition(sess.spans[sess.cur][0])
            self.edit.setTextCursor(cur)
            self.edit.ensureCursorVisible()
        self.pause.setText("Resume" if sess.paused else "Pause")
        n = len(sess.flags)
        if error:
            self.status.setText(error)
        elif sess.note:
            self.status.setText(sess.note)
        elif sess.finished:
            self.status.setText(f"Finished reading. {n} flagged. Copy to finish "
                                f"(Esc = copy plain and close).")
        else:
            state = "Paused" if sess.paused else "Reading"
            self.status.setText(f"{state} {sess.cur + 1}/{len(sess.spans)} · {n} flagged · "
                                f"SPACE = flag the word you just heard · P = pause · "
                                f"Esc = copy plain and close")

    # -- finishing: copy and close in one go ---------------------------------------------------
    def finish(self, marked: bool) -> None:
        if self.done:
            return
        self.done = True
        out = self.sess.marked_text() if marked else self.text
        self.sess.stop()
        self.timer.stop()
        if not self.review_cb:
            QtGui.QGuiApplication.clipboard().setText(out)
        cb = self.review_cb
        self.close()
        if cb:  # give focus back to the app you were in, then paste
            QTimer.singleShot(300, lambda: cb(out))

    def closeEvent(self, e) -> None:
        if not self.done:  # closed with the window's own X: same as "copy plain and close"
            self.done = True
            self.sess.stop()
            self.timer.stop()
            if self.review_cb:
                cb, out = self.review_cb, self.text
                QTimer.singleShot(300, lambda: cb(out))
            else:
                QtGui.QGuiApplication.clipboard().setText(self.text)
        super().closeEvent(e)

    def keyPressEvent(self, e) -> None:
        k = e.key()
        if k == Qt.Key.Key_Space:
            self.sess.flag()
        elif k == Qt.Key.Key_P:
            self.sess.toggle_pause()
        elif k == Qt.Key.Key_Left:
            self.sess.jump(-1)
        elif k == Qt.Key.Key_Right:
            self.sess.jump(1)
        elif k == Qt.Key.Key_Escape:
            self.finish(False)
        else:
            super().keyPressEvent(e)
