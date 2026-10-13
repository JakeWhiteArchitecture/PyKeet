"""PyKeet's user interface in Qt 6 (PySide6): the floating pill, the transcript viewer and the
read-back (proofreading) window. The app logic stays in main.py; call init(main_module) once.

Qt talks to Wayland natively. What Wayland does NOT allow any app to do (so Qt cannot either):
choose where its window appears, or stay above other windows. On Wayland the pill is therefore
placed by the compositor and may not float above everything; on X11 both work.
"""
from __future__ import annotations

import logging
import queue
import re
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

    def open_proof(self, text, review_cb=None, raw=None) -> None:
        win = ProofWindow(self, text, review_cb, raw)
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

CHANGE_COLOUR = "#ffb020"
COMMON_WORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i",
                "in", "is", "it", "of", "on", "or", "so", "that", "the", "to", "was", "we",
                "with", "you"}


class ChangeBubble(QtWidgets.QFrame):
    """Small box next to a changed word: shows what was there before, with a Restore button."""

    def __init__(self, parent, on_restore, on_enter, on_leave):
        super().__init__(parent)
        t = theme()
        self.setObjectName("bubble")
        self.setStyleSheet(f"QFrame#bubble {{ background:{t['card']}; border:1px solid "
                           f"{t['cyan']}; border-radius:10px; }} QLabel {{ background:"
                           f"transparent; }}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 6, 6)
        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)
        self.label.setMinimumWidth(200)
        self.label.setMaximumWidth(380)
        lay.addWidget(self.label)
        self.btn = QtWidgets.QPushButton("Restore")
        self.btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn.clicked.connect(on_restore)
        lay.addWidget(self.btn)
        self._enter, self._leave = on_enter, on_leave
        self.hide()

    def enterEvent(self, e) -> None:
        self._enter()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self._leave()
        super().leaveEvent(e)


class ProofWindow(QtWidgets.QWidget):
    """Reads the text aloud from the start.
      SPACE: tap = flag the word you just heard; HOLD = rewind word by word, release to flag
             that word and play again from it.
      Edit text: fix words in place (reading pauses; it reads your version afterwards).
      Changed words show in colour; hover for the original, Restore to put it back;
      right-click to teach PyKeet the correction for future dictation.
      Two buttons copy and close in one step (highlighted rich text, or plain). For a dictation
      (review_cb) the plain text is handed back to be pasted and dictation stays blocked until
      the window is closed."""

    def __init__(self, ui: QtWidget, text: str, review_cb=None, raw: str | None = None):
        super().__init__()
        self.ui, self.review_cb = ui, review_cb
        self.text = text
        self.raw = raw if raw is not None else text
        self.cfg = ui.app.cfg
        self.done = False
        self.edit_mode = False
        self.rw = None            # rewind state while Space is held
        self.hover = None         # index into self.changes shown in the bubble
        self.events: queue.Queue = queue.Queue()
        self.tts_state: dict = {"tried": False, "cache": None, "fails": 0}
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("PyKeet - read back" if review_cb else "PyKeet proofreading")
        self.resize(820, 600)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.changes = M.compute_changes(self.raw, self.text)
        self.sess = self.make_session(text, set(), 0, paused=False)
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
        self.edit.setMouseTracking(True)
        self.edit.viewport().setMouseTracking(True)
        self.edit.viewport().installEventFilter(self)
        self.edit.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.edit.customContextMenuRequested.connect(self.show_menu)
        f = self.edit.font()
        f.setPointSize(14)
        self.edit.setFont(f)
        lay.addWidget(self.edit, 1)

        bar = QtWidgets.QHBoxLayout()
        self.rewind = QtWidgets.QPushButton("Rewind to start")
        self.rewind.clicked.connect(lambda: self.sess.restart())
        self.pause = QtWidgets.QPushButton("Pause")
        self.pause.clicked.connect(lambda: self.sess.toggle_pause())
        self.edit_btn = QtWidgets.QPushButton("Edit text")
        self.edit_btn.clicked.connect(self.toggle_edit)
        marked = QtWidgets.QPushButton("Copy with highlights and close")
        marked.clicked.connect(lambda: self.finish(True))
        plain = QtWidgets.QPushButton("Copy plain and close")
        plain.clicked.connect(lambda: self.finish(False))
        for b in (self.rewind, self.pause, self.edit_btn, marked, plain):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        bar.addWidget(self.rewind)
        bar.addWidget(self.pause)
        bar.addWidget(self.edit_btn)
        bar.addStretch(1)
        bar.addWidget(marked)
        bar.addWidget(plain)
        lay.addLayout(bar)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, activated=self.commit_edit)

        self.bubble = ChangeBubble(self, self.restore_hovered, lambda: self.hide_timer.stop(),
                                   lambda: self.hide_timer.start(350))
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.bubble.hide)
        self.rw_timer = QTimer(self)
        self.rw_timer.timeout.connect(self.rewind_step)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.pump)
        self.timer.start(120)
        if self.sess.spans:
            self.sess.start()
        else:
            QTimer.singleShot(0, lambda: self.finish(False))

    # -- sessions ------------------------------------------------------------------------
    def make_session(self, text: str, flags: set, pos: int, paused: bool):
        sess = M.ProofSession(text, self.speak, lambda **kw: self.events.put(kw),
                              int(self.cfg["proofread_phrase_words"]),
                              float(self.cfg["proofread_reaction_seconds"]))
        sess.flags = set(flags)
        if paused and sess.spans:
            idx = next((k for k, (x, y) in enumerate(sess.spans) if x <= pos < y),
                       min(len(sess.spans) - 1, 0))
            sess.paused = True
            sess._go.clear()
            sess.cur = sess.next = idx
        return sess

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
        if sess.next != me + 1 or sess._stop or sess.hold:
            return  # skipped, paused or rewinding while this was being prepared
        words = tts.word_spans(phrase, sess.spans[me][0], len(samples) / sr)
        skip = sess.skip_to
        if skip is not None:  # start from a word (after a rewind), a little early
            sess.skip_to = None
            first = next((k for k, w in enumerate(words) if w[0] >= skip), 0)
            t0 = max(0.0, words[first][2] - 0.12)
            samples = samples[int(t0 * sr):]
            words = [(a, b, max(0.0, ts - t0), te - t0) for a, b, ts, te in words[first:]]
        pb = tts.Playback(samples, sr)
        sess.proc = pb
        sess.set_words(me, words)
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

    def word_at(self, pos: int):
        for a, b in self.sess.word_list:
            if a <= pos <= b:
                return (a, b)
        return None

    def anchors(self):
        """(char span, change) pairs: where each change is drawn (a removal is drawn on the
        word that now follows it)."""
        out = []
        for c in self.changes:
            if c["end"] > c["start"]:
                out.append(((c["start"], c["end"]), c))
            else:
                w = self.word_at(c["start"]) or (self.sess.word_list[-1]
                                                 if self.sess.word_list else None)
                if w:
                    out.append((w, c))
        return out

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
        chg = QtGui.QTextCharFormat()
        chg.setForeground(QtGui.QColor(CHANGE_COLOUR))
        chg.setUnderlineStyle(QtGui.QTextCharFormat.UnderlineStyle.DotLine)
        chg.setUnderlineColor(QtGui.QColor(CHANGE_COLOUR))
        now = QtGui.QTextCharFormat()
        now.setBackground(QtGui.QColor(t["purple"]))
        flag = QtGui.QTextCharFormat()
        flag.setBackground(QtGui.QColor(t["pink"]))
        flag.setUnderlineStyle(QtGui.QTextCharFormat.UnderlineStyle.SingleUnderline)
        playing = 0 <= sess.cur < len(sess.spans) and not sess.finished
        if playing and self.rw is None and not self.edit_mode:
            a, b = sess.spans[sess.cur]
            sels.append(self._selection(a, b, now))
        for (a, b), _c in self.anchors():  # after the reading highlight, so the colour shows
            sels.append(self._selection(a, b, chg))
        for a, b in sess.flags:
            sels.append(self._selection(a, b, flag))
        if self.rw is not None:
            cur = QtGui.QTextCharFormat()
            cur.setBackground(QtGui.QColor(t["cyan"]))
            cur.setForeground(QtGui.QColor(t["bg"]))
            a, b = sess.word_list[self.rw["idx"]]
            sels.append(self._selection(a, b, cur))
        self.edit.setExtraSelections(sels)
        if not self.edit_mode:
            focus = None
            if self.rw is not None:
                focus = sess.word_list[self.rw["idx"]][0]
            elif playing:
                focus = sess.spans[sess.cur][0]
            if focus is not None:
                c = QtGui.QTextCursor(self.edit.document())
                c.setPosition(focus)
                self.edit.setTextCursor(c)
                self.edit.ensureCursorVisible()
        self.pause.setText("Resume" if sess.paused else "Pause")
        n = len(sess.flags)
        if error:
            self.status.setText(error)
        elif self.edit_mode:
            self.status.setText("Editing: change the text, then press Done editing "
                                "(Ctrl+Enter). Reading is paused.")
        elif self.rw is not None:
            self.status.setText("Rewinding… let go of SPACE to flag the highlighted word and "
                                "play again from it")
        elif sess.note:
            self.status.setText(sess.note)
        elif sess.finished:
            self.status.setText(f"Finished reading. {n} flagged. Copy to finish "
                                f"(Esc = copy plain and close).")
        else:
            state = "Paused" if sess.paused else "Reading"
            extra = f" · {len(self.changes)} changed (hover them)" if self.changes else ""
            self.status.setText(f"{state} {sess.cur + 1}/{len(sess.spans)} · {n} flagged"
                                f"{extra} · tap SPACE to flag, HOLD it to rewind · P = pause")

    # -- hover bubble, restore, right-click ---------------------------------------------------
    def eventFilter(self, obj, ev) -> bool:
        if obj is self.edit.viewport() and not self.edit_mode:
            if ev.type() == QtCore.QEvent.Type.MouseMove:
                self.on_hover(ev.position().toPoint())
            elif ev.type() == QtCore.QEvent.Type.Leave:
                self.hide_timer.start(350)
        return False

    def change_at(self, pos: int):
        for (a, b), c in self.anchors():
            if a <= pos <= b:
                return c
        return None

    def on_hover(self, pt) -> None:
        pos = self.edit.cursorForPosition(pt).position()
        c = self.change_at(pos)
        if c is None:
            if self.bubble.isVisible():
                self.hide_timer.start(350)
            return
        self.hide_timer.stop()
        if self.hover is c and self.bubble.isVisible():
            return
        self.hover = c
        what = {"removed": "Removed", "added": "Added", "changed": "Heard"}[c["kind"]]
        self.bubble.label.setText(f"{what}: “{c['original']}”" if c["original"]
                                  else "Added by the cleanup")
        self.bubble.btn.setText("Restore")
        anchor = next((ab for ab, cc in self.anchors() if cc is c), None)
        cur = QtGui.QTextCursor(self.edit.document())
        cur.setPosition(anchor[0] if anchor else pos)
        rect = self.edit.cursorRect(cur)
        p = self.edit.viewport().mapTo(self, rect.bottomLeft())
        self.bubble.adjustSize()
        self.bubble.move(min(p.x(), max(0, self.width() - self.bubble.width() - 12)), p.y() + 4)
        self.bubble.show()
        self.bubble.raise_()

    def restore_hovered(self) -> None:
        if self.hover is not None:
            self.restore_change(self.hover)
        self.bubble.hide()

    def restore_change(self, c: dict) -> None:
        t = self.text
        if c["kind"] == "removed":
            pos = c["start"]
            if pos < len(t.rstrip()):
                new = t[:pos] + c["original"] + " " + t[pos:]
            else:
                new = t[:pos] + " " + c["original"] + t[pos:]
        else:
            new = re.sub(r"  +", " ", t[:c["start"]] + c["original"] + t[c["end"]:])
        self.apply_text(new)

    def show_menu(self, pt) -> None:
        if self.edit_mode:
            return
        c = self.change_at(self.edit.cursorForPosition(pt).position())
        if c is None:
            return
        menu = QtWidgets.QMenu(self.edit)
        restore = menu.addAction("Restore original")
        remember = None
        if c["kind"] == "changed" and c["original"]:
            remember = menu.addAction("Always replace this in future dictation…")
        chosen = menu.exec(self.edit.viewport().mapToGlobal(pt))
        if chosen is restore:
            self.restore_change(c)
        elif remember is not None and chosen is remember:
            self.remember(c)

    def remember(self, c: dict) -> None:
        def clean(s: str) -> str:
            return s.strip().strip(",.;:!?\"'").strip()

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Remember this correction")
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel("When PyKeet hears:"))
        wrong = QtWidgets.QLineEdit(clean(c["original"]))
        lay.addWidget(wrong)
        lay.addWidget(QtWidgets.QLabel("write:"))
        right = QtWidgets.QLineEdit(clean(self.text[c["start"]:c["end"]]))
        lay.addWidget(right)
        warn = QtWidgets.QLabel("")
        warn.setObjectName("muted")
        warn.setWordWrap(True)
        lay.addWidget(warn)

        def check():
            w = wrong.text().strip().lower()
            warn.setText("Careful: that is a very common word, so it would be replaced "
                         "everywhere. Add a neighbouring word to be safe."
                         if w in COMMON_WORDS or len(w) < 3 else "")

        wrong.textChanged.connect(check)
        check()
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(dlg.reject)
        ok = QtWidgets.QPushButton("Remember")
        ok.clicked.connect(dlg.accept)
        row.addWidget(cancel)
        row.addWidget(ok)
        lay.addLayout(row)
        if dlg.exec() and wrong.text().strip() and right.text().strip():
            self.ui.app.add_correction(wrong.text(), right.text())
            self.status.setText(f"Remembered: “{wrong.text().strip()}” → "
                                f"“{right.text().strip()}” (used from your next dictation; "
                                f"see python main.py --corrections)")

    # -- editing -----------------------------------------------------------------------------
    def toggle_edit(self) -> None:
        if self.edit_mode:
            self.commit_edit()
            return
        self.edit_mode = True
        if not self.sess.paused:
            self.sess.toggle_pause()
        self.bubble.hide()
        self.edit.setReadOnly(False)
        self.edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.edit.setFocus()
        self.edit_btn.setText("Done editing")

    def commit_edit(self) -> None:
        if not self.edit_mode:
            return
        self.edit_mode = False
        new = self.edit.toPlainText()
        self.edit.setReadOnly(True)
        self.edit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFocus()
        self.edit_btn.setText("Edit text")
        self.apply_text(new)

    def apply_text(self, new: str) -> None:
        """Make `new` the text: keep flags that were outside the edited stretch, re-split the
        phrases, recompute what differs from what was heard, and wait (paused) at the edit."""
        old = self.text
        if new == old:
            self.refresh()
            return
        p = 0
        while p < min(len(old), len(new)) and old[p] == new[p]:
            p += 1
        sfx = 0
        while sfx < min(len(old), len(new)) - p and old[-1 - sfx] == new[-1 - sfx]:
            sfx += 1
        delta = len(new) - len(old)
        flags = set()
        for a, b in self.sess.flags:
            if b <= p:
                flags.add((a, b))
            elif a >= len(old) - sfx:
                flags.add((a + delta, b + delta))
        self.sess.stop()
        self.text = new
        self.edit.setPlainText(new)
        self.changes = M.compute_changes(self.raw, new)
        self.sess = self.make_session(new, flags, p, paused=True)
        if self.sess.spans:
            self.sess.start()
        self.hover = None
        self.refresh()

    # -- keys: tap = flag, hold = rewind ------------------------------------------------------
    def keyPressEvent(self, e) -> None:
        k = e.key()
        if self.edit_mode:
            return super().keyPressEvent(e)
        if k == Qt.Key.Key_Space:
            if not e.isAutoRepeat():
                self.begin_hold()
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

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat() and self.rw is not None:
            self.end_hold()
        else:
            super().keyReleaseEvent(e)

    def focusOutEvent(self, e) -> None:
        if self.rw is not None:
            self.end_hold()
        super().focusOutEvent(e)

    def begin_hold(self) -> None:
        if self.rw is not None or not self.sess.spans:
            return
        span = self.sess.rewind_begin()
        if span is None:
            return
        words = self.sess.word_list
        idx = max((k for k, (a, _b) in enumerate(words) if a <= span[0]), default=0)
        self.rw = {"idx": idx, "steps": 0, "span": span}
        self.rw_timer.start(260)
        self.refresh()

    def rewind_step(self) -> None:
        if self.rw is None:
            return
        self.rw["idx"] = max(0, self.rw["idx"] - 1)
        self.rw["steps"] += 1
        if self.rw["steps"] == 8:
            self.rw_timer.setInterval(140)  # speed up when held a long time
        self.refresh()

    def end_hold(self) -> None:
        rw, self.rw = self.rw, None
        self.rw_timer.stop()
        if rw is None:
            return
        if rw["steps"] == 0:  # a tap: flag (or unflag) the word you just heard
            span = rw["span"]
            self.sess.flags.symmetric_difference_update({span})
        else:
            span = self.sess.word_list[rw["idx"]]
            self.sess.flags.add(span)
        self.sess.resume_from(span[0])
        self.refresh()

    # -- finishing: copy and close in one go ---------------------------------------------------
    def finish(self, marked: bool) -> None:
        if self.done:
            return
        self.commit_edit()
        self.done = True
        self.sess.stop()
        for tm in (self.timer, self.rw_timer, self.hide_timer):
            tm.stop()
        clip = QtGui.QGuiApplication.clipboard()
        cb = self.review_cb
        if marked:  # rich text: flagged words highlighted, plain text as the fallback
            mime = QtCore.QMimeData()
            mime.setHtml(self.sess.html())
            mime.setText(self.text)
            clip.setMimeData(mime)
            out = None  # already on the clipboard; nothing to type in
        else:
            out = self.text
            if not cb:
                clip.setText(out)
        self.close()
        if cb:  # give focus back to the app you were in, then paste
            QTimer.singleShot(300, lambda: cb(out))

    def closeEvent(self, e) -> None:
        if not self.done:  # closed with the window's own X: same as "copy plain and close"
            self.commit_edit()
            self.done = True
            self.sess.stop()
            for tm in (self.timer, self.rw_timer, self.hide_timer):
                tm.stop()
            if self.review_cb:
                cb, out = self.review_cb, self.text
                QTimer.singleShot(300, lambda: cb(out))
            else:
                QtGui.QGuiApplication.clipboard().setText(self.text)
        super().closeEvent(e)
