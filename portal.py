"""Wayland helpers for PyKeet: talk to the XDG desktop portals over D-Bus.

  * RemoteDesktop portal  -> type text into the focused app (GNOME asks you once)
  * GlobalShortcuts portal -> global hotkeys with press AND release events (GNOME 48+)

Needs `pip install jeepney` (pure Python). Only imported when `wayland_portals = true`.
The mechanics are tested against a fake portal; how real GNOME behaves is not.
"""
from __future__ import annotations

import itertools
import logging
import os
import queue
import threading
import time
from pathlib import Path

log = logging.getLogger("pykeet")

PORTAL_NAME = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
REQUEST_IFACE = "org.freedesktop.portal.Request"
RD_IFACE = "org.freedesktop.portal.RemoteDesktop"
GS_IFACE = "org.freedesktop.portal.GlobalShortcuts"

KEY_RETURN, KEY_TAB = 0xFF0D, 0xFF09
EVDEV_CTRL, EVDEV_V = 29, 47


class PortalError(Exception):
    pass


def keysym_for(ch: str) -> int:
    """X11 keysym for a character (newline/tab are the Return/Tab keys)."""
    if ch == "\n":
        return KEY_RETURN
    if ch == "\t":
        return KEY_TAB
    cp = ord(ch)
    if 0x20 <= cp <= 0x7E or 0xA0 <= cp <= 0xFF:
        return cp
    return 0x01000000 | cp


class Portal:
    def __init__(self, bus: str = "SESSION", state_file: Path | None = None,
                 request_timeout: float = 120.0):
        from jeepney.io.threading import DBusRouter, open_dbus_connection

        self.conn = open_dbus_connection(bus=bus)
        self.router = DBusRouter(self.conn)
        self.timeout = request_timeout
        self.state_file = state_file
        self._tok = itertools.count(1)
        self._salt = os.urandom(3).hex()
        self.rd_session: str | None = None
        self.gs_session: str | None = None
        self._type_lock = threading.Lock()
        self._sig_thread: threading.Thread | None = None
        self._closed = False

    # -- plumbing ------------------------------------------------------------
    def _token(self) -> str:
        return f"pykeet{self._salt}_{next(self._tok)}"

    def _addr(self, iface: str):
        from jeepney import DBusAddress

        return DBusAddress(PORTAL_PATH, bus_name=PORTAL_NAME, interface=iface)

    def _call(self, iface: str, method: str, sig: str, body: tuple, timeout: float = 10.0):
        from jeepney import new_method_call

        reply = self.router.send_and_get_reply(
            new_method_call(self._addr(iface), method, sig, body), timeout=timeout)
        if reply.header.message_type.name == "error":
            raise PortalError(f"{method}: {reply.header.fields} {reply.body}")
        return reply

    def request(self, iface: str, method: str, sig: str, make_body, timeout: float | None = None):
        """Call a portal method that answers through a Request/Response signal.
        make_body(token) -> body tuple. Returns (response_code, results dict)."""
        from jeepney import MatchRule

        token = self._token()
        sender = self.conn.unique_name.lstrip(":").replace(".", "_")
        handle = f"{PORTAL_PATH}/request/{sender}/{token}"
        q: queue.Queue = queue.Queue()
        rule = MatchRule(type="signal", interface=REQUEST_IFACE, member="Response", path=handle)
        # subscribe BEFORE calling, or a fast answer can be missed
        from jeepney.bus_messages import message_bus

        self.router.send_and_get_reply(message_bus.AddMatch(rule), timeout=5)
        with self.router.filter(rule, queue=q):
            reply = self._call(iface, method, sig, make_body(token))
            real = reply.body[0] if reply.body else handle
            if real != handle:  # old portals pick their own path; listen there too
                rule2 = MatchRule(type="signal", interface=REQUEST_IFACE, member="Response",
                                  path=real)
                self.router.send_and_get_reply(message_bus.AddMatch(rule2), timeout=5)
                with self.router.filter(rule2, queue=q):
                    return self._await(q, method, timeout)
            return self._await(q, method, timeout)

    def _await(self, q, method, timeout):
        try:
            msg = q.get(timeout=timeout or self.timeout)
        except queue.Empty:
            raise PortalError(f"{method}: no answer (did the permission dialog stay open?)")
        code, results = msg.body
        return code, {k: v[1] for k, v in results.items()}

    # -- RemoteDesktop: typing -------------------------------------------------
    def _load_restore(self) -> str | None:
        try:
            return self.state_file.read_text().strip() or None if self.state_file else None
        except OSError:
            return None

    def _save_restore(self, token: str) -> None:
        if not self.state_file:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(token)
        except OSError:
            log.exception("could not save the portal restore token")

    def start_typing(self) -> None:
        """Ask for keyboard access. GNOME shows a permission dialog the first time."""
        code, res = self.request(
            RD_IFACE, "CreateSession", "a{sv}",
            lambda t: ({"handle_token": ("s", t), "session_handle_token": ("s", t + "s")},))
        if code != 0:
            raise PortalError(f"RemoteDesktop CreateSession refused (code {code})")
        session = res["session_handle"]

        def select(t):
            opts = {"handle_token": ("s", t), "types": ("u", 1), "persist_mode": ("u", 2)}
            tok = self._load_restore()
            if tok:
                opts["restore_token"] = ("s", tok)
            return (session, opts)

        code, _ = self.request(RD_IFACE, "SelectDevices", "oa{sv}", select)
        if code != 0:
            raise PortalError(f"RemoteDesktop SelectDevices refused (code {code})")
        code, res = self.request(
            RD_IFACE, "Start", "osa{sv}", lambda t: (session, "", {"handle_token": ("s", t)}))
        if code != 0:
            raise PortalError("keyboard permission was declined")
        if res.get("restore_token"):
            self._save_restore(res["restore_token"])
        self.rd_session = session
        log.info("portal typing ready")

    def _key(self, method: str, code: int, down: bool) -> None:
        self._call(RD_IFACE, method, "oa{sv}iu", (self.rd_session, {}, code, 1 if down else 0),
                   timeout=3)

    def type_text(self, text: str, delay: float = 0.002) -> None:
        if not self.rd_session:
            raise PortalError("typing not started")
        with self._type_lock:
            for ch in text:
                ks = keysym_for(ch)
                self._key("NotifyKeyboardKeysym", ks, True)
                self._key("NotifyKeyboardKeysym", ks, False)
                if delay:
                    time.sleep(delay)

    def paste_shortcut(self) -> None:
        """Press Ctrl+V (evdev key codes)."""
        if not self.rd_session:
            raise PortalError("typing not started")
        with self._type_lock:
            for code, down in ((EVDEV_CTRL, True), (EVDEV_V, True), (EVDEV_V, False),
                               (EVDEV_CTRL, False)):
                self._key("NotifyKeyboardKeycode", code, down)

    # -- GlobalShortcuts ---------------------------------------------------------
    def bind_shortcuts(self, shortcuts: dict[str, tuple[str, str]], on_activated, on_deactivated):
        """shortcuts: id -> (description, 'CTRL+space'). Callbacks get the shortcut id."""
        from jeepney import MatchRule
        from jeepney.bus_messages import message_bus

        code, res = self.request(
            GS_IFACE, "CreateSession", "a{sv}",
            lambda t: ({"handle_token": ("s", t), "session_handle_token": ("s", t + "g")},))
        if code != 0:
            raise PortalError(f"GlobalShortcuts CreateSession refused (code {code})")
        session = res["session_handle"]
        self.gs_session = session

        q: queue.Queue = queue.Queue()
        rule = MatchRule(type="signal", interface=GS_IFACE)
        self.router.send_and_get_reply(message_bus.AddMatch(rule), timeout=5)
        self._gs_filter = self.router.filter(rule, queue=q)

        entries = [(sid, {"description": ("s", desc), "preferred_trigger": ("s", trig)})
                   for sid, (desc, trig) in shortcuts.items()]
        code, res = self.request(
            GS_IFACE, "BindShortcuts", "oa(sa{sv})sa{sv}",
            lambda t: (session, entries, "", {"handle_token": ("s", t)}))
        if code != 0:
            raise PortalError(f"BindShortcuts refused (code {code})")

        def loop():
            while not self._closed:
                try:
                    msg = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                member = msg.header.fields.get(3)  # 3 = member
                try:
                    sess, sid = msg.body[0], msg.body[1]
                    if sess != session:
                        continue
                    if member == "Activated":
                        on_activated(sid)
                    elif member == "Deactivated":
                        on_deactivated(sid)
                except Exception:
                    log.exception("shortcut callback failed")

        self._sig_thread = threading.Thread(target=loop, daemon=True)
        self._sig_thread.start()
        log.info("portal shortcuts bound: %s", ", ".join(shortcuts))

    def close(self) -> None:
        self._closed = True
        try:
            self.router.close()
            self.conn.close()
        except Exception:
            pass
