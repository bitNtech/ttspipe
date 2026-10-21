"""WebSocket endpoints (ARCHITECTURE.md §8)."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.events import GLOBAL_TYPES, dumps, envelope

router = APIRouter()
log = logging.getLogger("ttspipe.ws")


@router.websocket("/ws/tts")
async def ws_tts(ws: WebSocket, session_id: str | None = None, label: str = "", voice: str | None = None):
    m = ws.app.state.manager
    await ws.accept()
    old = m.sessions.get(session_id) if session_id else None
    if old is not None and getattr(old, "ws", None) is not None:
        try:
            await old.ws.close(code=4409)
        except Exception:
            pass
    loop = asyncio.get_running_loop()
    holder: dict = {}

    def subscribe(sess):
        holder["sub"] = m.bus.subscribe(loop, lambda ev: ev["session_id"] == sess.id and ev["type"] not in GLOBAL_TYPES)

    sess = await m.open_session(session_id, label, voice, before_start=subscribe)
    sess.ws = ws
    sub = holder["sub"]

    async def forward():
        while True:
            for ev in await sub.get_batch():
                sess.out_q.put_nowait(dumps(ev))

    async def writer():
        while True:
            item = await sess.out_q.get()
            if isinstance(item, (bytes, bytearray)):
                await ws.send_bytes(item)
            else:
                await ws.send_text(item)

    tasks = [asyncio.create_task(forward()), asyncio.create_task(writer())]
    reason = "disconnect"
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                t = msg.get("type")
            except (ValueError, AttributeError):
                sess.emit("error", {"where": "ws", "message": "invalid JSON", "kind": None, "impl": None, "fatal": False})
                continue
            if t == "text":
                if isinstance(msg.get("delta"), str):
                    sess.text(msg["delta"], msg.get("utterance_id"))
            elif t == "end_utterance":
                sess.end_utterance(msg.get("utterance_id"))
            elif t == "speak":
                if isinstance(msg.get("text"), str):
                    sess.speak(msg["text"], msg.get("utterance_id"))
            elif t in ("duck", "stop", "resume"):
                getattr(sess, t)("client")
            elif t == "reset":
                sess.reset()
            elif t == "voice":
                sess.set_voice(msg.get("voice"), msg.get("speed"), msg.get("pitch"))
            elif t == "ping":
                sess.out_q.put_nowait(dumps(envelope("pong", {"t": msg.get("t"), "server_ts": loop.time()},
                                                     sess.id, sess.playback.t_out_ms)))
            else:
                sess.emit("error", {"where": "ws", "message": f"unknown message type {t!r}", "kind": None,
                                    "impl": None, "fatal": False})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        reason = f"error: {e}"
        log.warning("ws/tts %s: %s", sess.id, e)
    finally:
        for t in tasks:
            t.cancel()
        m.bus.unsubscribe(sub)
        sess.connected = False
        if m.sessions.get(sess.id) is sess:
            await m.close_session(sess.id, reason)


@router.websocket("/ws/events")
async def ws_events(ws: WebSocket, session_id: str = "*"):
    m = ws.app.state.manager
    await ws.accept()
    loop = asyncio.get_running_loop()

    def filt(ev):
        return session_id == "*" or ev["session_id"] in (None, session_id) or ev["type"] in GLOBAL_TYPES

    sub = m.bus.subscribe(loop, filt)

    async def writer():
        while True:
            for ev in await sub.get_batch():
                await ws.send_text(dumps(ev))

    task = asyncio.create_task(writer())
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("type") == "ping":
                await ws.send_text(dumps(envelope("pong", {"t": msg.get("t"), "server_ts": loop.time()})))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug("ws/events closed: %s", e)
    finally:
        task.cancel()
        m.bus.unsubscribe(sub)
