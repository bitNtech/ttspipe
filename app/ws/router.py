"""`GET /v1/tts/live` -- the WebSocket entrypoint.

Two independent tasks per connection: one drains inbound frames into
`Session.handle_event`, the other drains `Session.outbound` onto the socket.
Keeping them separate is what makes barge-in responsive -- a slow, ongoing
`send` for a big backlog of audio never blocks us from reading and acting on
the next `cancel`, and a slow synthesis never blocks reading either, since
`Session` itself hands the actual engine call off to its own worker task.

On a client disconnect or a protocol error, whichever task reacts to it
first drives cleanup: the other task is cancelled and `session.aclose()` is
called, which cancels any in-flight `engine.synthesize()` so the GPU is not
left grinding away for a socket nobody's listening on.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import suppress

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketState

from app.config import Settings
from app.engine.base import TTSEngine
from app.protocol import ProtocolError, decode_client_frame, finish_frame
from app.session import Session

logger = logging.getLogger("tts.ws")
router = APIRouter()

_active_sessions: set[str] = set()


def active_session_count() -> int:
    return len(_active_sessions)


@router.websocket("/v1/tts/live")
async def tts_live(websocket: WebSocket) -> None:
    settings: Settings = websocket.app.state.settings
    engine: TTSEngine | None = websocket.app.state.engine
    load_error = getattr(websocket.app.state, "engine_load_error", None)

    if len(_active_sessions) >= settings.max_sessions:
        await websocket.close(code=1013, reason="server at max_sessions capacity")
        return
    if engine is None:
        reason = f"engine unavailable: {load_error}" if load_error else "engine still starting"
        await websocket.close(code=1013, reason=reason[:120])
        return

    await websocket.accept()
    session_id = uuid.uuid4().hex
    _active_sessions.add(session_id)
    session = Session(
        engine=engine,
        settings=settings,
        session_id=session_id,
        voices=getattr(websocket.app.state, "voices", None),
    )
    logger.info("session %s: connected (%d active)", session_id, len(_active_sessions))

    async def receive_loop() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data is None:
                data = message.get("text", "")
            event = decode_client_frame(data)
            await session.handle_event(event)

    async def send_loop() -> None:
        while True:
            frame = await session.outbound.get()
            await websocket.send_bytes(frame)

    recv_task = asyncio.create_task(receive_loop(), name=f"ws-recv-{session_id}")
    send_task = asyncio.create_task(send_loop(), name=f"ws-send-{session_id}")

    protocol_error: ProtocolError | None = None
    try:
        done, _pending = await asyncio.wait(
            {recv_task, send_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if isinstance(exc, ProtocolError):
                protocol_error = exc
            elif exc is not None:
                logger.warning("session %s: %s ended with %r", session_id, task.get_name(), exc)
    finally:
        for task in (recv_task, send_task):
            task.cancel()
        for task in (recv_task, send_task):
            with suppress(asyncio.CancelledError, Exception):
                await task

        # Abandon any in-flight GPU work -- nothing is reading the outbound
        # queue anymore, whether this connection ended cleanly or not.
        await session.aclose()
        _active_sessions.discard(session_id)
        logger.info("session %s: closed (%d active)", session_id, len(_active_sessions))

        if websocket.client_state == WebSocketState.CONNECTED:
            with suppress(Exception):
                if protocol_error is not None:
                    await websocket.send_bytes(finish_frame("error", str(protocol_error)))
                    await websocket.close(code=1002)
                else:
                    await websocket.close(code=1000)
