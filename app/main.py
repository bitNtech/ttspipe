"""FastAPI app: wiring, lifespan, and the two small operational endpoints.

The WebSocket protocol lives in `app.ws.router` and the HTTP routes (voice
cloning, the active voice, the test page) in `app.http`; this module's job is
getting an engine built and
loaded in the background -- so `/health` answers, and stays cheap to poll,
while a GPU model is still coming up -- plus the operational endpoints:

    /health        liveness: 200 while loading, 503 only once loading failed
                   or the engine worker died
    /ready         readiness: 200 only when a request would be served now
    /status        human-readable engine and server state
    /metrics       Prometheus text format (app/metrics.py)
    /admin/drain   stop taking new sessions ahead of a restart

`app.engine.factory` (and everything behind it: torch, bitsandbytes,
fish_speech) is imported lazily, inside the lifespan, and only when
`TTS_ENGINE != "fake"`. Importing this module must work on a machine with
none of that installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import REPO_ROOT, get_settings
from app.engine.base import EngineStatus, TTSEngine
from app.http.router import router as http_router
from app.metrics import METRICS
from app.security import APIKeyMiddleware
from app.voice import ACTIVE_VOICE_ID, VoiceStore, warm_active_voice
from app.ws.router import active_session_count
from app.ws.router import router as ws_router

logger = logging.getLogger("tts")


class _JsonFormatter(logging.Formatter):
    """One JSON object per line, for log pipelines that parse rather than grep."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging(level_name: str, fmt: str = "text") -> None:
    """Make `TTS_LOG_LEVEL` mean something under uvicorn.

    `basicConfig` is defined to do nothing once the root logger already has a
    handler, and uvicorn installs its own before the app's lifespan ever runs -
    so setting the level through it worked when `app.main` was the entry point
    and was silently ignored in the only way this server is actually started.
    `run.sh` execs uvicorn. The levels below are therefore set on the loggers
    directly.

    `TTS_LOG_LEVEL=DEBUG` raises this application's loggers and leaves the root
    at INFO on purpose: torch, huggingface and the vendored fork are all noisy
    at DEBUG, and what an operator asking for it wants is the per-segment timing
    in `app.engine.worker` and `tts.session`, not a megabyte of somebody else's.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    root_level = max(level, logging.INFO)
    logging.basicConfig(level=root_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(root_level)
    if not logging.getLogger().handlers:  # pragma: no cover - uvicorn always has one
        logging.getLogger().addHandler(logging.StreamHandler())
    for name in ("tts", "app"):
        logging.getLogger(name).setLevel(level)
    if fmt == "json":
        formatter = _JsonFormatter()
        for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
            for handler in logging.getLogger(name).handlers:
                handler.setFormatter(formatter)


async def _build_engine() -> TTSEngine:
    if os.environ.get("TTS_ENGINE", "").strip().lower() == "fake":
        from app.engine.fake import FakeEngine

        return FakeEngine()
    from app.engine.factory import build_engine  # lazy: pulls in torch et al.

    return build_engine(get_settings())


async def _load_engine(app: FastAPI) -> None:
    """Runs as a background task so `/health` is reachable while this is slow."""
    # The engine reports ready before its own warm-up synthesis and the voice
    # prefill below finish; a turn routed in during either queues behind it
    # (seconds on the CPU profile), so /ready says "warming" until both are done.
    app.state.warming = True
    try:
        engine = await _build_engine()
        app.state.engine = engine
        await engine.load()
        logger.info("engine ready: %s", engine.status().as_dict())
        # A voice set before this restart is about to be used cold, so prefill
        # its prompt now rather than on whatever turn arrives first.
        await warm_active_voice(app.state)
    except Exception as exc:  # pragma: no cover - depends on the real backend
        logger.exception("engine failed to load")
        app.state.engine_load_error = str(exc)
    finally:
        app.state.warming = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level, settings.log_format)
    app.state.settings = settings
    app.state.start_time = time.monotonic()
    app.state.engine = None
    app.state.engine_load_error = None
    app.state.draining = False
    app.state.warming = False
    if not settings.api_keys and settings.host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "TTS_API_KEYS is unset and the server binds %s: anyone who can reach "
            "this port can synthesize and replace the active voice",
            settings.host,
        )
    # Every in-flight segment and speculative prefill parks one executor
    # thread on its worker queue for as long as it runs. The default pool is
    # min(32, cores + 4), which `max_sessions` busy sessions plus their hints
    # can exhaust - and then a new turn's first chunk waits for a thread, not
    # for the engine. Size it to the load the server admits.
    executor = ThreadPoolExecutor(
        max_workers=max(32, settings.max_sessions * 4), thread_name_prefix="tts-io"
    )
    asyncio.get_running_loop().set_default_executor(executor)
    # Built before the engine, and independently of it: setting a voice is
    # pure I/O, so it must work (and survive a restart) while the model is
    # still loading.
    app.state.voices = VoiceStore(REPO_ROOT / "references" / ACTIVE_VOICE_ID)

    load_task = asyncio.create_task(_load_engine(app))
    try:
        yield
    finally:
        load_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await load_task
        engine: TTSEngine | None = app.state.engine
        if engine is not None:
            await engine.aclose()


def _engine_problem(app: FastAPI) -> str | None:
    """Why the engine cannot serve at all, or None. Loading is not a problem."""
    if app.state.engine_load_error is not None:
        return app.state.engine_load_error
    engine: TTSEngine | None = app.state.engine
    if engine is not None and getattr(engine, "worker_dead", False):
        return "engine worker thread has exited"
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="TTS streaming server", lifespan=lifespan)
    app.include_router(ws_router)
    app.include_router(http_router)

    # Read at import time, like the rest of the wiring: middleware cannot be
    # added once the app has started.
    settings = get_settings()
    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-Id", "X-Audio-Format", "X-Sample-Rate", "X-Seed", "X-Warning"],
        )
    # Added last so it runs first: a refused request never reaches CORS or a route.
    app.add_middleware(APIKeyMiddleware, keys=settings.api_keys)

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500, content={"error": str(exc), "type": type(exc).__name__}
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        engine: TTSEngine | None = app.state.engine
        error = _engine_problem(app)
        if error is not None:
            return JSONResponse(status_code=503, content={"status": "error", "error": error})
        if engine is None or not engine.ready:
            return JSONResponse(status_code=200, content={"status": "loading"})
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """Route traffic here? Unlike /health, "loading" and "draining" are no."""
        engine: TTSEngine | None = app.state.engine
        error = _engine_problem(app)
        if error is not None:
            return JSONResponse(status_code=503, content={"ready": False, "reason": error})
        if engine is None or not engine.ready:
            return JSONResponse(status_code=503, content={"ready": False, "reason": "loading"})
        if app.state.draining:
            return JSONResponse(status_code=503, content={"ready": False, "reason": "draining"})
        if app.state.warming:
            return JSONResponse(status_code=503, content={"ready": False, "reason": "warming"})
        return JSONResponse(status_code=200, content={"ready": True})

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        engine: TTSEngine | None = app.state.engine
        status_ = engine.status() if engine is not None else None
        body = METRICS.render(
            status_,
            ready=bool(engine is not None and engine.ready and _engine_problem(app) is None),
            draining=app.state.draining,
            uptime_s=time.monotonic() - app.state.start_time,
        )
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4")

    @app.post("/admin/drain")
    async def drain() -> JSONResponse:
        """Refuse new WebSocket sessions; let the open ones finish.

        Call it from a pre-stop hook, then wait for `active_sessions` to reach
        0 (or a deadline) before sending SIGTERM - a SIGTERM alone closes every
        socket at once, which cuts off whatever each caller was hearing.
        Irreversible short of a restart, on purpose.
        """
        app.state.draining = True
        logger.warning("drain requested: refusing new sessions (%d open)", active_session_count())
        return JSONResponse(
            status_code=200, content={"draining": True, "active_sessions": active_session_count()}
        )

    @app.get("/status")
    async def status() -> JSONResponse:
        engine: TTSEngine | None = app.state.engine
        payload = engine.status().as_dict() if engine is not None else EngineStatus().as_dict()
        voices: VoiceStore | None = getattr(app.state, "voices", None)
        active_voice = voices.get() if voices is not None else None
        payload.update(
            {
                "uptime_seconds": time.monotonic() - app.state.start_time,
                "active_sessions": active_session_count(),
                "draining": app.state.draining,
                "active_voice": active_voice.as_dict() if active_voice else None,
                "settings": app.state.settings.as_dict(),
            }
        )
        return JSONResponse(status_code=200, content=payload)

    return app


app = create_app()
