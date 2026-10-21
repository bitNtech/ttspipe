"""FastAPI app factory. Run: uvicorn app.main:app --port 8200 (from backend/)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import rest, ws
from app.core import logs
from app.core.manager import Manager
from app.settings import VERSION, Settings


def create_app(settings: Settings | None = None, *, background_load: bool = True) -> FastAPI:
    s = settings or Settings.from_env()
    logs.setup(s.log_level, s.data_dir / "logs")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        m = Manager(s)
        app.state.manager = m
        await m.start(background=background_load)
        yield
        await m.shutdown()

    app = FastAPI(title="ttsPIPE", version=VERSION, lifespan=lifespan)
    app.include_router(rest.router)
    app.include_router(rest.root_router)
    app.include_router(ws.router)
    if s.static_dir.exists() and (s.static_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=s.static_dir, html=True), name="ui")
    return app


app = create_app()
