"""Stage slots: one single-thread executor per slot, hot swap, error counting, builtin fallback (§5.2, §5.6)."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from . import registry
from .base import BaseModule
from .events import Bus
from .metrics import Metrics

log = logging.getLogger("ttspipe.runtime")
MAX_CONSECUTIVE_ERRORS = 3
# One module load at a time, process-wide: concurrent first imports of torch / onnxruntime from two loader
# threads can deadlock on import locks, and parallel loads only fight over the same cores anyway.
LOAD_LOCK = threading.Lock()


class StageSlot:
    def __init__(self, kind: str, bus: Bus, metrics: Metrics, models_dir: Path, name: str | None = None):
        self.kind = kind
        self.name = name or kind  # "tts" or "shadow"
        self.bus, self.metrics, self.models_dir = bus, metrics, models_dir
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"stage-{self.name}")
        self.module: BaseModule | None = None
        self.impl: str | None = None  # active impl (may be the builtin fallback)
        self.configured_impl: str | None = None
        self.configured_params: dict = {}
        self.enabled = True
        self.status = "unloaded"
        self.error: str | None = None
        self.fallback = False
        self.load_ms: float | None = None
        self.generation = 0
        self.consecutive_errors = 0
        self.busy = 0
        self.lock = threading.Lock()
        self._load_seq = 0
        self.global_emit = True  # False for auxiliary slots (compare/preview)

    # ---- status -------------------------------------------------------------------------------

    def status_dict(self) -> dict:
        s = self.metrics.stage_summary(self.name) if self.global_emit else {}
        return {
            "kind": self.kind, "impl": self.impl, "configured_impl": self.configured_impl,
            "enabled": self.enabled, "status": self.status if self.enabled else "disabled",
            "error": self.error, "fallback": self.fallback, "busy": self.busy > 0,
            "params": dict(self.module.params) if self.module else dict(self.configured_params),
            "load_ms": self.load_ms, "count": s.get("count", 0), "errors": s.get("errors", 0),
            "consecutive_errors": self.consecutive_errors, "last_ms": s.get("last_ms"),
            "p50_ms": s.get("p50_ms"), "p95_ms": s.get("p95_ms"), "name": self.name,
        }

    def _emit_status(self) -> None:
        if self.global_emit:
            self.bus.emit("stage.status", self.status_dict())

    # ---- configuration / hot swap -------------------------------------------------------------

    def configure(self, impl: str, params: dict | None, enabled: bool = True, *, background: bool = True,
                  on_done: Callable[[], None] | None = None) -> None:
        params = dict(params or {})
        self.enabled = enabled
        same = (impl == self.configured_impl and self.module is not None and self.impl == impl
                and self.status == "ready" and not self.fallback)
        self.configured_impl, self.configured_params = impl, params
        if same:
            # ordered after queued jobs on the slot thread; reload only if the module refuses live params
            def done(fut):
                if not fut.exception() and fut.result():
                    self._emit_status()
                    if on_done:
                        on_done()
                else:
                    self._reload(impl, params, background, on_done)

            self.executor.submit(self.module.update_params, params).add_done_callback(done)
            return
        self._reload(impl, params, background, on_done)

    def _reload(self, impl: str, params: dict, background: bool, on_done: Callable[[], None] | None) -> None:
        self._load_seq += 1
        seq = self._load_seq
        cls = registry.get(f"{self.kind}/{impl}")
        if (cls.info.builtin and "router" not in cls.info.tags) or not background:  # routers load sub-engines: background
            self._load(seq, impl, params)
            if on_done:
                on_done()
            return
        self.status, self.error = "loading", None
        self._emit_status()

        def run():
            self._load(seq, impl, params)
            if on_done:
                on_done()

        threading.Thread(target=run, name=f"load-{self.name}-{impl}", daemon=True).start()

    def _instantiate(self, impl: str, params: dict) -> BaseModule:
        cls = registry.get(f"{self.kind}/{impl}")
        ok, reason = cls.availability()
        if not ok:
            raise RuntimeError(reason)
        mod = cls(params)
        mod.models_dir = self.models_dir / self.kind / impl
        mod.load()
        mod.loaded = True
        return mod

    def _load(self, seq: int, impl: str, params: dict) -> None:
        if "remote" in registry.get(f"{self.kind}/{impl}").info.tags:  # no heavy imports; may wait minutes for a server
            self._load_locked(seq, impl, params)
            return
        with LOAD_LOCK:
            self._load_locked(seq, impl, params)

    def _load_locked(self, seq: int, impl: str, params: dict) -> None:
        if seq != self._load_seq:  # superseded while waiting for the lock
            return
        t0 = time.perf_counter()
        error = None
        try:
            mod = self._instantiate(impl, params)
            active, fallback = impl, False
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            log.error("%s/%s failed to load: %s — using builtin %s", self.kind, impl, error,
                      registry.BUILTIN_FALLBACK[self.kind])
            fb = registry.BUILTIN_FALLBACK[self.kind]
            mod = registry.get(f"{self.kind}/{fb}")({})
            mod.load()
            active, fallback = fb, True
        if seq != self._load_seq:  # superseded by a newer configure() or shutdown()
            self._unload(mod)
            return
        self._swap(mod, active, fallback, error, (time.perf_counter() - t0) * 1000)

    def _swap(self, mod: BaseModule, active: str, fallback: bool, error: str | None, load_ms: float | None) -> None:
        with self.lock:
            old = self.module
            self.module, self.impl, self.fallback, self.error = mod, active, fallback, error
            self.status = "ready"
            self.load_ms = round(load_ms, 1) if load_ms is not None else None
            self.generation += 1
            self.consecutive_errors = 0
        if old is not None and old is not mod:
            self._unload(old)  # runs after already-queued jobs drain
        self._emit_status()

    def unload_to_builtin(self, reason: str | None = None, status: str = "ready") -> None:
        fb = registry.BUILTIN_FALLBACK[self.kind]
        mod = registry.get(f"{self.kind}/{fb}")({})
        mod.load()
        self._swap(mod, fb, fb != self.configured_impl, reason, 0)
        if status != "ready":
            self.status = status
            self._emit_status()

    def _unload(self, mod: BaseModule) -> None:
        try:
            self.executor.submit(mod.unload)
        except RuntimeError:  # executor already shut down (a background load finished late)
            mod.unload()

    def shutdown(self) -> None:
        self._load_seq += 1  # in-flight background loads see themselves superseded and unload
        if self.module is not None:
            self._unload(self.module)
        self.executor.shutdown(wait=False, cancel_futures=True)

    # ---- calls --------------------------------------------------------------------------------

    def snapshot(self) -> tuple[BaseModule, int]:
        with self.lock:
            return self.module, self.generation

    def _record(self, mod_impl: str, ms: float, ok: bool, err: BaseException | None, ctx: dict) -> None:
        if not self.global_emit:
            return
        self.metrics.stage_call(self.name, mod_impl, ms, ok)
        self.bus.emit("stage.activity", {"kind": self.name, "impl": mod_impl, "ms": round(ms, 2), "ok": ok,
                                         "utterance_id": ctx.get("utterance_id"), "chunk_id": ctx.get("chunk_id")},
                      session_id=ctx.get("session_id"), t_out_ms=ctx.get("t_out_ms"))
        if ok:
            self.consecutive_errors = 0
            return
        self.consecutive_errors += 1
        msg = f"{self.kind}/{mod_impl}: {type(err).__name__}: {err}"
        self.bus.emit("error", {"where": f"stage.{self.name}", "message": msg, "kind": self.kind,
                                "impl": mod_impl, "fatal": False}, session_id=ctx.get("session_id"),
                      t_out_ms=ctx.get("t_out_ms"))
        log.error("%s\n%s", msg, "".join(traceback.format_exception(err))[-2000:])
        if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS and self.module is not None \
                and self.impl != registry.BUILTIN_FALLBACK[self.kind]:
            log.error("%s: %d consecutive errors — marking error and falling back", self.name, self.consecutive_errors)
            failing = self.impl
            self.unload_to_builtin(f"{failing}: {self.consecutive_errors} consecutive errors; last: {err}")
            self.status = "error"
            self.fallback = True
            self._emit_status()

    def run_job(self, mod: BaseModule, fn: Callable[[BaseModule], Any], ctx: dict) -> Any:
        """Executes on the slot thread."""
        self.busy += 1
        t0 = time.perf_counter()
        try:
            r = fn(mod)
        except BaseException as e:
            self.busy -= 1
            self._record(mod.info.impl, (time.perf_counter() - t0) * 1000, False, e, ctx)
            raise
        self.busy -= 1
        self._record(mod.info.impl, (time.perf_counter() - t0) * 1000, True, None, ctx)
        return r

    async def call(self, fn: Callable[[BaseModule], Any], ctx: dict | None = None, mod: BaseModule | None = None) -> Any:
        if mod is None:
            mod, _ = self.snapshot()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.run_job, mod, fn, ctx or {})

    async def raw(self, fn: Callable[[], Any]) -> Any:
        """Run an untimed helper (new_state / close_state) on the slot thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, fn)


class StateHolder:
    """Per-session module state bound to a slot generation (recreated lazily after a hot swap)."""

    def __init__(self, slot: StageSlot):
        self.slot = slot
        self.gen = -1
        self.mod: BaseModule | None = None
        self.state: Any = None

    async def get(self) -> tuple[BaseModule, Any]:
        mod, gen = self.slot.snapshot()
        while mod is None:  # slot's first module still loading
            await asyncio.sleep(0.05)
            mod, gen = self.slot.snapshot()
        if gen != self.gen or self.mod is not mod:
            self.state = await self.slot.raw(mod.new_state)
            self.mod, self.gen = mod, gen
        return self.mod, self.state

    def get_sync(self) -> tuple[BaseModule, Any]:
        """For the pacer thread (post stage): state created inline."""
        mod, gen = self.slot.snapshot()
        if gen != self.gen or self.mod is not mod:
            self.state = mod.new_state()
            self.mod, self.gen = mod, gen
        return self.mod, self.state
