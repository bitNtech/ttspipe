"""Module registry: @register + auto-discovery of app/modules/<kind>/<impl>.py (§4.2)."""
from __future__ import annotations

import importlib
import logging
import pkgutil

from .base import KINDS, BaseModule

log = logging.getLogger("ttspipe.registry")

_REGISTRY: dict[str, type[BaseModule]] = {}
_discovered = False

BUILTIN_FALLBACK = {
    "ingest": "plain", "normalize": "rules", "segment": "script", "chunker": "punct",
    "g2p": "lexicon", "prosody": "rules", "tts": "tone", "post": "passthrough",
}


def register(cls: type[BaseModule]) -> type[BaseModule]:
    info = cls.info
    assert info.kind in KINDS, info.id
    assert info.id == f"{info.kind}/{info.id.split('/', 1)[1]}", info.id
    _REGISTRY[info.id] = cls
    return cls


def discover() -> None:
    global _discovered
    if _discovered:
        return
    import app.modules as pkg

    for kind in KINDS:
        try:
            sub = importlib.import_module(f"{pkg.__name__}.{kind}")
        except ModuleNotFoundError:
            continue
        for m in pkgutil.iter_modules(sub.__path__):
            name = f"{sub.__name__}.{m.name}"
            try:
                importlib.import_module(name)
            except Exception as e:  # a broken module file must not take the server down
                log.error("module %s failed to import: %s", name, e)
    _discovered = True


def get(module_id: str) -> type[BaseModule]:
    discover()
    if module_id not in _REGISTRY:
        raise KeyError(f"unknown module {module_id}")
    return _REGISTRY[module_id]


def all_modules() -> list[type[BaseModule]]:
    discover()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def impls(kind: str) -> list[str]:
    discover()
    return [m.split("/", 1)[1] for m in sorted(_REGISTRY) if m.startswith(kind + "/")]
