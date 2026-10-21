"""Pipeline graph for the dashboard (§9.3)."""
from __future__ import annotations

from .base import KINDS

LABELS = {"text_in": "TEXT IN", "audio_out": "AUDIO OUT", "shadow_tts": "SHADOW TTS"}


def build(cfg: dict, stages: dict[str, dict]) -> dict:
    nodes = [{"id": "text_in", "kind": "source", "label": "TEXT IN", "impl": None}]
    for k in KINDS:
        nodes.append({"id": k, "kind": k, "label": k.upper(), "impl": stages.get(k, {}).get("impl")})
    nodes.append({"id": "audio_out", "kind": "sink", "label": "AUDIO OUT", "impl": None})
    chain = ["text_in", *KINDS, "audio_out"]
    edges = [{"from": a, "to": b, "dashed": False} for a, b in zip(chain, chain[1:])]
    shadow = ((cfg.get("graph") or {}).get("shadows") or {}).get("tts")
    if shadow:
        nodes.insert(len(nodes) - 2, {"id": "shadow_tts", "kind": "shadow", "label": "SHADOW TTS",
                                      "impl": shadow.get("impl")})
        edges.append({"from": "prosody", "to": "shadow_tts", "dashed": True})
    return {"nodes": nodes, "edges": edges, "positions": (cfg.get("graph") or {}).get("positions") or {}}
