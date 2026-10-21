"""A chunk stalled at the head of the queue gets a filler ahead of it; one try per chunk."""
import time
from collections import deque

import numpy as np

from app.core.base import Chunk
from app.core.playback import ChunkAudio, PlaybackManager


def _pm(source):
    cfg = {"jitter_ms": 100, "filler_enabled": True, "filler_after_ms": 0, "filler_gap_ms": 0}
    return PlaybackManager(session_id="s", rate=24000, frame_ms=20, cfg=cfg, emit=lambda *a: None, sink=lambda b: None,
                           post=lambda f: f, realtime=True, filler_source=source)


def test_filler_plays_before_stalled_chunk():
    calls = []

    def source(ca, stalled_ms):
        calls.append(ca.chunk.chunk_id)
        pcm = np.full(480, 0.1, np.float32)  # one 20 ms frame
        return ChunkAudio(seq=ca.seq, chunk=Chunk("c0.f", "u", 0, [], []), buf=deque([pcm]), buffered=480, total=480,
                          synth_done=True, filler_tried=True)

    pm = _pm(source)
    ca = ChunkAudio(seq=0, chunk=Chunk("c0", "u", 0, [], [], is_first=True))
    pm.add_chunk(ca)
    with pm.lock:
        f = pm._next_frame()
    assert f.kind == "speech" and f.chunk_id == "c0.f" and calls == ["c0"]
    assert pm.utts["u"]["chunks"] == 2
    with pm.lock:  # filler spent, real chunk still empty: silence, and no second filler
        for _ in range(5):
            f = pm._next_frame()
    assert calls == ["c0"] and (f is None or f.kind != "speech")


def test_no_filler_when_disabled_or_audio_ready():
    pm = _pm(lambda ca, ms: 1 / 0)
    pm.cfg["filler_enabled"] = False
    ca = ChunkAudio(seq=0, chunk=Chunk("c0", "u", 0, [], [], is_first=True))
    pm.add_chunk(ca)
    with pm.lock:
        pm._next_frame()
    pm.cfg["filler_enabled"] = True
    ca2 = ChunkAudio(seq=1, chunk=Chunk("c1", "u", 1, [], []), buf=deque([np.zeros(4800, np.float32)]), buffered=4800)
    pm.queue.clear()
    pm.add_chunk(ca2)
    with pm.lock:
        assert pm._next_frame().kind == "speech"
