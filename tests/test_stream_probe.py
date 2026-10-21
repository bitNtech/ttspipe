"""The continuity measurements in `scripts/stream_probe.py`.

`min_preroll_ms` is the number the whole tool exists to produce - the smallest
buffer a client could start playing from and never run dry - and everything
about how this server should be driven follows from it. A number like that is
worth pinning down: it is defined as a boundary, so the test is that it *is*
one, gapless at or above it and not gapless below.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_probe():
    """Import `scripts/stream_probe.py`, which is a script rather than a package.

    It has to be registered in `sys.modules` before it is executed: `@dataclass`
    resolves a class's annotations through `sys.modules[cls.__module__]`, and
    raises on the `None` it finds there otherwise.
    """
    spec = importlib.util.spec_from_file_location(
        "stream_probe", REPO_ROOT / "scripts" / "stream_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()
RATE = 44100


def _turn(blocks, rate=RATE):
    """blocks: (arrival_seconds, audio_seconds) pairs, in order."""
    turn = probe.Turn(started_at=0.0, sample_rate=rate)
    for at, seconds in blocks:
        turn.frames.append(
            probe.Frame(
                at=at,
                samples=np.zeros(int(seconds * rate), dtype=np.float32),
                reported_ms=0.0,
            )
        )
    return turn


def test_faster_than_realtime_needs_no_preroll():
    # 200 ms of audio every 80 ms: the queue only ever grows.
    turn = _turn([(0.1 + i * 0.08, 0.2) for i in range(10)])
    assert probe.playout(turn, 0.0)["min_preroll_ms"] == pytest.approx(0.0, abs=1e-6)
    assert probe.playout(turn, 0.0)["stalls"] == 0


def test_min_preroll_is_the_boundary_between_gapless_and_not():
    # 46 ms of audio every 120 ms - roughly this server on a GTX 1650.
    turn = _turn([(0.5 + i * 0.12, 0.0464) for i in range(40)])
    need = probe.playout(turn, 0.0)["min_preroll_ms"] / 1000.0
    assert need > 0

    assert probe.playout(turn, need)["stalls"] == 0
    assert probe.playout(turn, need * 1.5)["stalls"] == 0
    # Just below it, the player runs dry at least once.
    assert probe.playout(turn, need * 0.9)["stalls"] >= 1


def test_min_preroll_ignores_the_wait_for_the_first_frame():
    """Time-to-first-audio is latency, not a shortfall the buffer has to cover.

    A client starts its buffer when the first frame lands, so however long it
    waited for that frame cannot be part of what it then has to hold back.
    """
    fast = _turn([(0.2 + i * 0.12, 0.0464) for i in range(20)])
    slow = _turn([(9.0 + i * 0.12, 0.0464) for i in range(20)])
    assert probe.playout(fast, 0.0)["min_preroll_ms"] == pytest.approx(
        probe.playout(slow, 0.0)["min_preroll_ms"]
    )


def test_stall_total_shrinks_as_the_preroll_grows():
    turn = _turn([(0.5 + i * 0.12, 0.0464) for i in range(40)])
    deep = probe.playout(turn, 0.5)
    shallow = probe.playout(turn, 0.05)
    assert deep["stall_ms"] < shallow["stall_ms"]
    assert deep["audio_s"] == pytest.approx(shallow["audio_s"])


def test_seams_flags_a_discontinuity_at_a_frame_join():
    """A click is a jump at a join sharper than anything inside the frames.

    The clean case is one continuous tone cut into frames, which is what a
    correctly streamed utterance is: the joins are ordinary sample steps. The
    clicky case offsets every other frame, which is what a block decoded without
    its left context looks like.
    """
    rate = RATE
    size = rate // 4
    t = np.arange(size * 4) / rate
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def turn_of(pieces):
        turn = probe.Turn(started_at=0.0, sample_rate=rate)
        for i, piece in enumerate(pieces):
            turn.frames.append(probe.Frame(at=i * 0.1, samples=piece, reported_ms=0.0))
        return turn

    cut = [tone[i * size : (i + 1) * size].copy() for i in range(4)]
    clean = turn_of(cut)
    clicky = turn_of([p + (0.5 if i % 2 else 0.0) for i, p in enumerate(cut)])

    # A continuous signal's joins are indistinguishable from its own steps, so
    # the ratio sits at about 1; a step at the join dwarfs them.
    assert probe.seams(clean)["join_max_over_p99_99"] < 2.0
    assert probe.seams(clicky)["join_max_over_p99_99"] > 10.0


def test_silence_runs_are_reported_with_their_position():
    rate = RATE
    audio = np.concatenate(
        [
            np.full(rate // 10, 0.2, dtype=np.float32),
            np.zeros(rate // 2, dtype=np.float32),  # 500 ms of silence
            np.full(rate // 10, 0.2, dtype=np.float32),
        ]
    )
    turn = probe.Turn(started_at=0.0, sample_rate=rate)
    turn.frames.append(probe.Frame(at=0.0, samples=audio, reported_ms=0.0))
    found = probe.silence_runs(turn)
    assert found["count"] == 1
    assert found["longest_ms"] == pytest.approx(500.0, abs=1.0)
    assert found["longest_at_s"] == pytest.approx(0.1, abs=0.01)
