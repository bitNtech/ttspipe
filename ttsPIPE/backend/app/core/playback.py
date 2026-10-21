"""PlaybackManager: jitter buffer, real-time pacer thread, duck/stop gain, state machine (§5.4, §5.5)."""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .audio import frame_samples, levels, to_pcm16
from .base import Chunk, Frame, WordTiming

log = logging.getLogger("ttspipe.playback")

STATES = ("idle", "buffering", "synthesizing", "speaking", "ducked", "interrupted", "done")


@dataclass
class ChunkAudio:
    seq: int
    chunk: Chunk
    pause_after_ms: int = 0
    buf: deque = field(default_factory=deque)
    buffered: int = 0
    total: int = 0
    consumed: int = 0
    synth_done: bool = False
    failed: bool = False
    started: bool = False
    finished: bool = False
    timings: list[WordTiming] = field(default_factory=list)
    timings_exact: bool = False
    pause_frames: int = -1
    t_out_start: int | None = None
    t_out_end: int | None = None
    word_index: int = -1
    rate: int = 24000  # playback (output) rate of buf
    head_since: float | None = None  # monotonic time it first waited at the queue head with too little audio
    filler_tried: bool = False

    @property
    def utterance_id(self) -> str:
        return self.chunk.utterance_id


class PlaybackManager:
    def __init__(self, *, session_id: str, rate: int, frame_ms: int, cfg: dict, emit: Callable,
                 sink: Callable[[bytes], None], post: Callable[[Frame], Frame | None],
                 realtime: bool = True, paused: Callable[[], bool] = lambda: False,
                 on_first_frame: Callable[[str], None] = lambda uid: None,
                 on_progress: Callable[[ChunkAudio, int], None] = lambda ca, i: None,
                 on_utterance_done: Callable[[str], None] = lambda uid: None,
                 on_interrupted: Callable[[dict], None] = lambda info: None,
                 ref_sink: Callable[[bytes], None] | None = None,
                 filler_source: Callable[[ChunkAudio, int], ChunkAudio | None] | None = None):
        self.sid, self.rate, self.frame_ms = session_id, rate, frame_ms
        self.n = frame_samples(rate, frame_ms)
        self.cfg = dict(cfg)
        self.emit, self.sink, self.post, self.realtime, self.paused = emit, sink, post, realtime, paused
        self.on_first_frame, self.on_progress = on_first_frame, on_progress
        self.on_utterance_done, self.on_interrupted, self.ref_sink = on_utterance_done, on_interrupted, ref_sink
        self.filler_source = filler_source
        self.lock = threading.RLock()
        self.queue: deque[ChunkAudio] = deque()
        self.utts: dict[str, dict] = {}  # uid → {complete, chunks, finished, first_sent}
        self.state, self.prev_state = "idle", None
        self.tick = 0
        self.gain_db = 0.0
        self.ducked = False
        self.duck_since = 0.0
        self.stopping: dict | None = None
        self.underrun = False
        self.gap_ms = 0
        self.renders: dict[str, list[np.ndarray]] = {}
        self.last_frames: deque[np.ndarray] = deque(maxlen=2)
        self.level_zero_sent = True
        self.running = False
        self.thread: threading.Thread | None = None
        self._pending_calls: list[Callable[[], None]] = []
        self.frames_sent = 0
        self.wake = threading.Event()

    # ---- clock / state ------------------------------------------------------------------------

    @property
    def t_out_ms(self) -> int:
        return self.tick * self.frame_ms

    def _ev(self, type_: str, data: dict) -> None:
        self.emit(type_, data)

    def set_state(self, new: str, reason: str = "", uid: str | None = None) -> None:
        with self.lock:
            if new == self.state:
                return
            prev, self.state = self.state, new
            self.prev_state = prev
        self._ev("playback.state", {"state": new, "prev": prev, "reason": reason, "utterance_id": uid})

    def update_cfg(self, cfg: dict) -> None:
        with self.lock:
            self.cfg.update(cfg)

    # ---- producer side (session) --------------------------------------------------------------

    def note_text(self, uid: str) -> None:
        with self.lock:
            if self.state == "idle":
                self.set_state("buffering", "text", uid)

    def add_chunk(self, ca: ChunkAudio) -> None:
        with self.lock:
            u = self.utts.setdefault(ca.utterance_id, {"complete": False, "chunks": 0, "finished": 0, "first_sent": False})
            u["chunks"] += 1
            self.queue.append(ca)
        self.wake.set()

    def synth_started(self, uid: str) -> None:
        with self.lock:
            if self.state in ("idle", "buffering"):
                self.set_state("synthesizing", "synth_start", uid)

    def add_audio(self, ca: ChunkAudio, pcm: np.ndarray, timings: list[WordTiming] | None) -> None:
        with self.lock:
            if len(pcm):
                ca.buf.append(pcm)
                ca.buffered += len(pcm)
                ca.total += len(pcm)
            if timings:
                ca.timings.extend(timings)
                ca.timings_exact = True
        self.wake.set()

    def finish_synth(self, ca: ChunkAudio, failed: bool = False) -> None:
        with self.lock:
            ca.synth_done, ca.failed = True, failed
            if not ca.timings_exact:
                ca.timings = estimate_timings(ca)
        self.wake.set()

    def utterance_complete(self, uid: str) -> None:
        """No more chunks will be added for uid."""
        with self.lock:
            u = self.utts.setdefault(uid, {"complete": False, "chunks": 0, "finished": 0, "first_sent": False})
            u["complete"] = True
            if u["finished"] >= u["chunks"]:
                self._utterance_finished(uid)
        self._run_pending()

    def head_seq(self) -> int | None:
        with self.lock:
            for ca in self.queue:
                if not ca.finished:
                    return ca.seq
            return None

    # ---- duplex -------------------------------------------------------------------------------

    def duck(self, source: str = "client") -> None:
        with self.lock:
            self.ducked, self.duck_since = True, time.monotonic()
            if self.state == "speaking":
                self.set_state("ducked", source)
        self._ev("duck", {"action": "duck", "gain_db": -float(self.cfg["duck_db"]), "source": source})

    def resume(self, source: str = "client", action: str = "resume") -> None:
        with self.lock:
            was = self.ducked
            self.ducked = False
            if self.state == "ducked":
                self.set_state("speaking", "timeout" if action == "release" else source)
        if was:
            self._ev("duck", {"action": action, "gain_db": 0.0, "source": source})

    def stop(self, source: str = "client") -> bool:
        """Fade out and cancel everything. Returns False if there was nothing to stop."""
        with self.lock:
            if self.stopping:
                return True
            active = self.state not in ("idle", "done", "interrupted") or any(not c.finished for c in self.queue)
            if not active:
                return False
            playing = self.state in ("speaking", "ducked")
            fade_frames = max(1, math.ceil(self.cfg["stop_fade_ms"] / self.frame_ms)) if playing else 0
            self.stopping = {"source": source, "frames": fade_frames, "left": fade_frames}
            if not playing:
                self._finish_stop()
        self._run_pending()
        return True

    def _finish_stop(self) -> None:
        info = {"at_ms": self.t_out_ms, "source": self.stopping["source"], "chunks": {}}
        uids = []
        for ca in self.queue:
            info["chunks"][ca.chunk.chunk_id] = {
                "played_ms": ca.consumed * 1000.0 / self.rate, "started": ca.started, "finished": ca.finished,
                "timings": list(ca.timings) if ca.synth_done or ca.timings_exact else estimate_timings(ca),
            }
            if ca.utterance_id not in uids:
                uids.append(ca.utterance_id)
        info["utterance_ids"] = uids or [u for u, v in self.utts.items() if not v.get("closed")]
        self.queue.clear()
        for u in self.utts.values():
            u["closed"] = True
        self.stopping = None
        self.underrun, self.gap_ms = False, 0
        self.ducked = False
        uid = info["utterance_ids"][0] if info["utterance_ids"] else None
        self.set_state("interrupted", info["source"], uid)
        self._pending_calls.append(lambda: self.on_interrupted(info))
        self.set_state("idle", "interrupted", uid)

    def reset_utterances(self) -> None:
        with self.lock:
            self.utts = {k: v for k, v in self.utts.items() if not v.get("closed")}

    # ---- pacer --------------------------------------------------------------------------------

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._loop, name=f"pacer-{self.sid}", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.running = False
        self.wake.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def _loop(self) -> None:
        try:
            if self.realtime:
                self._loop_realtime()
            else:
                self._loop_offline()
        except Exception:
            log.exception("pacer crashed (session %s)", self.sid)

    def _loop_realtime(self) -> None:
        period = self.frame_ms / 1000.0
        deadline = time.perf_counter()
        while self.running:
            deadline += period
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.1:
                deadline = time.perf_counter()  # stalled > 100 ms: resync instead of bursting
            self._tick()
            self.tick += 1

    def _loop_offline(self) -> None:
        while self.running:
            if self._tick():
                self.tick += 1
            else:
                self.wake.wait(0.002)
                self.wake.clear()

    def _tick(self) -> bool:
        if self.paused():
            return False
        sent = False
        for _ in range(50):  # trimmed frames do not cost a tick
            with self.lock:
                frame = self._next_frame()
                ca = self.queue[0] if self.queue else None
            self._run_pending()
            if frame is None:
                break
            out = self.post(frame)
            if out is None:
                continue
            self._send(out, ca)
            sent = True
            break
        self._housekeeping(sent)
        return sent

    def _housekeeping(self, sent: bool) -> None:
        if self.ducked and self.cfg.get("duck_release_ms", 0) > 0 and \
                time.monotonic() - self.duck_since > self.cfg["duck_release_ms"] / 1000:
            self.resume("timeout", action="release")
        every = max(1, int(self.cfg.get("level_every_ticks", 2)))
        if self.tick % every == 0:
            if sent or not self.level_zero_sent:
                x = np.concatenate(list(self.last_frames)) if sent and self.last_frames else np.zeros(self.n, np.float32)
                lv = levels(x)
                lv["gain_db"] = round(self.gain_db, 1)
                self._ev("audio.level", lv)
                self.level_zero_sent = not sent
            if not sent:
                self.last_frames.clear()

    def _silence(self, kind: str, ca: ChunkAudio | None) -> Frame:
        return Frame(pcm=np.zeros(self.n, np.float32), utterance_id=ca.utterance_id if ca else None,
                     chunk_id=ca.chunk.chunk_id if ca else None, kind=kind, rate=self.rate)

    def _underrun(self, waiting: ChunkAudio | None) -> Frame | None:
        if self.state not in ("speaking", "ducked"):
            return None
        if not self.underrun:
            self.underrun = True
            self.gap_ms = 0
            self._ev("underrun", {"utterance_id": waiting.utterance_id if waiting else None,
                                  "waiting_chunk_id": waiting.chunk.chunk_id if waiting else None, "gap_ms": 0})
        self.gap_ms += self.frame_ms
        if self.cfg.get("fill_underrun_silence", True) and self.gap_ms <= self.cfg.get("max_silence_gap_ms", 400):
            return self._silence("silence", waiting)
        return None

    def _filler_for(self, ca: ChunkAudio) -> ChunkAudio | None:
        """Called with the lock held. A filler to play ahead of `ca` once it has stalled at the head of the queue too long:
        filler_after_ms for the first chunk of an utterance (the user is waiting for any sound), filler_gap_ms after that
        (a mid-speech gap). One try per chunk."""
        if self.filler_source is None or ca.filler_tried or not self.cfg.get("filler_enabled", False):
            return None
        now = time.monotonic()
        if ca.head_since is None:
            ca.head_since = now
        limit = self.cfg.get("filler_after_ms", 800) if ca.chunk.is_first else self.cfg.get("filler_gap_ms", 350)
        if (now - ca.head_since) * 1000 < limit or self.ducked or self.stopping:
            return None
        ca.filler_tried = True
        filler = self.filler_source(ca, int((now - ca.head_since) * 1000))
        if filler is not None:
            u = self.utts.get(ca.utterance_id)
            if u:
                u["chunks"] += 1
        return filler

    def _next_frame(self) -> Frame | None:
        """Called with the lock held. Returns the next frame to send, or None."""
        if self.stopping and self.stopping["left"] <= 0:
            self._finish_stop()
            return None
        while self.queue:
            ca = self.queue[0]
            if not ca.started:
                jitter = int(self.cfg.get("jitter_ms", 60) * self.rate / 1000)
                if not (ca.synth_done or ca.buffered >= max(jitter, 1)):
                    if not self.realtime:
                        return None
                    filler = self._filler_for(ca)
                    if filler is not None:
                        self.queue.insert(0, filler)
                        continue
                    return self._underrun(ca)
                ca.started = True
                ca.t_out_start = self.t_out_ms
                if self.underrun:
                    self.underrun = False
                self._ev("playback.chunk", {"utterance_id": ca.utterance_id, "chunk_id": ca.chunk.chunk_id,
                                            "event": "start", "t_out_ms": self.t_out_ms})
                first = True
            else:
                first = False
            if ca.buffered > 0 and (ca.buffered >= self.n or ca.synth_done):
                take = min(self.n, ca.buffered)
                parts, need = [], take
                while need > 0:
                    head = ca.buf[0]
                    if len(head) <= need:
                        parts.append(ca.buf.popleft())
                        need -= len(head)
                    else:
                        parts.append(head[:need])
                        ca.buf[0] = head[need:]
                        need = 0
                pcm = np.concatenate(parts) if len(parts) > 1 else parts[0]
                if take < self.n:
                    pcm = np.concatenate([pcm, np.zeros(self.n - take, np.float32)])
                ca.buffered -= take
                ca.consumed += take
                self.underrun = False
                self.gap_ms = 0
                self._progress(ca)
                return Frame(pcm=pcm.astype(np.float32, copy=False), utterance_id=ca.utterance_id,
                             chunk_id=ca.chunk.chunk_id, first_of_chunk=first,
                             last_of_chunk=ca.synth_done and ca.buffered == 0, trimmable=True, kind="speech",
                             rate=self.rate, fade_ms=int(self.cfg.get("crossfade_ms", 0)))
            if ca.synth_done and ca.buffered == 0:
                if ca.pause_frames < 0:
                    ca.pause_frames = int(round(ca.pause_after_ms / self.frame_ms))
                more = len(self.queue) > 1 or not self.utts.get(ca.utterance_id, {}).get("complete")
                if ca.pause_frames > 0 and more:
                    ca.pause_frames -= 1
                    return self._silence("pause", ca)
                self._finish_chunk(ca)
                continue
            # streaming engine slower than real time
            if not self.realtime:
                return None
            return self._underrun(ca)
        if not self.realtime:
            return None
        if self.stopping:
            self.stopping["left"] = 0
            self._finish_stop()
            return None
        return self._underrun(None) if any(not u["complete"] and not u.get("closed") for u in self.utts.values()) else None

    def _finish_chunk(self, ca: ChunkAudio) -> None:
        ca.finished = True
        ca.t_out_end = self.t_out_ms
        self.queue.popleft()
        self._ev("playback.chunk", {"utterance_id": ca.utterance_id, "chunk_id": ca.chunk.chunk_id,
                                    "event": "end", "t_out_ms": self.t_out_ms})
        if ca.word_index < len(ca.chunk.words) - 1 and ca.chunk.words:
            ca.word_index = len(ca.chunk.words) - 1
            self._pending_calls.append(lambda ca=ca: self.on_progress(ca, ca.word_index))
        u = self.utts.get(ca.utterance_id)
        if u:
            u["finished"] += 1
            if u["complete"] and u["finished"] >= u["chunks"]:
                self._utterance_finished(ca.utterance_id)

    def _utterance_finished(self, uid: str) -> None:
        u = self.utts.get(uid)
        if not u or u.get("closed"):
            return
        u["closed"] = True
        self._pending_calls.append(lambda: self.on_utterance_done(uid))
        if len(self.utts) > 100:
            self.utts = {k: v for k, v in self.utts.items() if not v.get("closed") or k == uid}
        if not self.queue:
            if self.state in ("speaking", "ducked"):
                self.set_state("done", "end", uid)
            self.set_state("idle", "done", uid)

    def _progress(self, ca: ChunkAudio) -> None:
        if not ca.timings or not ca.chunk.words:
            return
        pos = ca.consumed * 1000.0 / self.rate
        k = -1
        for i, t in enumerate(ca.timings):
            if (t.start_ms + t.end_ms) / 2 <= pos:
                k = i
        wi = map_word_index(k, len(ca.timings), len(ca.chunk.words))
        if wi != ca.word_index and wi >= 0:
            ca.word_index = wi
            self._pending_calls.append(lambda ca=ca, wi=wi: self.on_progress(ca, wi))

    def _run_pending(self) -> None:
        with self.lock:
            calls, self._pending_calls = self._pending_calls, []
        for c in calls:
            try:
                c()
            except Exception:
                log.exception("playback callback failed")

    def _send(self, frame: Frame, ca: ChunkAudio | None) -> None:
        x = frame.pcm
        target = -float(self.cfg["duck_db"]) if self.ducked else 0.0
        step = float(self.cfg["duck_db"]) * self.frame_ms / max(1.0, float(self.cfg["duck_ramp_ms"]))
        new = min(target, self.gain_db + step) if self.gain_db < target else max(target, self.gain_db - step)
        g = np.linspace(10 ** (self.gain_db / 20), 10 ** (new / 20), len(x), dtype=np.float32)
        self.gain_db = new
        with self.lock:
            st = self.stopping
            if st:
                total = max(1, st["frames"])
                a, b = st["left"] / total, (st["left"] - 1) / total
                g = g * np.linspace(a, max(b, 0.0), len(x), dtype=np.float32)
                st["left"] -= 1
        x = x * g
        uid = frame.utterance_id
        if uid:
            self.renders.setdefault(uid, []).append(x)
            u = self.utts.get(uid)
            if u is not None and not u["first_sent"] and frame.kind == "speech":
                u["first_sent"] = True
                with self.lock:
                    if self.state in ("buffering", "synthesizing", "idle", "done"):
                        self.set_state("ducked" if self.ducked else "speaking", "first_audio", uid)
                self.on_first_frame(uid)
        self.last_frames.append(x)
        data = to_pcm16(x)
        self.frames_sent += 1
        self.sink(b"\x01" + data)
        if self.ref_sink:
            try:
                self.ref_sink(b"\x02" + data)
            except Exception:
                pass

    def take_render(self, uid: str) -> np.ndarray:
        parts = self.renders.pop(uid, [])
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def map_word_index(k: int, n_tts: int, n_words: int) -> int:
    """Map engine word k (of n_tts) to chunk word index (of n_words)."""
    if k < 0:
        return -1
    if n_tts == n_words:
        return k
    return min(n_words - 1, max(0, math.ceil((k + 1) * n_words / max(1, n_tts)) - 1))


def estimate_timings(ca: ChunkAudio) -> list[WordTiming]:
    """Distribute the chunk's audio over its words by character count (trimmed of edge silence)."""
    words = ca.chunk.tts_text.split() or [w.text for w in ca.chunk.words]
    if not words or ca.total == 0:
        return []
    audio = np.concatenate(list(ca.buf)) if ca.buf and ca.consumed == 0 else None
    total_ms = ca.total * 1000.0 / ca_rate(ca)
    start_ms, end_ms = 0.0, total_ms
    if audio is not None and len(audio):
        thr = 10 ** (-45 / 20)
        idx = np.flatnonzero(np.abs(audio) > thr)
        if len(idx):
            r = ca_rate(ca)
            start_ms, end_ms = idx[0] * 1000.0 / r, (idx[-1] + 1) * 1000.0 / r
    weights = [max(1, len(w)) for w in words]
    tot = sum(weights)
    out, t = [], start_ms
    for w, wt in zip(words, weights):
        d = (end_ms - start_ms) * wt / tot
        out.append(WordTiming(w, round(t, 1), round(t + d, 1)))
        t += d
    return out


def ca_rate(ca: ChunkAudio) -> int:
    return ca.rate
