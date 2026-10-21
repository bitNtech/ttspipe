"""Session: text intake → chunks → look-ahead synthesis → paced playback (§5)."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Callable

import numpy as np

from . import textutil as T
from .audio import StreamResampler, resample_all
from .base import Chunk, ChunkPolicy, Frame, LangSegment, NormResult, ProsodyPlan, Voice, Word
from .cache import CacheEntry, cache_key
from .fillers import FILLERS
from .playback import ChunkAudio, PlaybackManager, map_word_index
from .runtime import StageSlot, StateHolder

if TYPE_CHECKING:
    from .manager import Manager

log = logging.getLogger("ttspipe.session")


def ms_since(t0: float | None, t1: float | None = None) -> float | None:
    if t0 is None:
        return None
    return round(((t1 if t1 is not None else time.perf_counter()) - t0) * 1000, 1)


class Utterance:
    def __init__(self, uid: str, profile: str | None):
        self.id = uid
        self.profile = profile
        self.created = time.time()
        self.t0: float | None = None
        self.t_end: float | None = None
        self.t_first_chunk: float | None = None
        self.t_first_audio: float | None = None
        self.first_synth_ms: float | None = None
        self.synth_ms = 0.0
        self.audio_ms = 0.0
        self.raw_text = ""
        self.words: list[Word] = []
        self.group_offset = 0
        self.edits: list[dict] = []
        self.spans: list[dict] = []
        self.normalized: list[str] = []
        self.chunks: list[Chunk] = []
        self.audio: dict[str, ChunkAudio] = {}
        self.lat: dict[str, dict] = {}
        self.cached: dict[str, bool] = {}
        self.ended = False
        self.status = "open"  # open | speaking | final | interrupted
        self.heard_text: str | None = None
        self.unplayed_text: str | None = None
        self.render_id: str | None = None
        self.shadow_audio: list[np.ndarray] = []
        self.ingest: StateHolder | None = None
        self.chunker: StateHolder | None = None
        self.summary: dict | None = None

    @property
    def done(self) -> bool:
        return self.status in ("final", "interrupted")


class Session:
    def __init__(self, manager: "Manager", sid: str, label: str = "", voice: Voice | None = None, *,
                 realtime: bool = True, slots: dict[str, StageSlot] | None = None,
                 shadow: StageSlot | None | bool = True, cfg: dict | None = None,
                 sink: Callable[[bytes], None] | None = None, profile: str | None = None,
                 record: bool = True):
        self.m = manager
        self.id = sid
        self.label = label
        self._cfg = cfg  # None = follow the live manager config
        self.rate = int(self.cfg["audio"]["out_rate"])
        self.frame_ms = int(self.cfg["audio"]["frame_ms"])
        self.slots = slots or manager.slots
        self.shadow_slot = manager.shadow if shadow is True else (shadow or None)
        v = self.cfg.get("voice") or {}
        self.voice = voice or Voice(impl=v.get("impl") or self.cfg["stages"]["tts"]["impl"], voice_id=v.get("voice_id", ""),
                                    speed=float(v.get("speed", 1.0)), pitch=float(v.get("pitch", 0.0)))
        self.profile = profile if profile is not None else self.cfg.get("active_profile")
        self.realtime = realtime
        self.record = record  # contribute to global metrics / renders
        self.started = time.time()
        self.utts: OrderedDict[str, Utterance] = OrderedDict()
        self.open: Utterance | None = None
        self.muted_uid: str | None = None
        self.seq = 0
        self.n_utt = 0
        self.cancel_gen = 0
        self.closed = False
        self.connected = True
        self.ext_sink = sink
        self.out_q: asyncio.Queue | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.tasks: list[asyncio.Task] = []
        self.tts_holder = StateHolder(self.slots["tts"])
        self.post_holder = StateHolder(self.slots["post"])
        self.shadow_holder = StateHolder(self.shadow_slot) if self.shadow_slot else None
        self.post_errors = 0
        self.playback: PlaybackManager | None = None
        self.filler_vkey: str | None = None

    @property
    def cfg(self) -> dict:
        return self._cfg if self._cfg is not None else self.m.config

    def _enabled(self, kind: str) -> bool:
        return bool(self.cfg["stages"][kind].get("enabled", True))

    # ---- lifecycle ----------------------------------------------------------------------------

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.intake_q: asyncio.Queue = asyncio.Queue()
        self.synth_q: asyncio.Queue = asyncio.Queue()
        self.shadow_q: asyncio.Queue = asyncio.Queue()
        if self.ext_sink is None:
            self.out_q = asyncio.Queue()
        self.playback = PlaybackManager(
            session_id=self.id, rate=self.rate, frame_ms=self.frame_ms, cfg=self.cfg["playback"],
            emit=self.emit, sink=self._sink, post=self._post, realtime=self.realtime,
            paused=lambda: self.m.paused, on_first_frame=self._on_first_frame, on_progress=self._on_progress,
            on_utterance_done=lambda uid: self._call_loop(self._finalize, uid),
            on_interrupted=lambda info: self._call_loop(self._on_interrupted, info),
            ref_sink=self.m.bridge.ref_sink(self.id) if self.record and self.m.bridge else None,
            filler_source=self._filler_source,
        )
        self.tasks = [asyncio.create_task(self._intake_loop(), name=f"intake-{self.id}"),
                      asyncio.create_task(self._synth_loop(), name=f"synth-{self.id}"),
                      asyncio.create_task(self._timer_loop(), name=f"timer-{self.id}"),
                      asyncio.create_task(self._warm_fillers(), name=f"fillers-warm-{self.id}")]
        if self.shadow_slot:
            self.tasks.append(asyncio.create_task(self._shadow_loop(), name=f"shadow-{self.id}"))
        self.playback.start()
        self.emit("session.started", {"label": self.label, "voice": self.voice.__dict__, "sample_rate": self.rate,
                                      "frame_ms": self.frame_ms, "format": "pcm16le", "profile": self.profile})

    async def close(self, reason: str = "closed") -> None:
        if self.closed:
            return
        self.closed = True
        self.cancel_gen += 1
        if self.playback:
            self.playback.close()
        for t in self.tasks:
            t.cancel()
        for holder in (self.tts_holder, self.post_holder):
            if holder.mod is not None and holder.state is not None:
                try:
                    await holder.slot.raw(lambda h=holder: h.mod.close_state(h.state))
                except Exception:
                    pass
        self.emit("session.ended", {"reason": reason, "utterances": len(self.utts),
                                    "duration_ms": int((time.time() - self.started) * 1000)})

    def _call_loop(self, fn, *args) -> None:
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(fn, *args)

    def emit(self, type_: str, data: dict) -> None:
        if not self.record and type_ not in ("utterance.end", "utterance.interrupted"):
            return
        self.m.bus.emit(type_, data, session_id=self.id,
                        t_out_ms=self.playback.t_out_ms if self.playback else 0)

    def _sink(self, data: bytes) -> None:
        if self.ext_sink is not None:
            self.ext_sink(data)
        elif self.out_q is not None:
            self._call_loop(self.out_q.put_nowait, data)

    # ---- client API (event-loop thread) -------------------------------------------------------

    def text(self, delta: str, uid: str | None = None) -> None:
        self.intake_q.put_nowait(("text", delta, uid))

    def end_utterance(self, uid: str | None = None) -> None:
        self.intake_q.put_nowait(("end", None, uid))

    def speak(self, text: str, uid: str | None = None) -> str:
        uid = uid or self._new_uid()
        self.intake_q.put_nowait(("text", text, uid))
        self.intake_q.put_nowait(("end", None, uid))
        return uid

    def duck(self, source: str = "client") -> None:
        self.playback.duck(source)

    def resume(self, source: str = "client") -> None:
        self.playback.resume(source)

    def stop(self, source: str = "client") -> None:
        self.cancel_gen += 1
        if self.open is not None:
            self.muted_uid = self.open.id
        while not self.synth_q.empty():
            self.synth_q.get_nowait()
        if not self.playback.stop(source):
            self._on_interrupted({"at_ms": self.playback.t_out_ms, "source": source, "chunks": {}, "utterance_ids": []})

    def reset(self) -> None:
        self.stop("reset")
        self.utts.clear()
        self.open = None
        self.muted_uid = None

    def set_voice(self, voice: str | None = None, speed: float | None = None, pitch: float | None = None) -> None:
        if voice:
            impl, _, vid = voice.partition(":")
            if vid:
                self.voice.impl, self.voice.voice_id = impl, vid
            else:
                self.voice.voice_id = impl
        if speed is not None:
            self.voice.speed = float(speed)
        if pitch is not None:
            self.voice.pitch = float(pitch)

    def _new_uid(self) -> str:
        self.n_utt += 1
        return f"u{self.n_utt}"

    # ---- intake -------------------------------------------------------------------------------

    async def _intake_loop(self) -> None:
        while True:
            kind, payload, uid = await self.intake_q.get()
            while self.m.paused:
                await asyncio.sleep(0.05)
            try:
                if kind == "text":
                    await self._on_text(payload, uid)
                elif kind == "end":
                    await self._on_end(uid)
                elif kind == "tick":
                    await self._on_tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("intake failed (session %s)", self.id)
                self.emit("error", {"where": "intake", "message": f"{type(e).__name__}: {e}", "kind": None,
                                    "impl": None, "fatal": False})

    async def _timer_loop(self) -> None:
        while True:
            await asyncio.sleep(0.02)
            u = self.open
            if u is not None and u.t0 is not None and not u.chunks and not u.ended and not u.done:
                self.intake_q.put_nowait(("tick", None, None))

    def _ctx(self, u: Utterance | None = None, chunk: Chunk | None = None) -> dict:
        return {"session_id": self.id if self.record else None, "utterance_id": u.id if u else None,
                "chunk_id": chunk.chunk_id if chunk else None,
                "t_out_ms": self.playback.t_out_ms if self.playback else 0}

    async def _new_utterance(self, uid: str | None) -> Utterance:
        uid = uid or self._new_uid()
        if uid in self.utts:  # client reused an id: make it unique
            uid = f"{uid}.{self._new_uid()}"
        u = Utterance(uid, self.profile)
        u.ingest = StateHolder(self.slots["ingest"])
        u.chunker = StateHolder(self.slots["chunker"])
        self.utts[uid] = u
        while len(self.utts) > 200:
            self.utts.popitem(last=False)
        self.open = u
        return u

    async def _on_text(self, delta: str, uid: str | None) -> None:
        if self.muted_uid is not None and (uid == self.muted_uid or (uid is None and self.open
                                                                     and self.open.id == self.muted_uid)):
            return  # interrupted reply keeps streaming: drop until end_utterance / new id
        if self.open is not None and uid and uid != self.open.id:
            await self._on_end(self.open.id)
        if self.open is None:
            if uid is not None and uid == self.muted_uid:
                return
            await self._new_utterance(uid)
        u = self.open
        if u.t0 is None:
            u.t0 = time.perf_counter()
            self.playback.note_text(u.id)
            self.emit("utterance.started", {"utterance_id": u.id})
        u.raw_text += delta
        self.emit("text.delta", {"utterance_id": u.id, "delta": delta, "chars_total": len(u.raw_text)})
        self.m.metrics.flow("chars", len(delta)) if self.record else None
        mod, st = await u.ingest.get()
        spans = await self.slots["ingest"].call(lambda m: m.feed(st, delta), self._ctx(u), mod=mod)
        for span in spans:
            await self._process_span(u, span)

    async def _on_end(self, uid: str | None) -> None:
        u = self.open
        if self.muted_uid is not None and (uid is None or uid == self.muted_uid):
            if u is not None and u.id == self.muted_uid:
                self.open = None
            self.muted_uid = None
            return
        if u is None or (uid and uid != u.id):
            return
        u.ended = True
        u.t_end = time.perf_counter()
        if u.t0 is None:
            self.open = None
            self.utts.pop(u.id, None)
            return
        mod, st = await u.ingest.get()
        for span in await self.slots["ingest"].call(lambda m: m.flush(st), self._ctx(u), mod=mod):
            await self._process_span(u, span)
        cmod, cst = await u.chunker.get()
        chunks = await self.slots["chunker"].call(lambda m: m.flush(cst, self._policy()), self._ctx(u), mod=cmod)
        if chunks:
            chunks[-1].is_last = True
        for c in chunks:
            await self._chunk_ready(u, c)
        if self.open is u:
            self.open = None
        self.playback.utterance_complete(u.id)

    async def _on_tick(self) -> None:
        u = self.open
        if u is None or u.t0 is None or u.chunks or u.ended:
            return
        cmod, cst = await u.chunker.get()
        now = (time.perf_counter() - u.t0) * 1000
        chunks = await self.slots["chunker"].call(lambda m: m.push(cst, [], self._policy(), now), self._ctx(u), mod=cmod)
        for c in chunks:
            await self._chunk_ready(u, c)

    def _policy(self) -> ChunkPolicy:
        p = ChunkPolicy(**{k: v for k, v in self.cfg["chunker"].items() if k in ChunkPolicy.__dataclass_fields__})
        mod, _ = self.slots["tts"].snapshot()
        if mod is not None and not mod.info.streaming:
            p.max_chunk_chars = min(p.max_chunk_chars, p.nonstreaming_max_chunk_chars)
        return p

    async def _process_span(self, u: Utterance, span) -> None:
        if span.kind == "break":
            segs = [LangSegment(text="", lang="en", attrs={"break_ms": span.break_ms})]
        else:
            if not span.text.strip():
                return
            t = time.perf_counter()
            ns = self.slots["normalize"]
            if self._enabled("normalize"):
                lang = span.lang_hint if span.lang_hint in ("ta", "en") else "auto"
                nr: NormResult = await ns.call(lambda m: m.normalize(span.text, language=lang), self._ctx(u))
            else:
                toks = span.text.split()
                nr = NormResult(text=" ".join(toks), words=[Word(text=w, raw=w, group=i) for i, w in enumerate(toks)])
            lat = ms_since(t)
            if self.record:
                self.m.metrics.add("normalize", lat)
            groups = {w.group for w in nr.words}
            remap = {g: u.group_offset + i for i, g in enumerate(sorted(groups))}
            for w in nr.words:
                w.group = remap[w.group]
            u.group_offset += len(groups)
            u.words += nr.words
            u.edits += nr.edits
            u.normalized.append(nr.text)
            self.emit("text.normalized", {"utterance_id": u.id, "before": span.text.strip(), "after": nr.text,
                                          "changed": nr.changed, "edits": nr.edits, "latency_ms": lat})
            t = time.perf_counter()
            ss = self.slots["segment"]
            if self._enabled("segment") and nr.text:
                segs = await ss.call(lambda m: m.split(nr.text), self._ctx(u))
            else:
                segs = [LangSegment(text=nr.text, lang="ta" if T.has_tamil(nr.text) else "en")]
            segs = self._attach_words(segs, nr.words)
            attrs = {k: v for k, v in (("emphasis", span.emphasis), ("rate", span.rate), ("pitch_st", span.pitch_st)) if v}
            for s in segs:
                s.attrs = dict(attrs)
            spans = [{"text": s.text, "lang": s.lang} for s in segs]
            u.spans += spans
            self.emit("segment", {"utterance_id": u.id, "spans": spans, "latency_ms": ms_since(t)})
        cmod, cst = await u.chunker.get()
        if getattr(cst, "utterance_id", None) != u.id and hasattr(cst, "utterance_id"):
            cst.utterance_id = u.id
        now = (time.perf_counter() - u.t0) * 1000
        chunks = await self.slots["chunker"].call(lambda m: m.push(cst, segs, self._policy(), now), self._ctx(u), mod=cmod)
        for c in chunks:
            await self._chunk_ready(u, c)

    def _attach_words(self, segs: list[LangSegment], words: list[Word]) -> list[LangSegment]:
        """Segmenters return text+lang; the runtime attaches the word map (§4.1: segment must not change text)."""
        i = 0
        ok = True
        for s in segs:
            n = len(s.text.split())
            s.words = words[i:i + n]
            if [w.text for w in s.words] != s.text.split():
                ok = False
            for w in s.words:
                w.lang = s.lang
            i += n
        if not ok or i != len(words):
            log.warning("segmenter changed text; falling back to one segment")
            lang = segs[0].lang if segs else "en"
            for w in words:
                w.lang = lang
            return [LangSegment(text=" ".join(w.text for w in words), lang=lang, words=list(words))]
        return [s for s in segs if s.words]

    async def _chunk_ready(self, u: Utterance, c: Chunk) -> None:
        if u.done or self.muted_uid == u.id:
            return
        c.utterance_id = u.id
        gs, ps = self.slots["g2p"], self.slots["prosody"]
        if self._enabled("g2p") and c.words:
            c = await gs.call(lambda m: m.apply(c), self._ctx(u, c))
        if self._enabled("prosody") and c.words:
            ctx = {"utterance_id": u.id, "index": c.index, "voice": self.voice, "is_first": c.is_first}
            plan = await ps.call(lambda m: m.plan(c, ctx), self._ctx(u, c))
        else:
            plan = ProsodyPlan(pause_after_ms=c.pause_after_ms)
        now = time.perf_counter()
        if u.t_first_chunk is None and c.words:
            u.t_first_chunk = now
        u.chunks.append(c)
        ca = ChunkAudio(seq=self.seq, chunk=c, pause_after_ms=plan.pause_after_ms, rate=self.rate)
        self.seq += 1
        u.audio[c.chunk_id] = ca
        u.lat[c.chunk_id] = {"t_flush": now}
        self.emit("chunk.flushed", {
            "utterance_id": u.id, "chunk_id": c.chunk_id, "index": c.index, "text": c.text, "norm_text": c.norm_text,
            "tts_text": c.tts_text, "lang": c.lang, "words": len(c.words), "reason": c.reason,
            "segments": [{"text": s.text, "lang": s.lang} for s in c.segments], "pause_after_ms": plan.pause_after_ms,
            "since_text_ms": ms_since(u.t0, now)})
        if self.record:
            self.m.metrics.flow("chunks", 1)
        self.playback.add_chunk(ca)
        if c.words:
            self.synth_q.put_nowait((u, c, plan, ca, self.cancel_gen))
            if self.shadow_slot:
                self.shadow_q.put_nowait((u, c, plan, self.cancel_gen))
        else:
            self.playback.finish_synth(ca)

    # ---- synthesis ----------------------------------------------------------------------------

    def _voice_for(self, mod) -> Voice:
        ids = [v.id for v in mod.list_voices()]
        vid = self.voice.voice_id
        if ids and vid not in ids:
            cfg_vid = (self.cfg.get("voice") or {}).get("voice_id")
            vid = cfg_vid if cfg_vid in ids else ids[0]
        return Voice(impl=mod.info.impl, voice_id=vid, speed=self.voice.speed, pitch=self.voice.pitch)

    async def _synth_loop(self) -> None:
        while True:
            u, c, plan, ca, gen = await self.synth_q.get()
            if gen != self.cancel_gen or u.done:
                continue
            look = int(self.cfg["playback"].get("lookahead_chunks", 2))
            while gen == self.cancel_gen:
                head = self.playback.head_seq()
                if head is None or ca.seq - head <= look:
                    break
                await asyncio.sleep(0.005)
            while self.m.paused:
                await asyncio.sleep(0.05)
            if gen != self.cancel_gen:
                continue
            try:
                await self._synthesize(u, c, plan, ca, gen)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.playback.finish_synth(ca, failed=True)
                self.emit("error", {"where": "synthesis", "message": f"{type(e).__name__}: {e}", "kind": "tts",
                                    "impl": self.slots["tts"].impl, "fatal": False})

    # ---- fillers ------------------------------------------------------------------------------

    async def _warm_fillers(self) -> None:
        """Start rendering at session start so even the first utterance can have fillers."""
        if not self.cfg["playback"].get("filler_enabled"):
            return
        try:
            mod, state = await self.tts_holder.get()
            voice = self._voice_for(mod)
            cache_voice = getattr(mod, "cache_voice", None)
            cvoice = await asyncio.to_thread(cache_voice, voice.voice_id) if cache_voice else voice.voice_id
            self.filler_vkey = cache_key(mod.info.id, mod.params, cvoice, 1.0, 0.0, "filler")
            self._ensure_fillers(mod, state, voice)
        except Exception as e:
            log.warning("filler warm-up skipped (%s: %s)", type(e).__name__, e)

    def _ensure_fillers(self, mod, state, voice: Voice) -> None:
        vkey = self.filler_vkey
        if self.cfg["playback"].get("filler_enabled") and vkey and vkey not in self.m.fillers.rendering:
            self.m.fillers.rendering.add(vkey)
            self.tasks.append(asyncio.create_task(self._render_fillers(mod, state, voice, vkey), name=f"fillers-{self.id}"))

    async def _render_fillers(self, mod, state, voice: Voice, vkey: str) -> None:
        """Render each missing filler in the live voice, one per slot job and only while playback is idle, so a real
        chunk never queues behind more than one short filler."""
        bank = self.m.fillers

        def render(text: str, lang: str):
            c = Chunk(f"filler.{lang}", "filler", 0, [Word(text, text)], [], tts_text=text, lang=lang)
            pcm, rate = [], 0
            for p in mod.synthesize(state, c, ProsodyPlan(), voice):
                if len(p.pcm):
                    pcm.append(np.asarray(p.pcm, np.float32))
                    rate = p.sample_rate
            return (np.concatenate(pcm) if pcm else np.zeros(0, np.float32)), rate

        try:
            for lang, texts in FILLERS.items():
                for text in texts:
                    if bank.has(vkey, lang, text):
                        continue
                    while self.playback.state != "idle" and not self.closed:
                        await asyncio.sleep(0.25)
                    if self.closed:
                        return
                    pcm, rate = await self.slots["tts"].raw(lambda: render(text, lang))
                    if len(pcm):
                        bank.add(vkey, lang, text, pcm, rate)
        except Exception as e:
            log.warning("filler render failed (%s: %s); continuing without the rest", type(e).__name__, e)
        finally:
            if not all(bank.has(vkey, lg, t) for lg, ts in FILLERS.items() for t in ts):
                bank.rendering.discard(vkey)  # incomplete (session closed / error): the next session picks it up

    def _filler_source(self, ca: ChunkAudio, stalled_ms: int) -> ChunkAudio | None:
        """Runs on the playback thread, so no I/O: the bank is in memory."""
        lang = "ta" if T.has_tamil(ca.chunk.norm_text) else "en"
        pick = self.m.fillers.pick(self.filler_vkey, lang) if self.filler_vkey else None
        if pick is None:
            return None
        text, pcm, rate = pick
        out = resample_all(pcm, rate, self.rate)
        chunk = Chunk(f"{ca.chunk.chunk_id}.f", ca.chunk.utterance_id, ca.chunk.index, [], [], reason="filler")
        self.emit("filler", {"utterance_id": ca.chunk.utterance_id, "chunk_id": ca.chunk.chunk_id, "text": text,
                             "lang": lang, "stalled_ms": stalled_ms, "audio_ms": round(len(out) * 1000 / self.rate)})
        return ChunkAudio(seq=ca.seq, chunk=chunk, rate=self.rate, buf=deque([out]), buffered=len(out), total=len(out),
                          synth_done=True, filler_tried=True)

    async def _synthesize(self, u: Utterance, c: Chunk, plan: ProsodyPlan, ca: ChunkAudio, gen: int) -> None:
        slot = self.slots["tts"]
        mod, state = await self.tts_holder.get()
        voice = self._voice_for(mod)
        cache_voice = getattr(mod, "cache_voice", None)
        cvoice = cache_voice(voice.voice_id) if cache_voice else voice.voice_id
        key = cache_key(mod.info.id, mod.params, cvoice, voice.speed * plan.speed, voice.pitch + plan.pitch_st, c.tts_text)
        self.filler_vkey = cache_key(mod.info.id, mod.params, cvoice, 1.0, 0.0, "filler")
        self._ensure_fillers(mod, state, voice)
        lat = u.lat[c.chunk_id]
        t_start = time.perf_counter()
        lat["t_synth_start"] = t_start
        self.playback.synth_started(u.id)
        entry = self.m.cache.get(key) if self.record else None
        self.emit("synth.start", {"utterance_id": u.id, "chunk_id": c.chunk_id, "impl": mod.info.impl,
                                  "queue_ms": ms_since(lat["t_flush"], t_start), "cached": entry is not None})
        u.cached[c.chunk_id] = entry is not None
        if entry is not None:
            pcm = resample_all(entry.pcm, entry.rate, self.rate)
            self.playback.add_audio(ca, pcm, entry.timings)
            self.playback.finish_synth(ca)
            lat["first_audio_ms"] = 0.0
            lat["synth_ms"] = ms_since(t_start)
            lat["audio_ms"] = round(len(entry.pcm) * 1000 / entry.rate, 1)
            if u.first_synth_ms is None:
                u.first_synth_ms = 0.0
            u.audio_ms += lat["audio_ms"]
            self.emit("synth.first_audio", {"utterance_id": u.id, "chunk_id": c.chunk_id, "latency_ms": 0.0, "cached": True})
            self.emit("synth.done", {"utterance_id": u.id, "chunk_id": c.chunk_id, "audio_ms": lat["audio_ms"],
                                     "synth_ms": lat["synth_ms"], "rtf": 0.0, "cached": True,
                                     "word_timings": bool(entry.timings)})
            return
        native: list[np.ndarray] = []
        rates_list: list[int] = []
        rates: set[int] = set()
        timings_all = []
        rs: dict = {"r": None, "rate": None}
        first: dict = {"t": None}

        def cancelled() -> bool:
            return gen != self.cancel_gen

        def on_piece(piece) -> None:
            if first["t"] is None and len(piece.pcm):
                first["t"] = time.perf_counter()
                fa = ms_since(t_start, first["t"])
                lat["first_audio_ms"] = fa
                if u.first_synth_ms is None:
                    u.first_synth_ms = fa
                self.emit("synth.first_audio", {"utterance_id": u.id, "chunk_id": c.chunk_id, "latency_ms": fa,
                                                "cached": False})
            if rs["rate"] != piece.sample_rate:
                if rs["r"] is not None:
                    self.playback.add_audio(ca, rs["r"].flush(), None)
                rs["r"], rs["rate"] = StreamResampler(piece.sample_rate, self.rate), piece.sample_rate
            out = rs["r"].process(piece.pcm)
            tim = piece.word_timings
            self.playback.add_audio(ca, out, tim)
            native.append(np.asarray(piece.pcm, np.float32))
            rates_list.append(piece.sample_rate)
            rates.add(piece.sample_rate)
            if tim:
                timings_all.extend(tim)

        def job(m):
            for piece in m.synthesize(state, c, plan, voice):
                if cancelled():
                    return False
                on_piece(piece)
            if rs["r"] is not None:
                self.playback.add_audio(ca, rs["r"].flush(), None)
            return not cancelled()

        try:
            completed = await slot.call(job, self._ctx(u, c), mod=mod)
        except Exception:
            self.playback.finish_synth(ca, failed=True)
            raise
        self.playback.finish_synth(ca)
        synth_ms = ms_since(t_start)
        audio_ms = round(sum(len(x) * 1000 / r for x, r in zip(native, rates_list)), 1)
        rtf = round(synth_ms / audio_ms, 3) if audio_ms else None
        lat.update({"synth_ms": synth_ms, "audio_ms": audio_ms, "rtf": rtf})
        u.synth_ms += synth_ms
        u.audio_ms += audio_ms
        if self.record:
            if c.is_first and lat.get("first_audio_ms") is not None:
                self.m.metrics.add("first_synth", lat["first_audio_ms"])
            self.m.metrics.add("chunk_rtf", rtf)
            self.m.metrics.flow("audio_s", audio_ms / 1000)
        self.emit("synth.done", {"utterance_id": u.id, "chunk_id": c.chunk_id, "audio_ms": audio_ms, "synth_ms": synth_ms,
                                 "rtf": rtf, "cached": False, "word_timings": bool(timings_all)})
        if completed and len(rates) == 1 and native and self.record:
            self.m.cache.put(key, CacheEntry(np.concatenate(native), rates_list[0], timings_all or None))

    async def _shadow_loop(self) -> None:
        while True:
            u, c, plan, gen = await self.shadow_q.get()
            if gen != self.cancel_gen or u.done or not self.shadow_slot:
                continue
            mod, state = await self.shadow_holder.get()
            voice_id = (((self.cfg.get("graph") or {}).get("shadows") or {}).get("tts") or {}).get("voice_id") or ""
            ids = [v.id for v in mod.list_voices()]
            voice = Voice(impl=mod.info.impl, voice_id=voice_id if voice_id in ids else (ids[0] if ids else ""),
                          speed=self.voice.speed, pitch=self.voice.pitch)
            t0 = time.perf_counter()
            first = {"t": None}
            out: list[np.ndarray] = []
            audio = {"ms": 0.0}

            def job(m):
                for piece in m.synthesize(state, c, plan, voice):
                    if gen != self.cancel_gen:
                        return
                    if first["t"] is None and len(piece.pcm):
                        first["t"] = time.perf_counter()
                    audio["ms"] += len(piece.pcm) * 1000 / piece.sample_rate
                    out.append(resample_all(np.asarray(piece.pcm, np.float32), piece.sample_rate, self.rate))

            err = None
            try:
                await self.shadow_slot.call(job, self._ctx(u, c), mod=mod)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            synth_ms = ms_since(t0)
            if out:
                u.shadow_audio += out
                u.shadow_audio.append(np.zeros(int(self.rate * plan.pause_after_ms / 1000), np.float32))
            self.emit("shadow", {"utterance_id": u.id, "chunk_id": c.chunk_id, "impl": mod.info.impl,
                                 "first_audio_ms": ms_since(t0, first["t"]) if first["t"] else None,
                                 "synth_ms": synth_ms, "audio_ms": round(audio["ms"], 1),
                                 "rtf": round(synth_ms / audio["ms"], 3) if audio["ms"] else None, "error": err,
                                 "render_id": None})

    # ---- post (pacer thread) ------------------------------------------------------------------

    def _post(self, frame: Frame) -> Frame | None:
        slot = self.slots["post"]
        if not self._enabled("post"):
            return frame
        mod, state = self.post_holder.get_sync()
        t0 = time.perf_counter()
        try:
            out = mod.process(state, frame)
            self.post_errors = 0
            if self.record:
                self.m.metrics.stage_call("post", mod.info.impl, (time.perf_counter() - t0) * 1000, True)
            return out
        except Exception as e:
            self.post_errors += 1
            if self.post_errors in (1, 3):
                self.emit("error", {"where": "stage.post", "message": f"{type(e).__name__}: {e}", "kind": "post",
                                    "impl": mod.info.impl, "fatal": False})
            if self.post_errors >= 3:
                slot.unload_to_builtin(f"{mod.info.impl}: 3 consecutive errors; last: {e}")
                slot.status, slot.fallback = "error", True
                slot._emit_status()
            return frame

    # ---- playback callbacks -------------------------------------------------------------------

    def _on_first_frame(self, uid: str) -> None:
        u = self.utts.get(uid)
        if u is not None and u.t_first_audio is None:
            u.t_first_audio = time.perf_counter()
            u.status = "speaking"

    def _on_progress(self, ca: ChunkAudio, wi: int) -> None:
        u = self.utts.get(ca.utterance_id)
        if u is None:
            return
        heard, _ = self._heard(u, {ca.chunk.chunk_id: {"word_index": wi}})
        self.emit("playback.progress", {"utterance_id": u.id, "chunk_id": ca.chunk.chunk_id, "word_index": wi,
                                        "heard_text": heard})

    def _heard(self, u: Utterance, info: dict) -> tuple[str, str]:
        """heard/unplayed text (§5.7). info: chunk_id → {played_ms, timings} or {word_index}."""
        heard_word = [False] * len(u.words)
        pos = {id(w): i for i, w in enumerate(u.words)}
        for c in u.chunks:
            ca = u.audio.get(c.chunk_id)
            if ca is None or not c.words:
                continue
            ci = info.get(c.chunk_id)
            if ca.finished and (ci is None or ci.get("finished", True)):
                n = len(c.words)
            elif ci is not None and "word_index" in ci:
                n = ci["word_index"] + 1
            elif ci is not None and ci.get("started"):
                tims = ci.get("timings") or []
                k = -1
                for i, t in enumerate(tims):
                    if (t.start_ms + t.end_ms) / 2 <= ci["played_ms"]:
                        k = i
                n = map_word_index(k, len(tims), len(c.words)) + 1
            else:
                n = 0
            for w in c.words[:n]:
                if id(w) in pos:
                    heard_word[pos[id(w)]] = True
        last_of_group: dict[int, int] = {}
        for i, w in enumerate(u.words):
            last_of_group[w.group] = i
        heard_groups = {g for g, i in last_of_group.items() if heard_word[i]}
        heard = " ".join(w.raw for w in u.words if w.raw and w.group in heard_groups)
        unplayed = " ".join(w.raw for w in u.words if w.raw and w.group not in heard_groups)
        held = getattr(u.ingest.state, "buf", "") if u.ingest else ""
        if held and held.strip():
            unplayed = (unplayed + " " + held.strip()).strip()
        return heard, unplayed

    def _on_interrupted(self, info: dict) -> None:
        for uid, u in list(self.utts.items()):
            if u.done or u.t0 is None:
                continue
            heard, unplayed = self._heard(u, info["chunks"])
            if not any(cid in info["chunks"] for cid in u.audio):
                heard, unplayed = "", " ".join(w.raw for w in u.words if w.raw)
                held = getattr(u.ingest.state, "buf", "") if u.ingest else ""
                unplayed = (unplayed + " " + held.strip()).strip() if held and held.strip() else unplayed
            u.heard_text, u.unplayed_text = heard, unplayed
            self.emit("utterance.interrupted", {"utterance_id": uid, "heard_text": heard, "unplayed_text": unplayed,
                                                "at_ms": info["at_ms"], "source": info["source"]})
            if self.record:
                self.m.metrics.inc("interruptions")
            self._finalize(uid, interrupted=True)
        # the interrupted reply stays "open" (muted) until end_utterance, so its trailing deltas are dropped
        if self.open is not None and self.open.done and self.open.id != self.muted_uid:
            self.open = None

    def _finalize(self, uid: str, interrupted: bool = False) -> None:
        u = self.utts.get(uid)
        if u is None or u.done:
            return
        u.status = "interrupted" if interrupted else "final"
        ttfa = ms_since(u.t0, u.t_first_audio) if u.t_first_audio else None
        ttfa_end = ms_since(u.t_end, u.t_first_audio) if (u.t_first_audio and u.t_end) else None
        latency = {
            "ttfa_ms": ttfa, "ttfa_end_ms": ttfa_end,
            "text_to_first_chunk_ms": ms_since(u.t0, u.t_first_chunk) if u.t_first_chunk else None,
            "first_synth_ms": u.first_synth_ms, "total_synth_ms": round(u.synth_ms, 1),
            "audio_ms": round(u.audio_ms, 1), "rtf": round(u.synth_ms / u.audio_ms, 3) if u.audio_ms else None,
        }
        pcm = self.playback.take_render(uid)
        render_id = None
        if self.record:
            self.m.metrics.ttfa(u.profile or "-", ttfa, ttfa_end)
            self.m.metrics.add("text_to_first_chunk", latency["text_to_first_chunk_ms"])
            self.m.metrics.inc("utterances")
            if len(pcm):
                render_id = self.m.renders.save(self.id, uid, pcm, self.rate, {
                    "text": u.raw_text, "interrupted": interrupted, "heard_text": u.heard_text})
            if u.shadow_audio:
                self.m.renders.save(self.id, uid + "__shadow", np.concatenate(u.shadow_audio), self.rate,
                                    {"text": u.raw_text, "shadow": True})
        u.render_id = render_id
        u.render_pcm = pcm  # kept for offline renders (compare / smoke)
        chunks = []
        for c in u.chunks:
            lat = u.lat.get(c.chunk_id, {})
            ca = u.audio.get(c.chunk_id)
            played = "full" if ca and ca.finished else ("partial" if ca and ca.started else "none")
            chunks.append({"chunk_id": c.chunk_id, "index": c.index, "text": c.text, "norm_text": c.norm_text,
                           "lang": c.lang, "reason": c.reason, "words": len(c.words), "played": played,
                           "cached": u.cached.get(c.chunk_id, False),
                           "latency": {"flush_ms": ms_since(u.t0, lat.get("t_flush")),
                                       "synth_queue_ms": ms_since(lat.get("t_flush"), lat.get("t_synth_start")) if lat.get("t_synth_start") else None,
                                       "first_audio_ms": lat.get("first_audio_ms"), "synth_ms": lat.get("synth_ms"),
                                       "audio_ms": lat.get("audio_ms"), "rtf": lat.get("rtf")}})
        u.summary = {"utterance_id": uid, "text": u.raw_text, "normalized": " ".join(u.normalized),
                     "chunks": chunks, "interrupted": interrupted, "profile": u.profile, "latency": latency,
                     "render_id": render_id, "edits": u.edits, "segments": u.spans, "heard_text": u.heard_text,
                     "unplayed_text": u.unplayed_text, "status": u.status, "created": u.created}
        self.emit("utterance.end", {k: u.summary[k] for k in ("utterance_id", "text", "normalized", "chunks",
                                                               "interrupted", "profile", "latency", "render_id")})
        if self.open is not None and self.open.t0 is not None and not self.open.done:
            self.playback.note_text(self.open.id)

    # ---- views --------------------------------------------------------------------------------

    def summary(self) -> dict:
        last = next(reversed(self.utts), None) if self.utts else None
        return {"id": self.id, "label": self.label, "started": int(self.started * 1000),
                "state": self.playback.state if self.playback else "idle", "voice": self.voice.__dict__,
                "utterances": len(self.utts), "last_utterance_id": last,
                "t_out_ms": self.playback.t_out_ms if self.playback else 0, "connected": self.connected}

    def utterance_rows(self, limit: int = 50) -> list[dict]:
        rows = []
        for u in list(self.utts.values())[-limit:]:
            if u.summary:
                rows.append(u.summary)
            else:
                rows.append({"utterance_id": u.id, "text": u.raw_text, "normalized": " ".join(u.normalized),
                             "chunks": [{"chunk_id": c.chunk_id, "index": c.index, "text": c.text, "lang": c.lang,
                                         "reason": c.reason, "words": len(c.words)} for c in u.chunks],
                             "interrupted": False, "profile": u.profile, "latency": {}, "render_id": None,
                             "edits": u.edits, "segments": u.spans, "status": u.status, "created": u.created})
        return rows

    async def wait_idle(self, timeout: float = 60.0) -> bool:
        """Offline helper: wait until all text is processed and played."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            busy = (not self.intake_q.empty() or self.open is not None or not self.synth_q.empty()
                    or any(not u.done and u.t0 is not None for u in self.utts.values()))
            if not busy:
                return True
            await asyncio.sleep(0.005)
        return False


def new_session_id() -> str:
    return "s" + uuid.uuid4().hex[:6]
