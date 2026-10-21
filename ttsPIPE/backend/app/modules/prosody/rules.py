"""prosody/rules: punctuation pauses, question/exclaim style, emphasis, SSML-lite rate/pitch (§4.1)."""
from __future__ import annotations

from app.core.base import Chunk, ModuleInfo, ProsodyModule, ProsodyPlan, schema
from app.core.registry import register
from app.core.textutil import core_word, strip_punct
from app.modules.chunker.punct import end_char


@register
class RulesProsody(ProsodyModule):
    info = ModuleInfo(
        id="prosody/rules", kind="prosody", name="Rule-based prosody",
        description="Pause after sentence/clause ends, slight pitch rise for questions, emphasis for <emphasis> and ALL-CAPS words.",
        builtin=True, tags=["builtin"],
        params_schema=schema(
            sentence_pause_ms=("integer", 180, {"min": 0, "max": 2000, "unit": "ms"}),
            clause_pause_ms=("integer", 60, {"min": 0, "max": 2000, "unit": "ms"}),
            paragraph_pause_ms=("integer", 300, {"min": 0, "max": 3000, "unit": "ms"}),
            question_pitch_st=("number", 1.0, {"min": -6, "max": 6, "unit": "st"}),
            exclaim_speed=("number", 1.05, {"min": 0.5, "max": 2.0}),
            first_chunk_speed=("number", 1.0, {"min": 0.5, "max": 2.0, "description": "Speed of the first chunk"}),
        ),
    )

    def plan(self, chunk: Chunk, ctx: dict) -> ProsodyPlan:
        p = self.params
        last = end_char(chunk.words[-1].text) if chunk.words else ""
        plan = ProsodyPlan()
        if last in ".।…":
            plan.pause_after_ms = p["sentence_pause_ms"]
        elif last == "?":
            plan.pause_after_ms = p["sentence_pause_ms"]
            plan.pitch_st = p["question_pitch_st"]
            plan.style = "question"
        elif last == "!":
            plan.pause_after_ms = p["sentence_pause_ms"]
            plan.speed = p["exclaim_speed"]
            plan.style = "exclaim"
        elif last in ",;:—–":
            plan.pause_after_ms = p["clause_pause_ms"]
        if chunk.is_last:
            plan.pause_after_ms = 0
        if chunk.pause_after_ms:
            plan.pause_after_ms = max(plan.pause_after_ms, chunk.pause_after_ms)
        if chunk.is_first:
            plan.speed *= p["first_chunk_speed"]
        if chunk.rate:
            plan.speed *= chunk.rate
        if chunk.pitch_st:
            plan.pitch_st += chunk.pitch_st
        plan.emphasis = [i for i, w in enumerate(chunk.words)
                         if chunk.emphasis or (len(core_word(w.text)) > 2 and strip_punct(w.text).isupper()
                                               and strip_punct(w.text).isalpha())]
        return plan
