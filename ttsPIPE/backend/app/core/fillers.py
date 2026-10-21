"""Fillers: short natural interjections, rendered once by the live TTS in the live voice, played when synthesis stalls.

Rendered ahead of time (never while a real chunk is waiting) so a filler costs nothing at the moment it is needed.
"""
from __future__ import annotations

import random

import numpy as np

FILLERS = {
    "ta": ["ம்ம்...", "ஆங்... சரி...", "ஒரு நிமிஷம்...", "அது வந்து...", "சரி, பாக்கலாம்..."],
    "en": ["Hmm...", "Um, let me see...", "Okay, so...", "Well...", "Right, one moment..."],
}


class FillerBank:
    """(voice key, lang) → [(text, native pcm, rate)], plus the last one played so it never repeats back to back."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], list[tuple[str, np.ndarray, int]]] = {}
        self.rendering: set[str] = set()
        self.last: dict[tuple[str, str], str] = {}

    def has(self, vkey: str, lang: str, text: str) -> bool:
        return any(t == text for t, _, _ in self.items.get((vkey, lang), []))

    def add(self, vkey: str, lang: str, text: str, pcm: np.ndarray, rate: int) -> None:
        self.items.setdefault((vkey, lang), []).append((text, pcm, rate))

    def pick(self, vkey: str, lang: str) -> tuple[str, np.ndarray, int] | None:
        pool = self.items.get((vkey, lang)) or []
        if not pool:
            return None
        last = self.last.get((vkey, lang))
        choice = random.choice([p for p in pool if p[0] != last] or pool)
        self.last[(vkey, lang)] = choice[0]
        return choice
