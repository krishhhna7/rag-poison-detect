"""Torch-free signal helpers, unit-tested on CPU.

Attention signals here are *NPAS-style* (attention mass on a passage, normalised),
inspired by Choudhary et al. [17]. They are our own implementation and should be
described as such in the report, not as a reproduction of the original score.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def char_spans_in_prompt(prompt: str, passages: Sequence[str]) -> List[Tuple[int, int]]:
    """Locate each passage (in order) inside the rendered prompt string.

    Searching sequentially from the previous end avoids mismatches when two
    passages share text (e.g. near-duplicate poison passages).
    """
    spans, cursor = [], 0
    for p in passages:
        s = prompt.find(p, cursor)
        if s < 0:
            raise ValueError("passage text not found in prompt; template altered it?")
        spans.append((s, s + len(p)))
        cursor = s + len(p)
    return spans


def token_spans_from_offsets(offsets: Sequence[Tuple[int, int]],
                             char_spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Map character spans to [start, end) token index spans using tokenizer offsets."""
    out = []
    for cs, ce in char_spans:
        idx = [i for i, (a, b) in enumerate(offsets) if b > a and a >= cs and b <= ce]
        if not idx:
            out.append((0, 0))
        else:
            out.append((idx[0], idx[-1] + 1))
    return out


def attention_shares(key_attention: np.ndarray, spans: Sequence[Tuple[int, int]]):
    """Passage-level attention features.

    ``key_attention``: 1-D array over key positions, already averaged over the
    answer-token queries, selected layers and heads.

    Returns (share, density):
      share   = passage attention mass / total attention mass over all passages
      density = mean attention per token in the passage, divided by the mean
                per-token attention over all passage tokens (length-normalised)
    """
    mass = np.array([key_attention[s:e].sum() for s, e in spans], dtype=np.float64)
    lens = np.array([max(e - s, 1) for s, e in spans], dtype=np.float64)
    total = mass.sum()
    if total <= 0:
        return np.zeros(len(spans)), np.zeros(len(spans))
    share = mass / total
    per_tok = mass / lens
    density = per_tok / (mass.sum() / lens.sum())
    return share, density


def loo_shift(logp_full: float, logp_without: Sequence[float]) -> np.ndarray:
    """Leave-one-out answer shift: how much removing passage i lowers the answer's log-prob.

    A passage that single-handedly drives the answer gives a large positive shift.
    """
    return logp_full - np.asarray(logp_without, dtype=np.float64)
