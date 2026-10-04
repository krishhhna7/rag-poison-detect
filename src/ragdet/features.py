"""Per-passage feature extraction.

One row per (retrieved set, passage). Feature groups (used for ablations):

  ppl   : log-perplexity of the passage (+ set-relative versions)
  sem   : semantic consistency: query similarity, similarity to the rest of the set
  attn  : NPAS-style attention share / density on the passage
  loo   : leave-one-out answer shift and entropy change
  ent   : query-level answer entropy / log-prob (token uncertainty)

Set-relative versions (``*_z``) z-score a feature within its retrieved set, so a
detector can ask "is this passage unusual *compared with its neighbours*".
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .core import RetrievedSet

BASE_FEATURES = {
    "ppl": ["ppl_log"],
    "sem": ["q_sim", "cent_sim", "max_sim"],
    "attn": ["attn_share", "attn_density"],
    "loo": ["loo_shift", "loo_ent_delta"],
    "ent": ["ans_entropy", "ans_logprob"],
}
# per-passage features that get a within-set z-score
Z_FEATURES = ["ppl_log", "q_sim", "cent_sim", "max_sim", "attn_share", "loo_shift"]


def feature_groups() -> Dict[str, List[str]]:
    groups = {g: list(cols) for g, cols in BASE_FEATURES.items()}
    for g, cols in groups.items():
        cols += [f"{c}_z" for c in cols if c in Z_FEATURES]
    return groups


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def semantic_features(passage_emb: np.ndarray, q_emb: np.ndarray):
    """Cosine-based semantic consistency of each passage within its retrieved set."""
    P = passage_emb / np.clip(np.linalg.norm(passage_emb, axis=1, keepdims=True), 1e-8, None)
    q = q_emb / max(np.linalg.norm(q_emb), 1e-8)
    q_sim = P @ q
    n = len(P)
    if n < 2:
        return q_sim, np.zeros(n), np.zeros(n)
    S = P @ P.T
    np.fill_diagonal(S, np.nan)
    max_sim = np.nanmax(S, axis=1)
    cent = np.array([(P[np.arange(n) != i].mean(0) @ P[i]) /
                     max(np.linalg.norm(P[np.arange(n) != i].mean(0)), 1e-8) for i in range(n)])
    return q_sim, cent, max_sim


def extract_set_features(rs: RetrievedSet, embedder, ppl_fn: Callable, generator) -> pd.DataFrame:
    passages = rs.passages
    texts = [p.text for p in passages]
    ppl = np.log(np.asarray(ppl_fn(texts), dtype=np.float64))
    emb = embedder.encode(texts)
    q_emb = embedder.encode([rs.question], is_query=True)[0]
    q_sim, cent, max_sim = semantic_features(emb, q_emb)
    an = generator.analyze(rs.question, passages)

    df = pd.DataFrame({
        "qid": rs.qid, "tier": rs.tier, "pid": [p.pid for p in passages],
        "label": [int(p.is_poison) for p in passages],
        "rank": np.arange(len(passages)),
        "ppl_log": ppl, "q_sim": q_sim, "cent_sim": cent, "max_sim": max_sim,
        "attn_share": an.attn_share, "attn_density": an.attn_density,
        "loo_shift": an.loo_shift, "loo_ent_delta": an.loo_ent_delta,
        "ans_entropy": an.mean_entropy, "ans_logprob": an.mean_logprob,
    })
    for c in Z_FEATURES:
        df[f"{c}_z"] = _zscore(df[c].to_numpy())
    df.attrs["answer"] = an.answer
    df["answer"] = an.answer
    return df


def extract_all(sets: Sequence[RetrievedSet], embedder, ppl_fn, generator, progress=None) -> pd.DataFrame:
    frames = []
    it = sets if progress is None else progress(sets)
    for rs in it:
        frames.append(extract_set_features(rs, embedder, ppl_fn, generator))
    return pd.concat(frames, ignore_index=True)
