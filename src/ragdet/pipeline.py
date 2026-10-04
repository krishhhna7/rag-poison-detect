"""Stage functions that glue retrieval, attacks, features, detectors and evaluation together."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import attacks
from .core import Passage, RetrievedSet, Target, answer_match
from .data import read_jsonl, write_jsonl
from .evaluate import run_protocol, split_qids
from .features import extract_all, extract_set_features
from .detector import LearnedDetector, build_detectors, calibrate_threshold
from .features import feature_groups


def _tqdm(it):
    try:
        from tqdm import tqdm
        return tqdm(it)
    except ImportError:
        return it


# ---- paths ---------------------------------------------------------------
def out_path(cfg, name: str) -> str:
    d = os.path.join(cfg.paths.workspace, "runs", cfg.run_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def save_sets(cfg, name: str, sets: List[RetrievedSet]) -> None:
    write_jsonl(out_path(cfg, f"sets_{name}.jsonl"), [s.to_dict() for s in sets])


def load_sets(cfg, name: str) -> List[RetrievedSet]:
    return [RetrievedSet.from_dict(d) for d in read_jsonl(out_path(cfg, f"sets_{name}.jsonl"))]


# ---- stage 1: clean + A0 + A1 sets ---------------------------------------------
def make_clean_set(t: Target, retriever, k: int) -> RetrievedSet:
    return RetrievedSet(t.qid, t.question, t.correct, t.target, "clean", retriever.retrieve(t.question, k))


def make_poisoned_set(t: Target, retriever, k: int, poison: List[Passage], tier: str) -> RetrievedSet:
    return RetrievedSet(t.qid, t.question, t.correct, t.target, tier, retriever.retrieve(t.question, k, poison))


def build_attack_sets(cfg, targets: List[Target], retriever, llm, ppl_fn, tiers=("A0", "A1")):
    k = cfg.retriever.top_k
    out: Dict[str, List[RetrievedSet]] = {"clean": [], "A0": [], "A1": []}
    diag = {"a1_fallback_slots": 0, "a1_slots": 0}
    for t in _tqdm(targets):
        out["clean"].append(make_clean_set(t, retriever, k))
        seeds = attacks.attack_a0(t, llm, cfg.attack.n_poison)
        if "A0" in tiers:
            out["A0"].append(make_poisoned_set(t, retriever, k, seeds, "A0"))
        if "A1" in tiers:
            a1, info = attacks.attack_a1(t, seeds, llm, retriever, ppl_fn, k, cfg.attack.n_candidates)
            out["A1"].append(make_poisoned_set(t, retriever, k, a1, "A1"))
            diag["a1_fallback_slots"] += info["fallback_slots"]
            diag["a1_slots"] += len(seeds)
    return out, diag


# ---- stage 2: features -----------------------------------------------------------
def build_features(cfg, sets_by_tier: Dict[str, List[RetrievedSet]], embedder, ppl_fn, generator) -> pd.DataFrame:
    frames = [extract_all(sets, embedder, ppl_fn, generator, progress=_tqdm)
              for sets in sets_by_tier.values() if sets]
    return pd.concat(frames, ignore_index=True)


# ---- stage 3: A2 (detector-in-the-loop) ------------------------------------------------
def fit_attacker_detector(cfg, feat_df: pd.DataFrame, seed: int):
    """The detector an A2 attacker can query: fusion fitted on A0+clean fit-split questions."""
    fit_q, _, _ = split_qids(feat_df["qid"], seed, tuple(cfg.eval.split))
    df = feat_df[feat_df["tier"].isin(["A0", "clean"]) & feat_df["qid"].isin(fit_q)]
    return build_detectors(cfg.detector.model, seed)["fusion"].fit(df)


def build_a2_sets(cfg, targets: List[Target], a0_sets: List[RetrievedSet], retriever, llm, ppl_fn,
                  embedder, generator, detector) -> tuple[List[RetrievedSet], dict]:
    k = cfg.retriever.top_k
    by_q = {s.qid: s for s in a0_sets}
    out, stats = [], {"success": 0, "n": 0}
    for t in _tqdm(targets):
        seeds = [p for p in by_q[t.qid].passages if p.is_poison] or \
                attacks.attack_a0(t, llm, cfg.attack.n_poison)

        def objective(cand: List[Passage]):
            rs = make_poisoned_set(t, retriever, k, cand, "A2")
            df = extract_set_features(rs, embedder, ppl_fn, generator)
            s = detector.score(df)
            pois = df["label"].to_numpy() == 1
            score = float(s[pois].max()) if pois.any() else 0.0  # no poison retrieved => undetectable
            return score, answer_match(df["answer"].iloc[0], t.target)

        cand, info = attacks.attack_a2(t, seeds, llm, retriever, ppl_fn, k, cfg.attack.n_candidates, objective)
        out.append(make_poisoned_set(t, retriever, k, cand, "A2"))
        stats["success"] += int(info["success"])
        stats["n"] += 1
    return out, stats


# ---- stage 4: evaluation -----------------------------------------------------------------
def evaluate_all(cfg, feat_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for seed in cfg.eval.seeds:
        dets = build_detectors(cfg.detector.model, seed)
        frames.append(run_protocol(feat_df, dets, seed, cfg.detector.fpr_target,
                                   train_tiers=tuple(cfg.eval.train_tiers),
                                   test_tiers=tuple(cfg.eval.test_tiers),
                                   n_boot=cfg.eval.n_boot, frac=tuple(cfg.eval.split)))
    return pd.concat(frames, ignore_index=True)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds (with across-seed std) per detector x tier x metric."""
    g = res.groupby(["tier", "detector", "metric"])["value"]
    return g.agg(["mean", "std", "count"]).reset_index()


def attack_success_rate(sets: List[RetrievedSet], generator) -> float:
    return float(np.mean([answer_match(generator.answer(s.question, s.passages), s.target) for s in sets]))
