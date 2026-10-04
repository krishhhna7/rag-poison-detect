"""Stage functions that glue retrieval, attacks, features, detectors and evaluation together."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import attacks
from .core import Passage, RetrievedSet, Target, answer_match
import hashlib

from .data import append_jsonl, read_jsonl, write_jsonl
from .evaluate import run_protocol, split_qids
from .features import extract_all, extract_set_features
from .detector import LearnedDetector, build_detectors, calibrate_threshold
from .features import feature_groups


def fingerprint(cfg, extra: str = "") -> str:
    """Hash of everything that changes what the attack stages produce. Partial results saved under a
    different fingerprint are ignored, so editing settings under the same run_name can never mix
    old and new results."""
    blob = json.dumps({"attack": vars(cfg.attack), "retriever": vars(cfg.retriever),
                       "generator": getattr(cfg.generator, "hf_id", ""), "seed": cfg.seed,
                       "detector": vars(cfg.detector), "extra": extra}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def load_partial(path: str, fp: str) -> dict:
    """{qid: row} for rows saved by an earlier (possibly interrupted) run with the same fingerprint."""
    if not os.path.exists(path):
        return {}
    done = {}
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:   # half-written last line after a disconnect
            continue
        if r.get("fp") == fp:
            done[r["qid"]] = r
    return done


def a0(cfg, t, llm):
    """A0 poison for target ``t`` honouring config options (released texts, question prefix)."""
    return attacks.attack_a0(t, llm, cfg.attack.n_poison,
                             getattr(cfg.attack, "use_released_poison", True),
                             getattr(cfg.attack, "prepend_question", True))


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
    partial, fp = out_path(cfg, "attack_partial.jsonl"), fingerprint(cfg)
    done = load_partial(partial, fp)
    if done:
        print(f"resuming: {len(done)} of {len(targets)} questions already done", flush=True)
    for t in _tqdm(targets):
        if t.qid in done:                               # finished before a disconnect: reuse
            r = done[t.qid]
            for name in ("clean", "A0", "A1"):
                out[name].append(RetrievedSet.from_dict(r[name]))
            diag["a1_fallback_slots"] += r["diag"]["a1_fallback_slots"]
            diag["a1_slots"] += r["diag"]["a1_slots"]
            continue
        clean = make_clean_set(t, retriever, k)
        seeds = a0(cfg, t, llm)
        a0_set = make_poisoned_set(t, retriever, k, seeds, "A0")
        a1, info = attacks.attack_a1(t, seeds, llm, retriever, ppl_fn, k, cfg.attack.n_candidates)
        a1_set = make_poisoned_set(t, retriever, k, a1, "A1")
        d = {"a1_fallback_slots": info["fallback_slots"], "a1_slots": len(seeds)}
        append_jsonl(partial, {"fp": fp, "qid": t.qid, "clean": clean.to_dict(), "A0": a0_set.to_dict(),
                               "A1": a1_set.to_dict(), "diag": d})
        out["clean"].append(clean); out["A0"].append(a0_set); out["A1"].append(a1_set)
        diag["a1_fallback_slots"] += d["a1_fallback_slots"]; diag["a1_slots"] += d["a1_slots"]
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
    partial, fp = out_path(cfg, "a2_partial.jsonl"), fingerprint(cfg, extra="a2")
    done = load_partial(partial, fp)
    if done:
        print(f"resuming: {len(done)} of {len(targets)} questions already done", flush=True)
    for t in _tqdm(targets):
        if t.qid in done:                               # finished before a disconnect: reuse
            out.append(RetrievedSet.from_dict(done[t.qid]["set"]))
            stats["success"] += int(done[t.qid]["success"]); stats["n"] += 1
            continue
        seeds = [p for p in by_q[t.qid].passages if p.is_poison] or a0(cfg, t, llm)

        def objective(cand: List[Passage]):
            rs = make_poisoned_set(t, retriever, k, cand, "A2")
            df = extract_set_features(rs, embedder, ppl_fn, generator)
            s = detector.score(df)
            pois = df["label"].to_numpy() == 1
            score = float(s[pois].max()) if pois.any() else 0.0  # no poison retrieved => undetectable
            return score, answer_match(df["answer"].iloc[0], t.target)

        cand, info = attacks.attack_a2(t, seeds, llm, retriever, ppl_fn, k, cfg.attack.n_candidates, objective)
        rs = make_poisoned_set(t, retriever, k, cand, "A2")
        append_jsonl(partial, {"fp": fp, "qid": t.qid, "set": rs.to_dict(), "success": bool(info["success"])})
        out.append(rs)
        stats["success"] += int(info["success"]); stats["n"] += 1
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


def attack_summary(feat_df: pd.DataFrame, targets: List[Target]) -> pd.DataFrame:
    """Per tier: number of sets, mean poison passages in the top-k, attack success rate (ASR)
    and answer accuracy. Needed to interpret detection results: a stealthier attack that rarely
    succeeds is a weak attack, not a strong one."""
    by_q = {t.qid: t for t in targets}
    sets = feat_df.groupby(["tier", "qid"]).agg(n_poison=("label", "sum"),
                                                answer=("answer", "first")).reset_index()
    sets["answer"] = sets["answer"].fillna("").astype(str)
    sets["asr"] = [answer_match(a, by_q[q].target) for a, q in zip(sets.answer, sets.qid)]
    sets["acc"] = [answer_match(a, by_q[q].correct) for a, q in zip(sets.answer, sets.qid)]
    return (sets.groupby("tier").agg(n_sets=("qid", "count"), poison_in_topk=("n_poison", "mean"),
                                     asr=("asr", "mean"), accuracy=("acc", "mean")).reset_index())
