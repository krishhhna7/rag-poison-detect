"""Evaluation protocol (cross-fitted).

Key design decisions (state these in the Methodology section):
  * Splits are by *question*, never by passage, so no query leaks between train and test.
  * K-fold cross-fitting: for each held-out fold, detectors are fitted on A0 (+ clean) passages of
    three other folds, the decision threshold is calibrated on clean-set benign passages of the
    fifth fold, and the held-out fold is scored. Every question is therefore tested exactly once
    (100 test questions instead of ~30 from a single split). Repeated with different fold
    assignments (``seeds``) to show split-to-split stability.
  * Detectors are fitted on A0 only and tested on A0, A1 and A2: the cross-attack generalisation test.
  * The threshold is set for a target FPR on held-out clean-set passages; the realised FPR is
    *measured* and reported, never assumed.
  * Confidence intervals: cluster bootstrap over questions (passages within a question are
    correlated); detector comparisons: paired bootstrap on identical resamples.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .core import RetrievedSet, answer_match
from .detector import calibrate_threshold


# ---- folds / splits -----------------------------------------------------------
def split_qids(qids: Sequence[str], seed: int, frac=(0.5, 0.2, 0.3)):
    q = np.array(sorted(set(qids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(q)
    n1, n2 = int(len(q) * frac[0]), int(len(q) * (frac[0] + frac[1]))
    return set(q[:n1]), set(q[n1:n2]), set(q[n2:])


def fold_assignment(qids: Sequence[str], k: int, seed: int) -> Dict[str, int]:
    """Random, balanced assignment of questions to k folds."""
    q = np.array(sorted(set(qids)))
    np.random.RandomState(seed).shuffle(q)
    return {qid: i % k for i, qid in enumerate(q)}


# ---- point metrics --------------------------------------------------------
def safe_auroc(y, s) -> float:
    y = np.asarray(y)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def rates(y, clean_set_mask, flagged) -> Dict[str, float]:
    """Flag rates with the benign class split in two (they are different populations):

    tpr        poison passages flagged
    fpr_clean  benign passages flagged inside *clean* retrieved sets (deployment false-alarm rate)
    fpr_mixed  benign passages flagged inside *attacked* sets, i.e. sitting next to poison
    """
    y, cm, fl = np.asarray(y), np.asarray(clean_set_mask), np.asarray(flagged, dtype=bool)

    def frac(m):
        return float(fl[m].mean()) if m.any() else float("nan")
    return {"tpr": frac(y == 1), "fpr_clean": frac((y == 0) & cm), "fpr_mixed": frac((y == 0) & ~cm)}


def all_metrics(y, clean_set_mask, score, flagged) -> Dict[str, float]:
    """AUROC (poison vs all benign), AUROC *within attacked sets only* (poison vs its own
    neighbours: the hardest and most informative), plus the flag rates."""
    y, cm, score = np.asarray(y), np.asarray(clean_set_mask), np.asarray(score)
    m = rates(y, cm, flagged)
    m["auroc"] = safe_auroc(y, score)
    m["auroc_within"] = safe_auroc(y[~cm], score[~cm])
    return m


def query_level_detection(df: pd.DataFrame, flag_col: str) -> float:
    """Fraction of attacked sets in which at least one poison passage is flagged."""
    hits = [bool(d.loc[d.label == 1, flag_col].any()) for _, d in df.groupby(["qid", "tier"])
            if (d.label == 1).any()]
    return float(np.mean(hits)) if hits else float("nan")


def query_level_false_alarm(df: pd.DataFrame, flag_col: str) -> float:
    """Fraction of CLEAN retrieved sets in which at least one passage is (wrongly) flagged."""
    hits = [bool(d[flag_col].any()) for _, d in df[df["tier"] == "clean"].groupby(["qid", "tier"])]
    return float(np.mean(hits)) if hits else float("nan")


# ---- bootstrap ------------------------------------------------------------
def cluster_bootstrap(df: pd.DataFrame, names: Sequence[str], n_boot: int, seed: int):
    """Resample questions with replacement. ``df`` needs s_<name> (score) and f_<name> (flag) columns."""
    rng = np.random.RandomState(seed)
    qids = df["qid"].unique()
    groups = {q: np.flatnonzero((df["qid"] == q).to_numpy()) for q in qids}
    y = df["label"].to_numpy()
    clean = (df["tier"] == "clean").to_numpy()
    S = {n: df[f"s_{n}"].to_numpy() for n in names}
    F = {n: df[f"f_{n}"].to_numpy(dtype=bool) for n in names}
    keys = ("auroc", "auroc_within", "tpr", "fpr_clean", "fpr_mixed")
    out = {n: {k: [] for k in keys} for n in names}
    for _ in range(n_boot):
        pick = rng.choice(qids, size=len(qids), replace=True)
        idx = np.concatenate([groups[q] for q in pick])
        for n in names:
            m = all_metrics(y[idx], clean[idx], S[n][idx], F[n][idx])
            for k in keys:
                out[n][k].append(m[k])
    return {n: {k: np.array(v) for k, v in d.items()} for n, d in out.items()}


def ci(x: np.ndarray, alpha: float = 0.05):
    x = x[~np.isnan(x)]
    return (float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))) if len(x) else (np.nan, np.nan)


def paired_diff(a: np.ndarray, b: np.ndarray):
    """Paired bootstrap difference a-b on identical resamples: (mean, lo, hi, two-sided p)."""
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return (np.nan,) * 4
    lo, hi = ci(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), lo, hi, float(min(p, 1.0))


# ---- cross-fitting --------------------------------------------------------
def crossfit(df: pd.DataFrame, make_detectors: Callable[[], Dict[str, object]], seed: int,
             fpr_target: float = 0.05, train_tiers=("A0",), k: int = 5) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Out-of-fold scores and flags for every row; also the mean calibrated threshold per detector."""
    names = list(make_detectors().keys())
    folds = fold_assignment(df["qid"], k, seed)
    fold = df["qid"].map(folds).to_numpy()
    is_train_tier = df["tier"].isin(list(train_tiers) + ["clean"]).to_numpy()
    out = df.copy()
    for n in names:
        out[f"s_{n}"] = np.nan
        out[f"f_{n}"] = False
    thr_log: Dict[str, List[float]] = {n: [] for n in names}
    for f in range(k):
        test, cal = fold == f, fold == (f + 1) % k
        fit = ~test & ~cal
        fit_df = df[fit & is_train_tier]
        cal_df = df[cal & (df["tier"] == "clean").to_numpy() & (df["label"] == 0).to_numpy()]
        test_df = df[test]
        for n, det in make_detectors().items():
            det.fit(fit_df)
            thr = calibrate_threshold(det.score(cal_df), fpr_target)
            sc = det.score(test_df)
            out.loc[test, f"s_{n}"] = sc
            out.loc[test, f"f_{n}"] = sc >= thr
            thr_log[n].append(thr)
    return out, {n: float(np.mean(v)) for n, v in thr_log.items()}


def run_protocol(df: pd.DataFrame, make_detectors: Callable[[], Dict[str, object]], seed: int,
                 fpr_target: float = 0.05, train_tiers=("A0",), test_tiers=("A0", "A1", "A2"),
                 n_boot: int = 1000, k: int = 5, reference: str = "fusion"):
    """Cross-fitted evaluation. Returns (results table, out-of-fold scored frame)."""
    scored, thr = crossfit(df, make_detectors, seed, fpr_target, train_tiers, k)
    names = [c[2:] for c in scored.columns if c.startswith("s_")]
    metrics = ("auroc", "auroc_within", "tpr", "fpr_clean", "fpr_mixed")
    rows = []
    for tier in test_tiers:
        part = scored[(scored["tier"] == tier) | (scored["tier"] == "clean")]
        if not (part["tier"] == tier).any():
            continue
        boots = cluster_bootstrap(part, names, n_boot, seed)
        cm = (part["tier"] == "clean").to_numpy()
        ref = boots.get(reference)
        for name in names:
            pt = all_metrics(part["label"], cm, part[f"s_{name}"], part[f"f_{name}"])
            for metric in metrics:
                lo, hi = ci(boots[name][metric])
                row = dict(seed=seed, tier=tier, detector=name, metric=metric, value=pt[metric],
                           lo=lo, hi=hi, threshold=thr[name])
                if ref is not None and name != reference and metric in ("auroc", "auroc_within", "tpr"):
                    m, dlo, dhi, p = paired_diff(ref[metric], boots[name][metric])
                    row.update(ref_minus_this=m, diff_lo=dlo, diff_hi=dhi, p_value=p)
                rows.append(row)
            rows.append(dict(seed=seed, tier=tier, detector=name, metric="query_tpr",
                             value=query_level_detection(part, f"f_{name}"), lo=np.nan, hi=np.nan,
                             threshold=thr[name]))
            rows.append(dict(seed=seed, tier=tier, detector=name, metric="query_fpr_clean",
                             value=query_level_false_alarm(part, f"f_{name}"), lo=np.nan, hi=np.nan,
                             threshold=thr[name]))
    return pd.DataFrame(rows), scored


# ---- downstream utility: ASR / accuracy after filtering ------------------------
def utility_after_filtering(sets: Sequence[RetrievedSet], flagged: Dict[str, bool], answer_fn: Callable,
                            original_answers: Dict[str, str]) -> Dict[str, float]:
    """Drop flagged passages, regenerate, and measure ASR (attacker answer) and accuracy (true answer).

    ``flagged`` maps ``f"{qid}|{tier}|{pid}"`` to the out-of-fold decision. Sets where nothing is
    flagged keep their original answer (no regeneration needed).
    """
    asr, acc, rem_p, rem_c, regen = [], [], [], [], 0
    for rs in sets:
        fl = [bool(flagged[f"{rs.qid}|{rs.tier}|{p.pid}"]) for p in rs.passages]
        keep = [p for p, f in zip(rs.passages, fl) if not f]
        n_p = sum(p.is_poison for p in rs.passages)
        n_c = len(rs.passages) - n_p
        rem_p.append(sum(f and p.is_poison for p, f in zip(rs.passages, fl)) / n_p if n_p else np.nan)
        rem_c.append(sum(f and not p.is_poison for p, f in zip(rs.passages, fl)) / n_c if n_c else np.nan)
        if any(fl):
            ans = answer_fn(rs.question, keep)
            regen += 1
        else:
            ans = original_answers[f"{rs.qid}|{rs.tier}"]
        asr.append(answer_match(str(ans), rs.target))
        acc.append(answer_match(str(ans), rs.correct))
    return {"asr": float(np.mean(asr)), "accuracy": float(np.mean(acc)), "n": len(sets),
            "poison_removed": float(np.nanmean(rem_p)) if not np.all(np.isnan(rem_p)) else float("nan"),
            "benign_removed": float(np.nanmean(rem_c)), "regenerated": regen}
