"""Evaluation protocol.

Key design decisions (state these in the Methodology section):
  * Splits are by *question*, never by passage, so no query leaks between train/test.
  * Detectors are fitted on A0 (+ clean) only and tested on A0, A1, A2: the cross-attack
    generalisation test.
  * The decision threshold is calibrated on held-out benign training-split passages to hit a
    target FPR; the test FPR is then *measured* (and reported), not assumed.
  * Confidence intervals use a cluster bootstrap over questions (rows within a question are
    correlated); detector comparisons use paired bootstrap on identical resamples.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .core import RetrievedSet, answer_match
from .detector import calibrate_threshold


# ---- splits ---------------------------------------------------------------
def split_qids(qids: Sequence[str], seed: int, frac=(0.5, 0.2, 0.3)):
    q = np.array(sorted(set(qids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(q)
    n1, n2 = int(len(q) * frac[0]), int(len(q) * (frac[0] + frac[1]))
    return set(q[:n1]), set(q[n1:n2]), set(q[n2:])


# ---- point metrics --------------------------------------------------------
def safe_auroc(y, s) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def flag_metrics(y, score, thr) -> Dict[str, float]:
    y, flagged = np.asarray(y), np.asarray(score) >= thr
    pos, neg = y == 1, y == 0
    return {
        "tpr": float(flagged[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(flagged[neg].mean()) if neg.any() else float("nan"),
    }


def query_level_detection(df: pd.DataFrame, score_col: str, thr: float) -> float:
    """Fraction of poisoned sets in which at least one poison passage is flagged."""
    g = df.groupby(["qid", "tier"])
    hits = [((d.loc[d.label == 1, score_col] >= thr).any()) for _, d in g if (d.label == 1).any()]
    return float(np.mean(hits)) if hits else float("nan")


# ---- bootstrap ------------------------------------------------------------
def cluster_bootstrap(df: pd.DataFrame, score_cols: Sequence[str], thresholds: Dict[str, float],
                      n_boot: int, seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Resample questions with replacement; return metric distributions per detector."""
    rng = np.random.RandomState(seed)
    qids = df["qid"].unique()
    groups = {q: np.flatnonzero((df["qid"] == q).to_numpy()) for q in qids}
    y = df["label"].to_numpy()
    S = {c: df[c].to_numpy() for c in score_cols}
    out = {c: {"auroc": [], "tpr": [], "fpr": []} for c in score_cols}
    for _ in range(n_boot):
        pick = rng.choice(qids, size=len(qids), replace=True)
        idx = np.concatenate([groups[q] for q in pick])
        for c in score_cols:
            out[c]["auroc"].append(safe_auroc(y[idx], S[c][idx]))
            m = flag_metrics(y[idx], S[c][idx], thresholds[c])
            out[c]["tpr"].append(m["tpr"])
            out[c]["fpr"].append(m["fpr"])
    return {c: {k: np.array(v) for k, v in d.items()} for c, d in out.items()}


def ci(x: np.ndarray, alpha: float = 0.05):
    x = x[~np.isnan(x)]
    return (float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))) if len(x) else (np.nan, np.nan)


def paired_diff(a: np.ndarray, b: np.ndarray):
    """Paired bootstrap difference a-b on identical resamples: (mean, lo, hi, two-sided p)."""
    d = a - b
    d = d[~np.isnan(d)]
    lo, hi = ci(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), lo, hi, float(min(p, 1.0))


# ---- protocol -------------------------------------------------------------
def run_protocol(df: pd.DataFrame, detectors: Dict[str, object], seed: int, fpr_target: float = 0.01,
                 train_tiers=("A0",), test_tiers=("A0", "A1", "A2"), n_boot: int = 1000,
                 frac=(0.5, 0.2, 0.3), reference: str = "fusion") -> pd.DataFrame:
    """Fit on ``train_tiers`` (+clean), calibrate threshold, evaluate on each test tier."""
    fit_q, cal_q, test_q = split_qids(df["qid"], seed, frac)
    is_train_tier = df["tier"].isin(list(train_tiers) + ["clean"])
    fit_df = df[is_train_tier & df["qid"].isin(fit_q)]
    cal_df = df[is_train_tier & df["qid"].isin(cal_q) & (df["label"] == 0)]

    scores, thr = {}, {}
    test_df_all = df[df["qid"].isin(test_q)].copy()
    for name, det in detectors.items():
        det.fit(fit_df)
        thr[name] = calibrate_threshold(det.score(cal_df), fpr_target)
        test_df_all[f"s_{name}"] = det.score(test_df_all)
        scores[name] = f"s_{name}"

    rows = []
    for tier in test_tiers:
        part = test_df_all[(test_df_all["tier"] == tier) | (test_df_all["tier"] == "clean")]
        if not (part["tier"] == tier).any():
            continue
        boots = cluster_bootstrap(part, list(scores.values()), {v: thr[k] for k, v in scores.items()},
                                  n_boot, seed)
        for name, col in scores.items():
            pt = flag_metrics(part["label"], part[col], thr[name])
            pt["auroc"] = safe_auroc(part["label"], part[col])
            pt["query_tpr"] = query_level_detection(part, col, thr[name])
            ref = boots[scores[reference]] if reference in scores else None
            for metric in ("auroc", "tpr", "fpr"):
                lo, hi = ci(boots[col][metric])
                row = dict(seed=seed, tier=tier, detector=name, metric=metric, value=pt[metric],
                           lo=lo, hi=hi, threshold=thr[name])
                if ref is not None and name != reference and metric in ("auroc", "tpr"):
                    m, dlo, dhi, p = paired_diff(ref[metric], boots[col][metric])
                    row.update(ref_minus_this=m, diff_lo=dlo, diff_hi=dhi, p_value=p)
                rows.append(row)
            rows.append(dict(seed=seed, tier=tier, detector=name, metric="query_tpr",
                             value=pt["query_tpr"], lo=np.nan, hi=np.nan, threshold=thr[name]))
    return pd.DataFrame(rows)


# ---- downstream utility: ASR / accuracy after filtering ------------------------
def utility_after_filtering(sets: Sequence[RetrievedSet], score_lookup: Dict[str, float], thr: float,
                            answer_fn: Callable) -> Dict[str, float]:
    """Drop flagged passages, regenerate, and measure ASR (attacker answer) and accuracy (true answer).

    ``score_lookup`` maps ``f"{qid}|{tier}|{pid}"`` to the detector score.
    """
    asr, acc = [], []
    for rs in sets:
        keep = [p for p in rs.passages if score_lookup[f"{rs.qid}|{rs.tier}|{p.pid}"] < thr]
        ans = answer_fn(rs.question, keep)
        asr.append(answer_match(ans, rs.target))
        acc.append(answer_match(ans, rs.correct))
    return {"asr": float(np.mean(asr)), "accuracy": float(np.mean(acc)), "n": len(sets)}
