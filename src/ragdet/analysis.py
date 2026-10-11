"""Regenerate every table used in the report from the saved result files (CPU only).

``make_report`` reads, for each run folder, the files written by the pipeline
(attack_summary.csv, results_summary.csv, significance.csv, features.csv, filter_results.jsonl,
oof_flags.csv) and writes ONE text file with all tables, so nothing has to be copied by hand.
A missing file never crashes the report; it is listed as missing.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SHOW = ["auroc", "auroc_within", "tpr", "fpr_clean", "fpr_mixed", "query_tpr", "query_fpr_clean"]
KEY_DETECTORS = ["sem", "internal", "attn", "ppl", "ppl_filter"]
PAIRS = [("internal", "sem"), ("fusion_minus_ppl", "fusion"), ("fusion_minus_ppl", "sem"),
         ("fusion_minus_ppl", "internal"), ("fusion", "sem"), ("fusion", "internal")]


def wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def tpr_diff(oof: pd.DataFrame, tier: str, a: str, b: str, n_boot: int = 2000, seed: int = 0):
    """Paired cluster bootstrap of TPR(a) - TPR(b) on poison passages of ``tier`` (first repetition)."""
    p = oof[(oof["tier"] == tier) & (oof["label"] == 1)]
    g = p.groupby("qid")[[f"f_{a}", f"f_{b}"]].agg(["sum", "count"])
    sa, sb = g[(f"f_{a}", "sum")].to_numpy(), g[(f"f_{b}", "sum")].to_numpy()
    n = g[(f"f_{a}", "count")].to_numpy()
    rng = np.random.RandomState(seed)
    d = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.randint(0, len(n), len(n))
        d[k] = sa[i].sum() / n[i].sum() - sb[i].sum() / n[i].sum()
    lo, hi = np.quantile(d, [0.025, 0.975])
    return float(sa.sum() / n.sum() - sb.sum() / n.sum()), float(lo), float(hi)


def _section(lines: List[str], title: str) -> None:
    lines += ["", "-" * 78, title, "-" * 78]


def _read(path: str, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def report_run(ws: str, run: str) -> List[str]:
    root = os.path.join(ws, "runs", run)
    out: List[str] = ["=" * 78, f"RUN: {run}", "=" * 78]
    pd.set_option("display.width", 200)

    atk = _read(os.path.join(root, "attack_summary.csv"))
    _section(out, "ATTACK SUMMARY (asr = attack success rate; accuracy = correct-answer rate)")
    out.append(atk.round(3).to_string(index=False) if atk is not None else "(missing: attack_summary.csv)")

    summ = _read(os.path.join(root, "results_summary.csv"))
    _section(out, "DETECTION, cross-fitted, mean over repetitions (all 100 questions tested once per repetition)")
    if summ is not None:
        tab = summ[summ.metric.isin(SHOW)].pivot_table(index=["tier", "detector"], columns="metric", values="mean")
        out.append(tab[[m for m in SHOW if m in tab.columns]].round(3).to_string())
    else:
        out.append("(missing: results_summary.csv)")

    sig = _read(os.path.join(root, "significance.csv"))
    _section(out, "FUSION MINUS DETECTOR (paired bootstrap; CI excluding 0 = clear difference; p=0.000 means p<0.002)")
    if sig is not None and len(sig):
        k = sig[sig.detector.isin(KEY_DETECTORS) & sig.metric.isin(["auroc", "tpr"])]
        out.append(k.round(3).to_string(index=False))
        _section(out, "ABLATIONS: fusion minus fusion-without-one-group (positive = the group helps)")
        ab = sig[sig.detector.str.startswith("fusion_minus") & sig.metric.isin(["auroc", "tpr"])]
        out.append(ab.round(3).to_string(index=False))
    else:
        out.append("(missing: significance.csv)")

    feats = _read(os.path.join(root, "features.csv"), dtype={"qid": str, "pid": str})
    _section(out, "SEMANTIC DETECTOR: standardised logistic weights (trained on A0 + clean, all questions)")
    if feats is not None:
        from .detector import build_detectors
        det = build_detectors("logreg", 0)["sem"].fit(feats[feats.tier.isin(["A0", "clean"])])
        w = pd.Series(det.pipe[-1].coef_[0], index=det.columns)
        out.append(w.sort_values(key=abs, ascending=False).round(2).to_string())
    else:
        out.append("(missing: features.csv)")

    fpath = os.path.join(root, "filter_results.jsonl")
    _section(out, "END-TO-END: attack success (asr) and accuracy AFTER removing flagged passages, with 95% Wilson CIs")
    if os.path.exists(fpath):
        df = pd.DataFrame([json.loads(l) for l in open(fpath) if l.strip()])
        for m in ("asr", "accuracy"):
            ci = [wilson(p, n) for p, n in zip(df[m], df["n"])]
            df[m + "_lo"], df[m + "_hi"] = [c[0] for c in ci], [c[1] for c in ci]
        cols = ["detector", "tier", "asr", "asr_lo", "asr_hi", "accuracy", "accuracy_lo", "accuracy_hi",
                "poison_removed", "benign_removed"]
        out.append(df[cols].round(3).to_string(index=False))
        if atk is not None:
            out.append("\nNo-filter baseline (same tiers): see ATTACK SUMMARY above.")
    else:
        out.append("(missing: filter_results.jsonl)")

    oof = _read(os.path.join(root, "oof_flags.csv"), dtype={"qid": str, "pid": str})
    _section(out, "PAIRED COMPARISONS of poison caught (TPR difference, first repetition; 95% bootstrap CI)")
    if oof is not None:
        for tier in ("A0", "A1", "A2"):
            if not (oof["tier"] == tier).any():
                continue
            for a, b in PAIRS:
                if f"f_{a}" in oof and f"f_{b}" in oof:
                    d, lo, hi = tpr_diff(oof, tier, a, b)
                    out.append(f"{tier}: TPR({a}) - TPR({b}) = {d:+.3f}   CI [{lo:+.3f}, {hi:+.3f}]")
    else:
        out.append("(missing: oof_flags.csv)")
    return out


def make_report(ws: str, runs: Sequence[str], out_file: str) -> str:
    lines: List[str] = ["ANALYSIS OUTPUTS (generated by `python -m ragdet.cli report`)",
                        "These are the final numbers; quote them from this file or the CSVs it reads."]
    for r in runs:
        lines += report_run(ws, r)
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    return text
