import numpy as np
import pandas as pd
import pytest

from ragdet.detector import LearnedDetector, SingleFeatureThreshold
from ragdet.evaluate import (all_metrics, cluster_bootstrap, crossfit, fold_assignment, paired_diff,
                             query_level_detection, query_level_false_alarm, run_protocol,
                             utility_after_filtering)
from ragdet.core import Passage, RetrievedSet


def synthetic(n_q=60, seed=0):
    """Clean sets (5 benign passages) + one attacked tier (3 poison + 2 benign) per question."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_q):
        for tier in ("clean", "A0", "A1"):
            for j in range(5):
                pois = int(tier != "clean" and j < 3)
                rows.append(dict(qid=f"q{i}", tier=tier, pid=f"{tier}{i}_{j}", label=pois,
                                 x=rng.randn() + 2.0 * pois, z=rng.randn()))
    return pd.DataFrame(rows)


def test_split_fpr_and_within_auroc():
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0])
    clean = np.array([True, True, True, True, False, False, False, False])  # last 4 = an attacked set
    s = np.array([0.9, 0.1, 0.1, 0.1, 0.8, 0.7, 0.6, 0.1])
    m = all_metrics(y, clean, s, s >= 0.5)
    assert m["tpr"] == 1.0
    assert m["fpr_clean"] == 0.25 and m["fpr_mixed"] == 0.5          # populations reported separately
    assert m["auroc_within"] == 1.0                                   # poison outranks its neighbours


def test_query_level_metrics():
    df = pd.DataFrame({"qid": ["a", "a", "b", "b"], "tier": "clean", "label": 0, "f": [True, False, False, False]})
    assert query_level_false_alarm(df, "f") == 0.5                   # 1 of 2 clean sets raised an alarm
    att = pd.DataFrame({"qid": ["a", "a", "b", "b"], "tier": "A0", "label": [1, 0, 1, 0],
                        "f": [True, False, False, True]})
    assert query_level_detection(att, "f") == 0.5                    # flagged a poison in 1 of 2 sets


def test_fold_assignment_is_balanced_and_deterministic():
    qs = [f"q{i}" for i in range(100)]
    f1, f2 = fold_assignment(qs, 5, 3), fold_assignment(qs, 5, 3)
    assert f1 == f2 and sorted(np.bincount(list(f1.values()))) == [20] * 5


def test_crossfit_tests_every_question_once_and_never_trains_on_it():
    df = synthetic()
    made = []

    class Spy(SingleFeatureThreshold):
        def fit(self, d):
            made.append(set(d["qid"]))
            return super().fit(d)

    out, thr = crossfit(df, lambda: {"x": Spy("x")}, seed=0, k=5)
    assert out["s_x"].notna().all()                                   # every row scored out-of-fold
    folds = fold_assignment(df["qid"], 5, 0)
    for f, fit_qids in enumerate(made):
        test_q = {q for q, v in folds.items() if v == f}
        cal_q = {q for q, v in folds.items() if v == (f + 1) % 5}
        assert not (fit_qids & test_q) and not (fit_qids & cal_q)     # no leakage into fit set


def test_run_protocol_end_to_end_and_threshold_calibration():
    df = synthetic()
    res, scored = run_protocol(df, lambda: {"x": SingleFeatureThreshold("x"), "z": SingleFeatureThreshold("z"),
                                            "fusion": LearnedDetector(["x", "z"])},
                               seed=0, fpr_target=0.05, train_tiers=("A0",), test_tiers=("A0", "A1"),
                               n_boot=50, k=5)
    v = res.set_index(["tier", "detector", "metric"])["value"]
    assert v[("A1", "x", "auroc")] > 0.85 and v[("A1", "z", "auroc")] < 0.65   # informative vs noise feature
    assert v[("A0", "x", "fpr_clean")] == pytest.approx(0.05, abs=0.05)         # realised, near target
    assert "p_value" in res.columns                                              # paired tests vs fusion


def test_bootstrap_ci_brackets_point_estimate():
    df = synthetic()
    out, _ = crossfit(df, lambda: {"x": SingleFeatureThreshold("x")}, seed=0)
    part = out[out.tier.isin(["A0", "clean"])]
    b = cluster_bootstrap(part, ["x"], 100, 0)
    pt = all_metrics(part.label, (part.tier == "clean"), part.s_x, part.f_x)["auroc"]
    lo, hi = np.quantile(b["x"]["auroc"], [0.025, 0.975])
    assert lo <= pt <= hi
    m, dlo, dhi, p = paired_diff(b["x"]["auroc"], b["x"]["auroc"] - 0.1)
    assert m == pytest.approx(0.1) and 0 <= p <= 1


def test_utility_after_filtering_regenerates_only_when_flagged():
    mk = lambda pid, pois: Passage(pid, f"text {pid}", pois)
    rs = RetrievedSet("q", "?", "right", "wrong", "A0", [mk("a", True), mk("b", False)])
    calls = []

    def answer_fn(q, passages):
        calls.append([p.pid for p in passages])
        return "right" if all(not p.is_poison for p in passages) else "wrong"

    flagged = {"q|A0|a": True, "q|A0|b": False}
    r = utility_after_filtering([rs], flagged, answer_fn, {"q|A0": "wrong"})
    assert r["asr"] == 0.0 and r["accuracy"] == 1.0 and calls == [["b"]]
    assert r["poison_removed"] == 1.0 and r["benign_removed"] == 0.0
    calls.clear()
    r2 = utility_after_filtering([rs], {"q|A0|a": False, "q|A0|b": False}, answer_fn, {"q|A0": "wrong"})
    assert calls == [] and r2["asr"] == 1.0                      # nothing flagged: original answer reused
