"""Detectors.

Every detector maps a feature table to a *suspicion score* (higher = more likely
poisoned). Baselines and the fused detector share the same learner and differ only
in the feature columns they see, so differences are attributable to the signals,
not to model capacity.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_groups


class SingleFeatureThreshold:
    """Classic heuristic filter on one feature (e.g. perplexity).

    The direction (is high or low suspicious?) is chosen by training AUROC rather than
    assumed, which avoids baking in a hypothesis about poisoned-text statistics.
    """

    def __init__(self, column: str):
        self.column, self.sign = column, 1.0

    def fit(self, df: pd.DataFrame):
        auc = roc_auc_score(df["label"], df[self.column])
        self.sign = 1.0 if auc >= 0.5 else -1.0
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.sign * df[self.column].to_numpy(dtype=float)


class LearnedDetector:
    """Logistic regression (default) or gradient boosting over selected columns."""

    def __init__(self, columns: Sequence[str], model: str = "logreg", seed: int = 0):
        self.columns = list(columns)
        if model == "gbdt":
            clf = GradientBoostingClassifier(random_state=seed, n_estimators=150, max_depth=3)
            self.pipe = make_pipeline(clf)
        else:
            self.pipe = make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                                         random_state=seed))

    def fit(self, df: pd.DataFrame):
        self.pipe.fit(df[self.columns].to_numpy(float), df["label"].to_numpy(int))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(df[self.columns].to_numpy(float))[:, 1]


def build_detectors(model: str = "logreg", seed: int = 0) -> Dict[str, object]:
    g = feature_groups()
    internal = g["attn"] + g["loo"] + g["ent"]
    return {
        # heuristic single-feature baselines
        "ppl_filter": SingleFeatureThreshold("ppl_log"),
        "attn_only_heur": SingleFeatureThreshold("attn_share"),
        # learned single-group baselines (same learner as the fusion)
        "ppl": LearnedDetector(g["ppl"], model, seed),
        "sem": LearnedDetector(g["sem"], model, seed),
        "attn": LearnedDetector(g["attn"], model, seed),
        "internal": LearnedDetector(internal, model, seed),
        # proposed multi-signal fusion
        "fusion": LearnedDetector(g["ppl"] + g["sem"] + internal, model, seed),
    }


def ablations(model: str = "logreg", seed: int = 0) -> Dict[str, LearnedDetector]:
    """Fusion with one signal group removed at a time."""
    g = feature_groups()
    allg = ["ppl", "sem", "attn", "loo", "ent"]
    return {f"fusion_minus_{drop}": LearnedDetector(sum((g[x] for x in allg if x != drop), []), model, seed)
            for drop in allg}


def calibrate_threshold(clean_scores: np.ndarray, target_fpr: float) -> float:
    """Threshold such that ~target_fpr of *benign* calibration passages are flagged.

    Must be computed on held-out benign data from the *training* split only; test-set
    FPR is then measured, not assumed.
    """
    return float(np.quantile(np.asarray(clean_scores), 1.0 - target_fpr))
