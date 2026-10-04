import os

import pandas as pd

from ragdet import cli
from ragdet.config import load_config


def test_toy_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGDET_ROOT", str(tmp_path))
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "toy.yaml")
    cli.main(["toy", "--config", cfg_path])
    run = tmp_path / "runs" / "toy"
    for f in ["sets_clean.jsonl", "sets_A0.jsonl", "sets_A1.jsonl", "sets_A2.jsonl",
              "features.csv", "results_raw.csv", "results_summary.csv"]:
        assert (run / f).exists(), f
    feats = pd.read_csv(run / "features.csv")
    assert set(feats.tier) == {"clean", "A0", "A1", "A2"}
    assert feats.loc[feats.tier == "clean", "label"].sum() == 0     # clean sets contain no poison
    assert feats.loc[feats.tier == "A0", "label"].sum() > 0
    res = pd.read_csv(run / "results_raw.csv")
    assert {"fusion", "ppl_filter", "attn", "sem"} <= set(res.detector)
