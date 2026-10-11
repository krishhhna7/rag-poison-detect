import os

from ragdet import cli


def test_report_stage_writes_all_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGDET_ROOT", str(tmp_path))
    cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "toy.yaml")
    cli.main(["toy", "--config", cfg])                     # toy end-to-end also runs the report
    text = open(tmp_path / "analysis_outputs.txt", encoding="utf-8").read()
    for header in ["RUN: toy", "DETECTION, cross-fitted", "FUSION MINUS DETECTOR", "ABLATIONS",
                   "SEMANTIC DETECTOR", "END-TO-END", "PAIRED COMPARISONS"]:
        assert header in text, header
    assert "(missing: attack_summary.csv)" in text         # toy has no targets file; missing files must not crash


def test_wilson_interval_known_value():
    from ragdet.analysis import wilson
    lo, hi = wilson(0.12, 100)
    assert abs(lo - 0.070) < 0.003 and abs(hi - 0.198) < 0.003


def test_attack_refuses_to_overwrite_foreign_sets(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setenv("RAGDET_ROOT", str(tmp_path))
    cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "toy.yaml")
    run = tmp_path / "runs" / "toy"
    run.mkdir(parents=True)
    (run / "sets_A1.jsonl").write_text("{}\n")             # sets from an older version, no checkpoint file
    with pytest.raises(SystemExit):
        cli.main(["attack", "--config", cfg])
