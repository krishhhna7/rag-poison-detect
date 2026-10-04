import json
import os

from ragdet import cli


def _run(stage, cfg_path):
    cli.main([stage, "--config", cfg_path])


def _lines(path):
    return [l for l in open(path, encoding="utf-8") if l.strip()]


def test_attack_and_a2_resume_after_interruption(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RAGDET_ROOT", str(tmp_path))
    cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "toy.yaml")
    run = tmp_path / "runs" / "toy"
    _run("attack", cfg)
    full_A1 = _lines(run / "sets_A1.jsonl")
    assert len(full_A1) == 20

    # simulate a disconnect after 8 questions (plus a half-written last line)
    part = run / "attack_partial.jsonl"
    keep = _lines(part)[:8]
    part.write_text("".join(keep) + '{"fp": "broken')
    os.remove(run / "sets_A1.jsonl"); os.remove(run / "sets_A0.jsonl"); os.remove(run / "sets_clean.jsonl")
    capsys.readouterr()
    _run("attack", cfg)
    assert "resuming: 8 of 20" in capsys.readouterr().out
    assert len(_lines(run / "sets_A1.jsonl")) == 20 and len(_lines(run / "sets_clean.jsonl")) == 20

    _run("features", cfg)
    _run("a2", cfg)
    p2 = run / "a2_partial.jsonl"
    p2.write_text("".join(_lines(p2)[:5]))
    os.remove(run / "sets_A2.jsonl")
    capsys.readouterr()
    _run("a2", cfg)
    assert "resuming: 5 of 20" in capsys.readouterr().out
    assert len(_lines(run / "sets_A2.jsonl")) == 20


def test_changed_settings_ignore_stale_partial(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RAGDET_ROOT", str(tmp_path))
    cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "toy.yaml")
    _run("attack", cfg)
    capsys.readouterr()
    cli.main(["attack", "--config", cfg, "--set", "attack.n_poison=2"])   # different setting, same run name
    assert "resuming" not in capsys.readouterr().out                       # old partial rows must not be reused
