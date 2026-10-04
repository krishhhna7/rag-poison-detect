"""YAML configuration loading.

Environment variables such as ``${RAGDET_ROOT}`` are expanded so the same
config works on a laptop, Kaggle and the cluster. If ``RAGDET_ROOT`` is unset
it defaults to ``./workspace``. On the cluster, point it at the large-storage
area (see README, "Cluster setup").
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import yaml


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _expand(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def load_config(path: str, overrides: dict | None = None) -> SimpleNamespace:
    os.environ.setdefault("RAGDET_ROOT", os.path.abspath("./workspace"))
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if overrides:
        for dotted, value in overrides.items():
            node = raw
            keys = dotted.split(".")
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = value
    cfg = _to_ns(_expand(raw))
    os.makedirs(cfg.paths.workspace, exist_ok=True)
    return cfg
