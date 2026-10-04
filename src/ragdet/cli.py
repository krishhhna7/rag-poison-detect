"""Command line entry point:  python -m ragdet.cli <stage> --config configs/xxx.yaml

Stages (each saves to <workspace>/runs/<run_name>/ so a crashed job can resume):
  attack    build clean / A0 / A1 retrieved sets
  features  extract per-passage features for all sets currently on disk
  a2        run the detector-in-the-loop attack, then append A2 features
  evaluate  train/calibrate/test detectors, write results tables
  toy       whole pipeline on the synthetic CPU-only dataset (plumbing check, NOT results)
"""
from __future__ import annotations

import argparse
import ast
import os

import pandas as pd

from . import pipeline as P
from .config import load_config
from .core import set_seed
from .data import build_subcorpus, load_beir_corpus, load_targets, make_toy_dataset, read_jsonl
from .retrieval import Retriever, build_embedder


def build_components(cfg):
    """Instantiate corpus, retriever, generator, perplexity scorer. Heavy imports are lazy."""
    if cfg.data.dataset == "toy":
        from .generation import ToyGenerator
        from .perplexity import UnigramPerplexity
        ids, texts, targets = make_toy_dataset(cfg.data.n_targets, cfg.data.corpus_size, cfg.seed)
        emb = build_embedder(cfg.retriever)
        ret = Retriever.build(emb, ids, texts)
        gen = ToyGenerator()
        return dict(targets=targets, retriever=ret, embedder=emb, generator=gen, llm=gen,
                    ppl_fn=UnigramPerplexity(texts))
    from .generation import HFGenerator
    from .perplexity import HFPerplexity
    corpus = load_beir_corpus(cfg.data.corpus_path)
    targets = load_targets(cfg.data.targets_path)[: cfg.data.n_targets]
    gold = [t.qid for t in targets]  # PoisonedRAG target ids match BEIR ids
    ids, texts = build_subcorpus(corpus, gold, cfg.data.corpus_size, cfg.seed)
    emb = build_embedder(cfg.retriever)
    cache = os.path.join(cfg.paths.workspace, "cache", f"{cfg.data.dataset}_{cfg.retriever.name}_{len(ids)}.npy")
    ret = Retriever.build(emb, ids, texts, cache)
    gen = HFGenerator(cfg.generator.hf_id, cfg.generator.dtype, cfg.generator.max_new_tokens,
                      cfg.generator.load_4bit, cfg.generator.attn_layers)
    return dict(targets=targets, retriever=ret, embedder=emb, generator=gen, llm=gen,
                ppl_fn=HFPerplexity(cfg.ppl.hf_id))


def _feat_path(cfg):
    return P.out_path(cfg, "features.csv")


def cmd_attack(cfg, c):
    sets, diag = P.build_attack_sets(cfg, c["targets"], c["retriever"], c["llm"], c["ppl_fn"])
    for name, s in sets.items():
        P.save_sets(cfg, name, s)
    for name in ("A0", "A1"):
        print(f"{name} retrieval success (>=1 poison in top-k): "
              f"{sum(any(p.is_poison for p in s.passages) for s in sets[name]) / len(sets[name]):.2f}")
    print("A1 diagnostics:", diag)


def cmd_features(cfg, c):
    tiers = {t: P.load_sets(cfg, t) for t in ("clean", "A0", "A1") if os.path.exists(P.out_path(cfg, f"sets_{t}.jsonl"))}
    df = P.build_features(cfg, tiers, c["embedder"], c["ppl_fn"], c["generator"])
    df.to_csv(_feat_path(cfg), index=False)
    print("wrote", _feat_path(cfg), df.shape)


def cmd_a2(cfg, c):
    df = pd.read_csv(_feat_path(cfg))
    det = P.fit_attacker_detector(cfg, df, cfg.seed)
    sets, stats = P.build_a2_sets(cfg, c["targets"], P.load_sets(cfg, "A0"), c["retriever"], c["llm"],
                                  c["ppl_fn"], c["embedder"], c["generator"], det)
    P.save_sets(cfg, "A2", sets)
    a2 = P.build_features(cfg, {"A2": sets}, c["embedder"], c["ppl_fn"], c["generator"])
    pd.concat([df[df.tier != "A2"], a2], ignore_index=True).to_csv(_feat_path(cfg), index=False)
    print("A2 stats:", stats)


def cmd_evaluate(cfg, c=None):
    df = pd.read_csv(_feat_path(cfg))
    res = P.evaluate_all(cfg, df)
    res.to_csv(P.out_path(cfg, "results_raw.csv"), index=False)
    summ = P.summarise(res)
    summ.to_csv(P.out_path(cfg, "results_summary.csv"), index=False)
    pd.set_option("display.width", 160)
    print(summ[summ.metric.isin(["auroc", "tpr", "fpr", "query_tpr"])].to_string(index=False))


def cmd_toy(cfg, c):
    cmd_attack(cfg, c)
    cmd_features(cfg, c)
    cmd_a2(cfg, c)
    cmd_evaluate(cfg)
    print("\nNOTE: toy run uses test doubles; these numbers only prove the plumbing works.")


STAGES = {"attack": cmd_attack, "features": cmd_features, "a2": cmd_a2, "evaluate": cmd_evaluate, "toy": cmd_toy}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. data.n_targets=50")
    a = ap.parse_args(argv)
    ov = {}
    for item in a.set:
        k, v = item.split("=", 1)
        try:
            v = ast.literal_eval(v)  # numbers / lists / bools from the command line
        except (ValueError, SyntaxError):
            pass  # keep as plain string
        ov[k] = v
    cfg = load_config(a.config, ov)
    set_seed(cfg.seed)
    comps = None if a.stage == "evaluate" else build_components(cfg)
    STAGES[a.stage](cfg, comps)


if __name__ == "__main__":
    main()
