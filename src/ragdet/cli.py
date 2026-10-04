"""Command line entry point:  python -m ragdet.cli <stage> --config configs/xxx.yaml

Stages (each saves to <workspace>/runs/<run_name>/ so a crashed job can resume):
  prepare   build + cache the sub-corpus and passage embeddings (GPU helps; run once)
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
from .data import cached_subcorpus, load_qrels, load_targets, make_toy_dataset
from .retrieval import Retriever, build_embedder


def build_components(cfg, need_generator: bool = True):
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
    from .perplexity import HFPerplexity
    all_targets = load_targets(cfg.data.targets_path, getattr(cfg.data, "drop_binary", False))
    qrels = load_qrels(cfg.data.qrels_path)   # query id -> gold corpus ids (ids differ from query ids)
    # Gold passages for ALL targets go in the sub-corpus, so a 10-question test run and the full run
    # share one cached sub-corpus and one embedding file (the raw corpus is then needed only once).
    gold = [g for t in all_targets for g in qrels.get(t.qid, [])]
    missing = sum(t.qid not in qrels for t in all_targets)
    targets = all_targets[: cfg.data.n_targets]
    print(f"{len(targets)} targets used ({len(all_targets)} in file), {len(gold)} gold passages, "
          f"{missing} targets without qrels", flush=True)
    cache_dir = os.path.join(cfg.paths.workspace, "cache")
    ids, texts, key = cached_subcorpus(cfg.data.corpus_path, gold, cfg.data.corpus_size, cfg.seed, cache_dir)
    print(f"sub-corpus: {len(ids)} passages (key {key})", flush=True)
    emb = build_embedder(cfg.retriever)
    cache = os.path.join(cache_dir, f"emb_{cfg.retriever.name}_{key}.npy")
    ret = Retriever.build(emb, ids, texts, cache)
    print("retriever ready", flush=True)
    if not need_generator:
        return dict(targets=targets, retriever=ret, embedder=emb)
    from .generation import HFGenerator
    gen = HFGenerator(cfg.generator.hf_id, cfg.generator.dtype, cfg.generator.max_new_tokens,
                      cfg.generator.load_4bit, cfg.generator.attn_layers)
    print("generator ready", flush=True)
    return dict(targets=targets, retriever=ret, embedder=emb, generator=gen, llm=gen,
                ppl_fn=HFPerplexity(cfg.ppl.hf_id))


def _feat_path(cfg):
    return P.out_path(cfg, "features.csv")


def cmd_prepare(cfg, c):
    print("Prepare done: sub-corpus and embeddings are cached. The raw 1.5 GB corpus is no longer needed.")


def cmd_attack(cfg, c):
    sets, diag = P.build_attack_sets(cfg, c["targets"], c["retriever"], c["llm"], c["ppl_fn"])
    for name, s in sets.items():
        P.save_sets(cfg, name, s)
    for name in ("A0", "A1"):
        print(f"{name} retrieval success (>=1 poison in top-k): "
              f"{sum(any(p.is_poison for p in s.passages) for s in sets[name]) / len(sets[name]):.2f}")
    print("A1 diagnostics:", diag)


def cmd_features(cfg, c):
    """Per-tier feature files are cached, so re-running after a disconnect skips finished tiers."""
    frames = []
    for tier in ("clean", "A0", "A1"):
        sets_file = P.out_path(cfg, f"sets_{tier}.jsonl")
        if not os.path.exists(sets_file):
            continue
        cache = P.out_path(cfg, f"features_{tier}.csv")
        if os.path.exists(cache):
            print(f"[{tier}] using cached {cache}")
            frames.append(pd.read_csv(cache))
            continue
        df_t = P.build_features(cfg, {tier: P.load_sets(cfg, tier)}, c["embedder"], c["ppl_fn"], c["generator"])
        df_t.to_csv(cache, index=False)
        frames.append(df_t)
    df = pd.concat(frames, ignore_index=True)
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
    pd.set_option("display.width", 170)
    tpath = getattr(cfg.data, "targets_path", None)
    if tpath and os.path.exists(tpath):
        from .data import load_targets
        summ_atk = P.attack_summary(df, load_targets(tpath, getattr(cfg.data, "drop_binary", False)))
        summ_atk.to_csv(P.out_path(cfg, "attack_summary.csv"), index=False)
        print("ATTACK SUMMARY (poison_in_topk = mean poisoned passages among the retrieved top-k)")
        print(summ_atk.to_string(index=False), "\n")
    res = P.evaluate_all(cfg, df)
    res.to_csv(P.out_path(cfg, "results_raw.csv"), index=False)
    summ = P.summarise(res)
    summ.to_csv(P.out_path(cfg, "results_summary.csv"), index=False)
    show = ["auroc", "auroc_within", "tpr", "fpr_clean", "fpr_mixed", "query_tpr", "query_fpr_clean"]
    print("DETECTION (mean over seeds)")
    print(summ[summ.metric.isin(show)].pivot_table(index=["tier", "detector"], columns="metric",
                                                  values="mean").round(3)[[m for m in show]].to_string())


def cmd_toy(cfg, c):
    cmd_attack(cfg, c)
    cmd_features(cfg, c)
    cmd_a2(cfg, c)
    cmd_evaluate(cfg)
    print("\nNOTE: toy run uses test doubles; these numbers only prove the plumbing works.")


STAGES = {"prepare": cmd_prepare, "attack": cmd_attack, "features": cmd_features, "a2": cmd_a2, "evaluate": cmd_evaluate, "toy": cmd_toy}


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
    comps = None if a.stage == "evaluate" else build_components(cfg, need_generator=a.stage != "prepare")
    STAGES[a.stage](cfg, comps)


if __name__ == "__main__":
    main()
