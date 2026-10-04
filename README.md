# Detecting adversarially retrieved contexts in RAG under stealth-aware poisoning

Course project (University of Auckland, deep learning). Authors: Aniketh Rao, Krishna Patil.

**Research question.** Do single-signal poisoning detectors (perplexity, attention, embedding
structure) degrade under fluency-preserving and detector-aware attacks, and does fusing
generator-internal signals with set-level semantic consistency degrade more gracefully?

## Status: what has and has not been verified

| Component | Status |
|---|---|
| Config, retrieval + poison injection, features, detectors, evaluation protocol | Unit-tested on CPU (`pytest`) |
| Whole pipeline plumbing | Runs end to end on a synthetic toy world with test doubles (`toy` stage) |
| `HFGenerator` (attention extraction, leave-one-out), `HFEmbedder`, `HFPerplexity` | **Written but NOT yet run on a GPU.** Expect to debug memory/shape issues on first run |
| Real datasets, real results | **None yet.** Toy numbers are meaningless |

## Layout
```
configs/     default.yaml (cluster), small_gpu.yaml (T4), toy.yaml (CPU plumbing check)
src/ragdet/  core, config, data, retrieval, perplexity, generation, signals, attacks,
             features, detector, evaluate, pipeline, cli
tests/       unit tests + toy end-to-end test
scripts/     job.sh (Slurm template, unconfirmed), run_all.sh
setup_env.sh cluster environment (paths/proxy marked CONFIRM)
```

## Quick start (CPU, no downloads)
```bash
pip install -r requirements.txt   # torch/transformers are optional for this
export PYTHONPATH=src
pytest -q
RAGDET_ROOT=./workspace python -m ragdet.cli toy --config configs/toy.yaml
```

## Real run
1. Data (in `$RAGDET_ROOT/data/nq/`): BEIR-format `corpus.jsonl` and a PoisonedRAG-format
   `targets.jsonl` (fields `id, question, correct answer, incorrect answer`). Verify the
   download sources from the BEIR and PoisonedRAG repositories; target ids must match BEIR ids.
2. `source setup_env.sh` (fill in the CONFIRM items), then `bash scripts/run_all.sh configs/default.yaml`.
3. Outputs land in `$RAGDET_ROOT/runs/<run_name>/`: `sets_*.jsonl`, `features.csv`,
   `results_raw.csv` (per seed, with bootstrap CIs and paired tests), `results_summary.csv`.
   Copy these back to your own machine and commit small result files.

## Experimental protocol (summarised)
* Attack tiers: **A0** PoisonedRAG-style; **A1** fluency-preserving rewrites chosen for low
  perplexity subject to being retrievable; **A2** detector-in-the-loop best-of-N.
  **A1 and A2 are our own stealth proxies**, not reproductions of SilentRetrieval or of the
  adaptive attacks in Choudhary et al. Label them as such in the report.
* Signals: perplexity; semantic consistency (query and intra-set similarity); NPAS-style
  attention share/density; leave-one-out answer shift; answer entropy. Set-relative z-scores.
* Detectors: `ppl_filter`, `attn_only_heur` (heuristics); `ppl`, `sem`, `attn`, `internal`
  (learned, same learner); `fusion` (all groups). `detector.ablations()` drops one group at a time.
* Train on A0 + clean only, test on A0/A1/A2. Splits are by question. Threshold calibrated on
  held-out benign training passages for a 1% target FPR; test FPR is measured and reported.
* Metrics: AUROC, TPR at calibrated threshold, measured FPR, query-level detection, cluster
  bootstrap CIs, paired bootstrap tests vs fusion, mean/std across seeds.
  `evaluate.utility_after_filtering` gives ASR and clean accuracy after removing flagged passages.

## Known limitations (carry into the report)
* Sub-corpus (gold + random distractors) rather than the full multi-million corpus.
* With `n_poison=5` and `top_k=5`, all retrieved passages can be poisoned, weakening set-relative
  features; ablate `n_poison` in {1, 3, 5}.
* A1 rewrites drop the question prefix, so some may fail retrieval; the fallback rate is printed
  by the `attack` stage and must be reported.
* A2 attacker queries a detector fitted on the same data distribution (strong attacker).
* Attention needs eager attention: memory heavy, white-box generator access only.
