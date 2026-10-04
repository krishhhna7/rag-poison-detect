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
| Target-file loader | Verified against the real released NQ and HotpotQA files (100 questions each) |
| Corpus loader, qrels-based gold lookup | Written to the BEIR standard layout; **not yet run on the real BEIR zip** |
| Real results | **None yet.** Toy numbers are meaningless |

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

## Running on Colab (free T4)
Open `notebooks/colab_run.ipynb` in Colab (GitHub tab, or
`https://colab.research.google.com/github/krishhhna7/rag-poison-detect/blob/main/notebooks/colab_run.ipynb`),
choose a T4 GPU runtime and run the cells in order. Uses `configs/small_gpu.yaml` (Qwen2.5-3B in fp16,
NQ only). The raw 1.5 GB corpus is kept on Colab's disposable local disk and only read once by the `prepare` stage; the small sub-corpus, embeddings, models and results are stored on Google Drive. Do the 10-question run first.
The 6 GB model is cached on Colab's local disk (re-downloads in ~1 min); never on Drive, which may be nearly full.
Finished stages are cached on disk (`sets_*.jsonl`, `features_<tier>.csv`): if you change the attack,
use a new `run_name` or delete the cached files.

## Real run
1. **Data** (verified formats; see "Data sources" below):
   `source setup_env.sh && bash scripts/download_data.sh nq`
   puts the raw corpus in `$RAGDET_DATA/nq/` (disposable) and the small files (qrels, targets) in
   `$RAGDET_ROOT/data_small/nq/`. Then run `python -m ragdet.cli prepare --config ...` once to cache the sub-corpus + embeddings.
2. `bash scripts/run_all.sh configs/default.yaml` (or `configs/small_gpu.yaml`, `configs/hotpotqa.yaml`).
3. Outputs land in `$RAGDET_ROOT/runs/<run_name>/`: `sets_*.jsonl`, `features.csv`,
   `results_raw.csv` (per seed, with bootstrap CIs and paired tests), `results_summary.csv`.
   Copy these back to your own machine and commit the small result files.

## Data sources
* **Targets + poison texts:** `results/adv_targeted_results/{nq,hotpotqa,msmarco}.json` in
  github.com/sleeepeer/PoisonedRAG. Verified: each file is ONE JSON object of 100 questions
  keyed by id, with fields `id, question, "correct answer", "incorrect answer", adv_texts` (5
  released poison texts per question). A0 uses these texts directly, so it is reproducible.
* **Corpus:** BEIR `nq` / `hotpotqa` zips (URL taken from PoisonedRAG's `prepare_dataset.py`).
  Query ids in the targets file (e.g. `test1`) are BEIR *query* ids, so gold passages are found via
  `qrels/test.tsv`, not by id equality.
* **Not yet verified:** the BEIR zip layout on disk (we assume the standard `corpus.jsonl`,
  `qrels/test.tsv`) and the download size. Check after downloading.
* **Answer matching** uses whole-word containment. HotpotQA has 15/100 yes/no targets, so
  `configs/hotpotqa.yaml` drops them (85 remain); NQ keeps 97/100 if you set `drop_binary: true`.

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
* **Observed in the 10-question pilot:** with `n_poison=5` and `top_k=5`, A0 sets were 100% poison (no clean
  neighbours), which makes per-passage detection degenerate. The main setting is therefore `n_poison=3`;
  `n_poison` in {1, 5} are ablations. Attack success must be reported per tier (pilot: A0 10/10, A1/A2 7/10).
* **False-alarm target is 5%, not 1%:** with ~100 questions there are only ~100 clean calibration passages.
  Benign passages are reported in two populations (`fpr_clean`: clean sets; `fpr_mixed`: next to poison),
  plus `auroc_within` (poison vs its neighbours in the same set) and set-level `query_fpr_clean`.
* A1 rewrites drop the question prefix, so some may fail retrieval; the fallback rate is printed
  by the `attack` stage and must be reported.
* A2 attacker queries a detector fitted on the same data distribution (strong attacker).
* Attention needs eager attention: memory heavy, white-box generator access only.
