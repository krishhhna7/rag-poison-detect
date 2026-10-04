"""Dataset utilities.

Real experiments use BEIR-format corpora (NQ, HotpotQA) plus a target file of
(question, correct answer, attacker answer) records, as in PoisonedRAG.
Embedding a multi-million-passage corpus is expensive, so we build a *sub-corpus*
= gold passages for the selected targets + random distractors. This is a
documented scale limitation (see README and the report's limitations section).
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

from .core import Target, normalize_text


# ---- readers ------------------------------------------------------------
def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_beir_corpus(path: str) -> Dict[str, str]:
    """BEIR corpus.jsonl: {'_id', 'title', 'text'} -> {id: 'title. text'}."""
    corpus = {}
    for r in read_jsonl(path):
        title = (r.get("title") or "").strip()
        text = (r.get("text") or "").strip()
        corpus[str(r["_id"])] = f"{title}. {text}" if title else text
    return corpus


def load_qrels(path: str) -> Dict[str, List[str]]:
    """BEIR qrels TSV (query-id, corpus-id, score) -> {query_id: [relevant corpus ids]}."""
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] == "query-id":
                continue
            if float(parts[2]) > 0:
                out.setdefault(parts[0], []).append(parts[1])
    return out


def load_targets(path: str, drop_binary: bool = False) -> List[Target]:
    """Load attack targets.

    Accepts the PoisonedRAG release format (verified): ONE JSON object mapping
    id -> {id, question, "correct answer", "incorrect answer", adv_texts[5]}; also a JSON list
    or JSONL of such records. ``drop_binary`` removes yes/no questions, whose answers are
    too easy to match by chance (15/100 in the HotpotQA file, 3/100 in NQ).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "question" not in obj:
            rows = [dict(v, id=v.get("id", k)) for k, v in obj.items()]
        else:
            rows = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:  # JSONL
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    out = []
    for r in rows:
        correct = r.get("correct answer", r.get("correct"))
        if isinstance(correct, list):
            correct = correct[0]
        target = r.get("incorrect answer", r.get("target"))
        t = Target(str(r["id"]), r["question"], str(correct), str(target), list(r.get("adv_texts", [])))
        if drop_binary and (normalize_text(t.correct) in ("yes", "no") or normalize_text(t.target) in ("yes", "no")):
            continue
        out.append(t)
    return out


def stream_subcorpus(path: str, gold_ids: List[str], n_total: int, seed: int = 0) -> Tuple[List[str], List[str]]:
    """One pass over a BEIR corpus.jsonl: keep all gold passages + a uniform random sample of the rest.

    Reservoir sampling means the multi-million-passage corpus is never held in memory
    (important on free Colab, ~12 GB RAM). Returns (ids, texts) with gold passages first.
    """
    rng = random.Random(seed)
    gold_set = set(map(str, gold_ids))
    k = max(0, n_total - len(gold_set))
    gold, reservoir, seen = {}, [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = str(r["_id"])
            title = (r.get("title") or "").strip()
            text = (r.get("text") or "").strip()
            doc = f"{title}. {text}" if title else text
            if pid in gold_set:
                gold[pid] = doc
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append((pid, doc))
            else:
                j = rng.randrange(seen)
                if j < k:
                    reservoir[j] = (pid, doc)
    ids = list(gold) + [i for i, _ in reservoir]
    texts = [gold[i] for i in gold] + [d for _, d in reservoir]
    return ids, texts


def build_subcorpus(corpus: Dict[str, str], gold_ids: List[str], n_total: int,
                    seed: int = 0) -> Tuple[List[str], List[str]]:
    """Gold passages + random distractors up to ``n_total`` passages."""
    rng = random.Random(seed)
    gold = [g for g in dict.fromkeys(gold_ids) if g in corpus]
    rest = [i for i in corpus if i not in set(gold)]
    rng.shuffle(rest)
    ids = gold + rest[: max(0, n_total - len(gold))]
    return ids, [corpus[i] for i in ids]


# ---- toy data (test double for CPU smoke runs; never used for results) ----
_COUNTRIES = ["Aldoria", "Brevania", "Corvany", "Delmora", "Estoria", "Faldwyn", "Gorvia",
              "Halmora", "Ithria", "Jandor", "Kelvia", "Lorwen", "Mostria", "Nervia",
              "Ostmark", "Pelvora", "Quenya", "Rostia", "Sildara", "Tovrin"]
_CITIES = ["Marlow", "Kestrel", "Dunmere", "Ashford", "Velton", "Oakhurst", "Riverton",
           "Stonebridge", "Fairhaven", "Norwick", "Eastgate", "Highcliff", "Lakemont",
           "Pinecrest", "Redwater", "Silverton", "Thornby", "Underhill", "Westmoor", "Yarrow"]


def make_toy_dataset(n_targets: int = 20, n_distractors: int = 200, seed: int = 0):
    """Synthetic 'capital of X' world: returns (corpus_ids, corpus_texts, targets)."""
    rng = random.Random(seed)
    n_targets = min(n_targets, len(_COUNTRIES))
    ids, texts, targets = [], [], []
    for i in range(n_targets):
        country, city = _COUNTRIES[i], _CITIES[i]
        wrong = _CITIES[(i + 7) % len(_CITIES)]
        ids.append(f"gold{i}")
        texts.append(f"{city} is the capital city of {country}. It is the seat of government "
                     f"and the largest urban centre in {country}.")
        targets.append(Target(f"t{i}", f"What is the capital of {country}?", city, wrong))
    topics = ["river", "mountain", "harbour", "market", "festival", "railway", "museum", "forest"]
    for j in range(n_distractors):
        c, t = rng.choice(_COUNTRIES), rng.choice(topics)
        ids.append(f"d{j}")
        texts.append(f"The {t} region of {c} is known for its long history, local traditions and "
                     f"seasonal visitors. Number {j} in the regional survey of {t} sites.")
    return ids, texts, targets
