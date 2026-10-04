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

from .core import Target


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


def load_targets(path: str) -> List[Target]:
    """Targets JSONL with fields: id, question, correct answer, incorrect answer.

    Field names follow the PoisonedRAG release; ``correct``/``target`` aliases
    are also accepted. The 'correct answer' field may be a string or a list.
    """
    out = []
    for r in read_jsonl(path):
        correct = r.get("correct answer", r.get("correct"))
        if isinstance(correct, list):
            correct = correct[0]
        target = r.get("incorrect answer", r.get("target"))
        out.append(Target(str(r["id"]), r["question"], str(correct), str(target)))
    return out


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
