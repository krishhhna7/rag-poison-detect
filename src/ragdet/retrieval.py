"""Dense retrieval with support for injecting (poison) passages at query time.

The base corpus embeddings are computed once and cached. Poison passages are
embedded separately and merged into the candidate scores at query time, so the
clean index is never mutated and clean/poisoned runs share the same corpus.
"""
from __future__ import annotations

import os
import re
import zlib
from typing import List, Optional, Sequence

import numpy as np

from .core import Passage


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------
class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (CPU-only).

    A *test double* used for unit tests and the toy smoke run. It is NOT used
    for any reported result.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.normalize = True

    def encode(self, texts: Sequence[str], is_query: bool = False, batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = zlib.crc32(tok.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else 1.0
        out = np.sqrt(out)  # sub-linear tf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-8, None)


class HFEmbedder:
    """Hugging Face encoder wrapper (Contriever: mean pooling; BGE: CLS pooling)."""

    def __init__(self, hf_id: str, pooling: str = "mean", normalize: bool = False,
                 query_prefix: str = "", device: Optional[str] = None, max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.pooling, self.normalize = pooling, normalize
        self.query_prefix, self.max_length = query_prefix, max_length

    def encode(self, texts: Sequence[str], is_query: bool = False, batch_size: int = 64) -> np.ndarray:
        torch = self.torch
        outs = []
        texts = [(self.query_prefix + t) if is_query else t for t in texts]
        starts = range(0, len(texts), batch_size)
        if len(texts) > 2000:
            from tqdm import tqdm
            starts = tqdm(starts, desc="embedding passages", unit=" batches", mininterval=5)
        for i in starts:
            batch = self.tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.model(**batch).last_hidden_state
            if self.pooling == "cls":
                emb = h[:, 0]
            else:
                m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
            if self.normalize:
                emb = torch.nn.functional.normalize(emb, dim=-1)
            outs.append(emb.float().cpu().numpy())
        return np.concatenate(outs, axis=0)


def build_embedder(rcfg):
    if rcfg.name == "hashing":
        return HashingEmbedder(getattr(rcfg, "dim", 512))
    return HFEmbedder(rcfg.hf_id, rcfg.pooling, rcfg.normalize, getattr(rcfg, "query_prefix", ""))


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------
class Retriever:
    """Dot-product retriever over a fixed corpus plus optional injected passages."""

    def __init__(self, embedder, corpus_ids: List[str], corpus_texts: List[str],
                 corpus_emb: np.ndarray):
        assert len(corpus_ids) == len(corpus_texts) == len(corpus_emb)
        self.embedder = embedder
        self.ids, self.texts, self.emb = corpus_ids, corpus_texts, corpus_emb

    @classmethod
    def build(cls, embedder, corpus_ids, corpus_texts, cache_path: Optional[str] = None):
        if cache_path and os.path.exists(cache_path):
            emb = np.load(cache_path)
        else:
            emb = embedder.encode(corpus_texts)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, emb)
        return cls(embedder, list(corpus_ids), list(corpus_texts), emb)

    def _scores(self, qvec: np.ndarray, extra: Optional[Sequence[Passage]]):
        base = self.emb @ qvec
        if not extra:
            return base, np.empty(0, dtype=np.float32)
        ev = self.embedder.encode([p.text for p in extra])
        return base, ev @ qvec

    def retrieve(self, question: str, k: int, extra: Optional[Sequence[Passage]] = None) -> List[Passage]:
        """Top-k passages for ``question`` with ``extra`` passages injected."""
        qvec = self.embedder.encode([question], is_query=True)[0]
        base, ext = self._scores(qvec, extra)
        all_scores = np.concatenate([base, ext])
        top = np.argsort(-all_scores)[:k]
        n = len(self.ids)
        out = []
        for j in top:
            if j < n:
                out.append(Passage(self.ids[j], self.texts[j], False, float(all_scores[j])))
            else:
                p = extra[j - n]
                out.append(Passage(p.pid, p.text, True, float(all_scores[j])))
        return out

    def kth_clean_score(self, question: str, k: int) -> float:
        """Score a passage must beat to enter the top-k of the clean corpus."""
        qvec = self.embedder.encode([question], is_query=True)[0]
        base = self.emb @ qvec
        return float(np.sort(base)[-k])

    def score_texts(self, question: str, texts: Sequence[str]) -> np.ndarray:
        qvec = self.embedder.encode([question], is_query=True)[0]
        return self.embedder.encode(list(texts)) @ qvec
