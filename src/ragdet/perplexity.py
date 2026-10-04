"""Passage perplexity scorers."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence


class HFPerplexity:
    """Perplexity under a small causal LM (default GPT-2), the standard PoisonedRAG-style filter signal."""

    def __init__(self, hf_id: str = "gpt2", device: str | None = None, max_length: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(hf_id).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length

    def __call__(self, texts: Sequence[str], batch_size: int = 16) -> List[float]:
        torch = self.torch
        out: List[float] = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(list(texts[i:i + batch_size]), return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            shift_logits, shift_labels = logits[:, :-1], enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            nll = torch.nn.functional.cross_entropy(
                shift_logits.transpose(1, 2), shift_labels, reduction="none")
            mean_nll = (nll * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.extend(torch.exp(mean_nll).cpu().tolist())
        return out


class UnigramPerplexity:
    """Add-one unigram LM fit on a reference corpus. CPU test double only."""

    def __init__(self, reference_texts: Sequence[str]):
        self.counts = Counter(t for s in reference_texts for t in re.findall(r"[a-z0-9]+", s.lower()))
        self.total = sum(self.counts.values())
        self.vocab = len(self.counts) + 1

    def __call__(self, texts: Sequence[str], batch_size: int = 0) -> List[float]:
        out = []
        for s in texts:
            toks = re.findall(r"[a-z0-9]+", s.lower()) or ["<empty>"]
            nll = -sum(math.log((self.counts.get(t, 0) + 1) / (self.total + self.vocab)) for t in toks)
            out.append(math.exp(nll / len(toks)))
        return out
