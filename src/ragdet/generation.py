"""Generator wrappers.

``HFGenerator`` is the real generator (GPU). ``ToyGenerator`` is a CPU test double
with the same interface, used only by unit tests and the toy smoke run.

Interface
---------
answer(question, passages)  -> str
analyze(question, passages) -> Analysis   (answer + generator-internal signals)
complete(prompt, max_new_tokens, temperature, n) -> list[str]   (used by attacks)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .core import Passage
from .signals import (attention_shares, char_spans_in_prompt, loo_shift,
                      token_spans_from_offsets)

SYSTEM_PROMPT = ("You answer questions using only the numbered context passages provided. "
                 "Reply with a short, direct answer. If the passages do not contain the "
                 "answer, reply exactly: I don't know.")


def format_context(passages: Sequence[Passage]) -> List[str]:
    return [p.text for p in passages]


def user_prompt(question: str, passages: Sequence[Passage]) -> str:
    ctx = "\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))
    return f"Context passages:\n{ctx}\n\nQuestion: {question}\nAnswer:"


@dataclass
class Analysis:
    answer: str
    mean_entropy: float          # mean token entropy of the generated answer (full context)
    mean_logprob: float          # mean token log-prob of the generated answer (full context)
    attn_share: List[float]      # per passage
    attn_density: List[float]    # per passage
    loo_shift: List[float]       # per passage: logp(answer|full) - logp(answer|without i)
    loo_ent_delta: List[float]   # per passage: entropy(without i) - entropy(full)


# --------------------------------------------------------------------------
class HFGenerator:
    """Instruction-tuned causal LM with attention and leave-one-out analysis.

    Attention needs ``attn_implementation='eager'`` (SDPA/flash kernels do not
    return attention weights), which is slower and memory hungry: keep the
    context short (top-k=5, ~1k tokens).
    """

    def __init__(self, hf_id: str, dtype: str = "bfloat16", max_new_tokens: int = 24,
                 load_4bit: bool = False, attn_layers: str = "last_half"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        kwargs = dict(attn_implementation="eager", device_map="auto")
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, dtype))
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs).eval()
        self.max_new_tokens = max_new_tokens
        self.attn_layers = attn_layers

    # -- prompt rendering --------------------------------------------------
    def render(self, question: str, passages: Sequence[Passage]) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, passages)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # -- generation ----------------------------------------------------------
    def _generate(self, text: str, max_new_tokens: int, temperature: float = 0.0, n: int = 1):
        torch = self.torch
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                          num_return_sequences=n, pad_token_id=self.tok.eos_token_id,
                          return_dict_in_generate=True, output_scores=True)
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=0.95)
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        seqs = out.sequences[:, enc["input_ids"].shape[1]:]
        texts = [self.tok.decode(s, skip_special_tokens=True).strip() for s in seqs]
        return texts, out, enc["input_ids"].shape[1]

    def complete(self, prompt: str, max_new_tokens: int = 80, temperature: float = 0.7,
                 n: int = 1) -> List[str]:
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)
        return self._generate(text, max_new_tokens, temperature, n)[0]

    def answer(self, question: str, passages: Sequence[Passage]) -> str:
        return self._generate(self.render(question, passages), self.max_new_tokens)[0][0]

    # -- scoring an answer under a context -----------------------------------
    def _answer_stats(self, prompt: str, answer: str, want_attention: bool = False,
                      spans_text: Sequence[str] = ()):
        """Teacher-forced pass over prompt+answer.

        Returns (sum_logprob, mean_logprob, mean_entropy, attn) where ``attn`` is
        None unless requested, else the per-passage (share, density).
        """
        torch = self.torch
        full = prompt + answer
        enc = self.tok(full, return_tensors="pt", add_special_tokens=False,
                       return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to(self.model.device)
        ans_pos = [i for i, (a, b) in enumerate(offsets) if a >= len(prompt) and b > a]
        if not ans_pos:
            return 0.0, 0.0, 0.0, None
        with torch.no_grad():
            out = self.model(input_ids=ids, output_attentions=want_attention)
        logits = out.logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        # token at position t is predicted by logits at t-1
        pred_pos = [p - 1 for p in ans_pos]
        tok_lp = logp[pred_pos, ids[0, ans_pos]]
        probs = logp[pred_pos].exp()
        ent = -(probs * logp[pred_pos]).sum(-1)
        attn = None
        if want_attention:
            L = len(out.attentions)
            layers = range(L // 2, L) if self.attn_layers == "last_half" else range(L)
            key_att = None
            for l in layers:
                a = out.attentions[l][0][:, pred_pos, :].float().mean(dim=(0, 1))  # over heads, queries
                key_att = a if key_att is None else key_att + a
            key_att = (key_att / len(layers)).cpu().numpy()
            cspans = char_spans_in_prompt(prompt, spans_text)
            tspans = token_spans_from_offsets(offsets, cspans)
            attn = attention_shares(key_att, tspans)
        return (float(tok_lp.sum()), float(tok_lp.mean()), float(ent.mean()), attn)

    def analyze(self, question: str, passages: Sequence[Passage]) -> Analysis:
        prompt = self.render(question, passages)
        ans = self._generate(prompt, self.max_new_tokens)[0][0]
        texts = [p.text for p in passages]
        lp_full, mlp, ment, attn = self._answer_stats(prompt, ans, True, texts)
        lp_wo, ent_wo = [], []
        for i in range(len(passages)):
            rest = [p for j, p in enumerate(passages) if j != i]
            lp_i, _, e_i, _ = self._answer_stats(self.render(question, rest), ans)
            lp_wo.append(lp_i)
            ent_wo.append(e_i)
        share, density = attn if attn is not None else (np.zeros(len(passages)),) * 2
        return Analysis(ans, ment, mlp, list(map(float, share)), list(map(float, density)),
                        list(map(float, loo_shift(lp_full, lp_wo))),
                        [float(e - ment) for e in ent_wo])


# --------------------------------------------------------------------------
class ToyGenerator:
    """CPU test double. Answers with the capital named in the passage that best
    overlaps the question; its 'signals' are simple deterministic functions of
    passage/question overlap. Exists only to exercise the pipeline plumbing."""

    _CAP = re.compile(r"([A-Z][a-z]+) is the capital(?: city)? of ([A-Z][a-z]+)")
    max_new_tokens = 24

    def _tokens(self, s):
        return set(re.findall(r"[a-z]+", s.lower()))

    def _overlaps(self, q, passages):
        qt = self._tokens(q)
        return np.array([len(qt & self._tokens(p.text)) / (1 + len(qt)) for p in passages])

    def answer(self, question, passages):
        if not passages:
            return "I don't know"
        ov = self._overlaps(question, passages)
        for i in np.argsort(-ov):
            m = self._CAP.search(passages[i].text)
            if m:
                return m.group(1)
        return "I don't know"

    def analyze(self, question, passages) -> Analysis:
        ans = self.answer(question, passages)
        ov = self._overlaps(question, passages)
        share = ov / ov.sum() if ov.sum() > 0 else np.zeros(len(passages))
        density = share * len(passages)
        loo = []
        for i in range(len(passages)):
            rest = [p for j, p in enumerate(passages) if j != i]
            loo.append(0.0 if self.answer(question, rest) == ans else 2.0)
        return Analysis(ans, 0.5, -0.5, share.tolist(), density.tolist(), loo, [0.0] * len(passages))

    def complete(self, prompt, max_new_tokens=80, temperature=0.7, n=1):
        return [f"Reference note {i}: " + prompt[-80:] for i in range(n)]
