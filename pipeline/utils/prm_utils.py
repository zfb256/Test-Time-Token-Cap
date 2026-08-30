"""
PRM scoring utilities for CUPID-G Phase 1.

Loads Qwen2.5-Math-PRM-7B (or any step-reward model that outputs per-step
logits) and scores a reasoning chain.

Usage:
    scorer = PRMScorer(model_name="./models/Qwen2.5-Math-PRM-7B", device="cuda")
    result = scorer.score_chain(question="...", chain="step1\\n step2\\n ...")
    # result = {"step_scores": [...], "mean": 0.82, "min": 0.61, "var": 0.04}
"""

import numpy as np
from typing import List, Dict
import logging
import hashlib

logger = logging.getLogger(__name__)


class PRMScorer:
    """
    Wraps a Process Reward Model for step-level chain scoring.

    Supports two backends:
      1. "qwen_prm"   – Qwen2.5-Math-PRM-7B (token-level reward on \\n steps)
      2. "math_shepherd" – Math-Shepherd (similar protocol)
      3. "mock"       – returns random scores (for unit testing without GPU)
    """

    def __init__(
        self,
        model_name: str = "./models/Qwen2.5-Math-PRM-7B",
        device: str = "auto",
        dtype: str = "bfloat16",
        aggregation: str = "mean",   # mean | min | last | product
        mock: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.aggregation = aggregation
        self.mock = mock
        self._model = None
        self._tokenizer = None

        if aggregation not in {"mean", "min", "last", "product"}:
            raise ValueError(f"Unsupported PRM aggregation: {aggregation}")
        if not mock:
            self._load_model(dtype)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, dtype: str):
        """Load PRM model and tokenizer (lazy, called once)."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading PRM model: {self.model_name}")
        torch_dtype = getattr(torch, dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if self.device not in {None, "none"}:
            model_kwargs["device_map"] = self.device
        try:
            self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        except KeyError as exc:
            if self.device != "auto":
                raise
            logger.warning(
                "PRM device_map='auto' load failed (%s). Falling back to single-device load.",
                exc,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(target_device)
        self._model.eval()
        logger.info("PRM model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_chain(
        self,
        question: str,
        chain: str,
        step_delimiter: str = "auto",
    ) -> Dict:
        """
        Score a single reasoning chain.

        Returns:
            {
                "step_scores": [float, ...],   # per-step reward
                "mean": float,
                "min": float,
                "var": float,
                "chain_score": float,          # aggregated score
            }
        """
        if self.mock:
            return self._mock_score(chain, step_delimiter)

        steps = self._split_steps(chain, step_delimiter)
        if not steps:
            return self._empty_result()

        step_scores = self._score_steps_qwen(question, steps)
        return self._aggregate(step_scores)

    # ------------------------------------------------------------------
    # Qwen2.5-Math-PRM scoring protocol
    # ------------------------------------------------------------------

    def _score_steps_qwen(self, question: str, steps: List[str]) -> List[float]:
        """
        Qwen2.5-Math-PRM scores each step at '<extra_0>' separator tokens.
        The score for step i = P(positive label | separator after step i).
        """
        import torch

        messages = [
            {
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]
        conversation = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self._tokenizer(
            conversation,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            add_special_tokens=False,  # chat template already added special tokens
        )
        model_device = next(self._model.parameters()).device
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        step_sep_ids = self._tokenizer.encode("<extra_0>", add_special_tokens=False)
        if len(step_sep_ids) != 1:
            raise ValueError(
                f"Expected '<extra_0>' to be one token, got token ids: {step_sep_ids}"
            )
        step_sep_id = step_sep_ids[0]
        token_masks = input_ids == step_sep_id

        if not token_masks.any():
            return []
        n_sep_found = int(token_masks.sum().item())
        if n_sep_found != len(steps):
            logger.warning(
                "PRM separator count mismatch: found %d '<extra_0>' tokens for %d steps. "
                "The input may have been truncated; only matched separator scores will be used.",
                n_sep_found,
                len(steps),
            )

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            sep_probs = probabilities[0, token_masks[0], :]
            if sep_probs.shape[-1] < 2:
                raise ValueError(f"Expected PRM logits with >=2 classes, got shape {tuple(sep_probs.shape)}")
            positive_probs = sep_probs[:, 1]

        return positive_probs.detach().float().cpu().tolist()[:len(steps)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_steps(self, chain: str, delimiter: str) -> List[str]:
        """Split chain into steps; filter empty."""
        if delimiter != "auto":
            steps = [s.strip() for s in chain.split(delimiter)]
            return [s for s in steps if s]

        # Prefer paragraph boundaries; fall back to line boundaries and then
        # sentence-like chunks for compact generations.
        normalized = chain.replace("\r\n", "\n").strip()
        if not normalized:
            return []

        paragraphs = [s.strip() for s in normalized.split("\n\n") if s.strip()]
        if len(paragraphs) > 1:
            return paragraphs

        lines = [s.strip() for s in normalized.split("\n") if s.strip()]
        if len(lines) > 1:
            return lines

        sentence_parts = []
        current = []
        for token in normalized.split():
            current.append(token)
            if token.endswith((".", "?", "!", ";")) and len(current) >= 4:
                sentence_parts.append(" ".join(current))
                current = []
        if current:
            sentence_parts.append(" ".join(current))
        steps = sentence_parts if len(sentence_parts) > 1 else [normalized]
        return [s for s in steps if s]

    def _aggregate(self, step_scores: List[float]) -> Dict:
        if not step_scores:
            return self._empty_result()
        arr = np.array(step_scores, dtype=float)
        if self.aggregation == "mean":
            agg = float(arr.mean())
        elif self.aggregation == "min":
            agg = float(arr.min())
        elif self.aggregation == "last":
            agg = float(arr[-1])
        elif self.aggregation == "product":
            agg = float(np.prod(arr))
        else:
            agg = float(arr.mean())

        return {
            "step_scores": step_scores,
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "var": float(arr.var()),
            "chain_score": agg,
        }

    def _empty_result(self) -> Dict:
        return {"step_scores": [], "mean": 0.5, "min": 0.5, "var": 0.0, "chain_score": 0.5}

    def _mock_score(self, chain: str, delimiter: str) -> Dict:
        """Return deterministic mock scores (for unit tests)."""
        steps = self._split_steps(chain, delimiter)
        n = max(len(steps), 1)
        digest = hashlib.sha256(chain.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little") % (2**32)
        rng = np.random.default_rng(seed=seed)
        scores = list(rng.uniform(0.4, 0.9, size=n))
        return self._aggregate(scores)
