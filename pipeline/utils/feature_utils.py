"""
Feature engineering for CUPID-G Phase 1.

Builds G1-G4 feature groups from raw inference + PRM data.

Feature groups:
  G1: Question surface features
  G2: PRM-only probe signals
  G3: Answer consistency (from 3 short chains)
  G4: All features combined

Utility label:
  helpful = 1  if  short_correct=0  AND  long_correct=1
  helpful = 0  otherwise
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:

    # ------------------------------------------------------------------
    # Utility label
    # ------------------------------------------------------------------

    @staticmethod
    def compute_utility_label(short_correct: bool, long_correct: bool) -> int:
        """
        helpful=1 if short was wrong but long was right.
        This is the core CUPID-G hypothesis signal.
        """
        return int((not short_correct) and long_correct)

    # ------------------------------------------------------------------
    # G1: Question surface features
    # ------------------------------------------------------------------

    @staticmethod
    def build_g1(question: str, dataset: str) -> Dict[str, float]:
        """
        G1 features derived purely from the question text.
        No inference needed.
        """
        text = question.strip()
        words = text.split()
        sentences = re.split(r'[.!?]+', text)

        # Number density: count of numeric tokens / total word count
        num_tokens = re.findall(r'\b\d+[\.,]?\d*\b', text)
        num_density = len(num_tokens) / max(len(words), 1)

        # Sub-question count: rough heuristic (questions containing "?")
        sub_qs = len(re.findall(r'\?', text))

        # Operator density: +, -, *, /, =, %, ^, sqrt, frac
        op_tokens = re.findall(r'[+\-*/=^%]|\\(?:sqrt|frac|cdot|times|div)', text)
        op_density = len(op_tokens) / max(len(words), 1)

        # Domain one-hot (GSM8K vs MATH)
        is_gsm8k = 1.0 if dataset.lower() == "gsm8k" else 0.0
        is_math = 1.0 if dataset.lower() == "math" else 0.0

        # MATH difficulty level (0 if not MATH)
        # Expected to be supplied externally; leave as 0 here
        # (caller can add "math_level" separately)

        return {
            "g1_question_len": float(len(words)),
            "g1_char_len": float(len(text)),
            "g1_num_density": float(num_density),
            "g1_op_density": float(op_density),
            "g1_sub_question_count": float(sub_qs),
            "g1_sentence_count": float(max(len([s for s in sentences if s.strip()]), 1)),
            "g1_is_gsm8k": is_gsm8k,
            "g1_is_math": is_math,
        }

    # ------------------------------------------------------------------
    # G2: PRM-only probe features
    # ------------------------------------------------------------------

    @staticmethod
    def build_g2(
        prm_scores: List[float],   # step-level PRM scores for the SHORT chain(s)
        short_correct: bool = False,  # kept for backward compatibility; not used as a feature
        prm_mean: Optional[float] = None,
        prm_min: Optional[float] = None,
        prm_var: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        G2 features: PRM statistics from cheap short-chain probes.

        Do not include short-chain ORM correctness here. In Phase 1 the ORM
        correctness is computed against the dataset answer and is valid only
        for label construction. Using it as an input feature leaks the label.
        """
        arr = np.array(prm_scores, dtype=float) if prm_scores else np.array([0.5])

        m = prm_mean if prm_mean is not None else float(arr.mean())
        mn = prm_min if prm_min is not None else float(arr.min())
        vr = prm_var if prm_var is not None else float(arr.var())

        # PRM high/low threshold: 0.5 (neutral boundary)
        prm_high = 1.0 if m >= 0.5 else 0.0
        step_count = float(len(arr))
        score_range = float(arr.max() - arr.min()) if len(arr) else 0.0

        return {
            "g2_prm_mean": float(m),
            "g2_prm_min": float(mn),
            "g2_prm_var": float(vr),
            "g2_prm_high": prm_high,
            "g2_prm_step_count": step_count,
            "g2_prm_score_range": score_range,
        }

    # ------------------------------------------------------------------
    # G3: Answer consistency (3 short chains)
    # ------------------------------------------------------------------

    @staticmethod
    def build_g3(answers: List[Optional[str]]) -> Dict[str, float]:
        """
        G3 features from the 3 short chains' extracted answers.

        answers: list of extracted answer strings (None if extraction failed)
        """
        valid = [a for a in answers if a is not None]
        n_valid = len(valid)
        n_total = len(answers)

        if n_valid == 0:
            return {
                "g3_agreement_rate": 0.0,
                "g3_n_unique_answers": 0.0,
                "g3_extraction_rate": 0.0,
                "g3_majority_confidence": 0.0,
            }

        # Count unique answers (normalized)
        def norm_ans(s: str) -> str:
            s = s.strip().lower()
            s = re.sub(r'\s+', '', s)
            s = s.replace(',', '')
            try:
                return f"{float(s):.10g}"
            except ValueError:
                pass
            return s

        normed = [norm_ans(a) for a in valid]
        from collections import Counter
        counter = Counter(normed)
        most_common_count = counter.most_common(1)[0][1]

        agreement_rate = most_common_count / n_valid
        n_unique = len(counter)
        extraction_rate = n_valid / max(n_total, 1)
        majority_confidence = most_common_count / max(n_total, 1)

        return {
            "g3_agreement_rate": float(agreement_rate),
            "g3_n_unique_answers": float(n_unique),
            "g3_extraction_rate": float(extraction_rate),
            "g3_majority_confidence": float(majority_confidence),
        }

    # ------------------------------------------------------------------
    # G4-only: model uncertainty from short-chain logprobs
    # ------------------------------------------------------------------

    @staticmethod
    def build_uncertainty(short_chains: List[Dict]) -> Dict[str, float]:
        avg_logprobs = [
            c.get("avg_logprob") for c in short_chains
            if c.get("avg_logprob") is not None
        ]
        margins = [
            c.get("avg_logprob_margin") for c in short_chains
            if c.get("avg_logprob_margin") is not None
        ]

        def mean_or_zero(vals: List[float]) -> float:
            return float(np.mean(vals)) if vals else 0.0

        def min_or_zero(vals: List[float]) -> float:
            return float(np.min(vals)) if vals else 0.0

        return {
            "g4_avg_logprob": mean_or_zero(avg_logprobs),
            "g4_min_logprob": min_or_zero(avg_logprobs),
            "g4_avg_logprob_margin": mean_or_zero(margins),
            "g4_min_logprob_margin": min_or_zero(margins),
            "g4_logprob_coverage": float(len(avg_logprobs) / max(len(short_chains), 1)),
        }

    # ------------------------------------------------------------------
    # Combine into feature groups
    # ------------------------------------------------------------------

    @classmethod
    def build_feature_groups(
        cls,
        g1: Dict, g2: Dict, g3: Dict,
        math_level: int = 0,
        uncertainty: Optional[Dict] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns dict with keys G1, G2, G3, G4 each mapping to feature dict.
        """
        g1_full = {**g1, "g1_math_level": float(math_level)}
        g2_full = dict(g2)
        g3_full = dict(g3)
        g4_full = {**g1_full, **g2_full, **g3_full, **(uncertainty or {})}

        return {
            "G1": g1_full,
            "G2": g2_full,
            "G3": g3_full,
            "G4": g4_full,
        }

    # ------------------------------------------------------------------
    # DataFrame helpers
    # ------------------------------------------------------------------

    @staticmethod
    def records_to_dataframe(
        records: List[Dict],
        feature_group: str,  # "G1" | "G2" | "G3" | "G4"
    ) -> Tuple["pd.DataFrame", "pd.Series"]:
        """
        Convert list of feature records to (X, y) for sklearn.

        Each record must have:
          - "features": {G1: {...}, G2: {...}, G3: {...}, G4: {...}}
          - "label": int (0 or 1)
          - "problem_id": str
        """
        rows = []
        labels = []
        ids = []
        expected_keys = None
        seen_ids = set()

        for r in records:
            problem_id = r.get("problem_id")
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError("every feature record needs a non-empty problem_id")
            if problem_id in seen_ids:
                raise ValueError(f"duplicate problem_id={problem_id}")
            seen_ids.add(problem_id)
            features = r.get("features")
            if not isinstance(features, dict) or feature_group not in features:
                raise ValueError(
                    f"problem_id={problem_id} lacks feature group {feature_group}"
                )
            feat = features[feature_group]
            if not isinstance(feat, dict) or not feat:
                raise ValueError(
                    f"problem_id={problem_id} has an empty or invalid "
                    f"feature group {feature_group}"
                )
            keys = set(feat)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(
                    f"problem_id={problem_id} has inconsistent {feature_group} "
                    "feature columns"
                )
            label = r.get("label")
            if label not in (0, 1) or isinstance(label, bool):
                raise ValueError(
                    f"problem_id={problem_id} label must be integer 0 or 1"
                )
            rows.append(feat)
            labels.append(label)
            ids.append(problem_id)

        df = pd.DataFrame(rows, index=ids)
        try:
            df = df.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{feature_group} features must be numeric"
            ) from error
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{feature_group} features contain missing or non-finite values"
            )
        y = pd.Series(labels, name="helpful")
        y.index = ids
        return df, y
