"""
ORM: Programmatic math answer verification (no model needed).

Supports:
  - GSM8K  : extract #### <number> from ground truth, then find last number in model output
  - MATH   : extract \\boxed{...} from both ground truth and model output
              and compare symbolically using sympy
"""

import atexit
import hashlib
import json
import os
import re
import signal
import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ---- optional sympy import (graceful fallback) ----
try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

try:
    from latex2sympy2 import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    LATEX2SYMPY_AVAILABLE = False


class MathVerifier:
    """
    Verify model-generated answers against ground truth.
    Returns True/False (correct / incorrect).
    """

    _CACHE_VERSION = "2"
    _MISSING = object()

    def __init__(
        self,
        numeric_tolerance: float = 1e-4,
        cache_path: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ):
        self.tol = float(numeric_tolerance)
        if not LATEX2SYMPY_AVAILABLE:
            raise RuntimeError(
                "latex2sympy2 is required for reproducible MATH/AIME scoring"
            )
        cache_path = cache_path or os.environ.get("MATH_VERIFY_CACHE")
        if timeout_s is None:
            timeout_s = float(os.environ.get("MATH_VERIFY_TIMEOUT", "0"))
        self.timeout_s = float(timeout_s)
        self._operation_timed_out = False
        self._cache = None
        if cache_path:
            path = Path(cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._cache = sqlite3.connect(
                path, timeout=60, isolation_level=None
            )
            self._cache.execute("PRAGMA journal_mode=WAL")
            self._cache.execute("PRAGMA synchronous=NORMAL")
            self._cache.execute(
                "CREATE TABLE IF NOT EXISTS scores "
                "(key BLOB PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._cache.execute(
                "CREATE TABLE IF NOT EXISTS timeouts "
                "(key BLOB PRIMARY KEY, operation TEXT NOT NULL)"
            )
            atexit.register(self.close)

    def close(self):
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def _cache_key(self, operation, *values):
        digest = hashlib.sha256()
        for value in (self._CACHE_VERSION, operation, self.tol, *values):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.digest()

    def _cache_get(self, key):
        if self._cache is None:
            return self._MISSING
        row = self._cache.execute(
            "SELECT value FROM scores WHERE key = ?", (key,)
        ).fetchone()
        return self._MISSING if row is None else json.loads(row[0])

    def _cache_put(self, key, value):
        if self._cache is None:
            return
        self._cache.execute(
            "INSERT OR REPLACE INTO scores(key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self._cache.execute("DELETE FROM timeouts WHERE key = ?", (key,))

    def _record_timeout(self, key, operation):
        if self._cache is not None:
            self._cache.execute(
                "INSERT OR IGNORE INTO timeouts(key, operation) VALUES (?, ?)",
                (key, operation),
            )

    @contextmanager
    def _time_limit(self):
        if self.timeout_s <= 0 or not hasattr(signal, "setitimer"):
            yield
            return

        def expired(_signum, _frame):
            raise TimeoutError("math verification timed out")

        previous_handler = signal.signal(signal.SIGALRM, expired)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, self.timeout_s)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
            signal.signal(signal.SIGALRM, previous_handler)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_gsm8k(self, prediction: str, ground_truth: str) -> bool:
        """
        GSM8K verification.
        Ground truth format: "... #### 42"
        Prediction: free-form text, extract last number.
        """
        gt_num = self._extract_gsm8k_answer(ground_truth)
        if gt_num is None:
            return False
        pred_num = self._extract_gsm8k_prediction(prediction)
        if pred_num is None:
            return False
        return abs(pred_num - gt_num) <= self.tol

    def verify_math(self, prediction: str, ground_truth: str) -> bool:
        """
        MATH dataset verification.
        Both ground truth and prediction should contain \\boxed{...}.
        Falls back to string matching if sympy fails.
        """
        gt_expr = self._extract_boxed(ground_truth)
        pred_expr = self._extract_boxed(prediction)

        if gt_expr is None or pred_expr is None:
            # Cannot extract \boxed{} from ground truth or prediction.
            # Do NOT fall back to full-solution string comparison — it produces
            # too many false positives/negatives and contaminates utility labels.
            return False

        if self._string_eq(gt_expr, pred_expr):
            return True
        if self._has_large_exponent_tower(gt_expr) or self._has_large_exponent_tower(
            pred_expr
        ):
            return False

        # Keep verification and plurality canonicalization consistent. This
        # also handles textual labels (for example ``odd`` vs
        # ``\text{odd}``) before latex2sympy2 can misparse them as products
        # of single-letter variables.
        try:
            canonical_equal = self._canonical_math_answer(
                gt_expr
            ) == self._canonical_math_answer(pred_expr)
        except TimeoutError:
            self._operation_timed_out = True
            canonical_equal = self._string_eq(gt_expr, pred_expr)
        if canonical_equal:
            return True

        # Try numeric comparison first
        gt_num = self._try_float(gt_expr)
        pred_num = self._try_float(pred_expr)
        if gt_num is not None and pred_num is not None:
            return abs(gt_num - pred_num) <= self.tol

        # Try sympy symbolic comparison
        if SYMPY_AVAILABLE:
            try:
                sympy_result = self._sympy_eq(gt_expr, pred_expr)
            except TimeoutError:
                self._operation_timed_out = True
                sympy_result = None
            if sympy_result is not None:
                return sympy_result

        # Final fallback: normalized string match
        return self._string_eq(gt_expr, pred_expr)

    def verify(self, prediction: str, ground_truth: str, dataset: str) -> bool:
        """Unified verify interface."""
        dataset = dataset.lower()
        key = self._cache_key("verify", prediction, ground_truth, dataset)
        cached = self._cache_get(key)
        if cached is not self._MISSING:
            return bool(cached)
        self._operation_timed_out = False
        if dataset == "gsm8k":
            result = self.verify_gsm8k(prediction, ground_truth)
        elif dataset in ("math", "aime"):
            # AIME answers are integers in boxed form; MATH-style boxed
            # extraction plus normalized numeric equality applies directly.
            result = self.verify_math(prediction, ground_truth)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        if self._operation_timed_out:
            self._record_timeout(key, "verify")
        else:
            self._cache_put(key, result)
        return result

    def extract_prediction_answer(self, prediction: str, dataset: str) -> Optional[str]:
        """Extract a normalized-ish answer string for consistency features."""
        dataset = dataset.lower()
        key = self._cache_key("extract", prediction, dataset)
        cached = self._cache_get(key)
        if cached is not self._MISSING:
            return cached
        self._operation_timed_out = False
        if dataset == "gsm8k":
            value = self._extract_gsm8k_prediction(prediction)
            result = f"{value:.10g}" if value is not None else None
        elif dataset in ("math", "aime"):
            answer = self._extract_boxed(prediction)
            try:
                result = (
                    (
                        self._normalize_string(answer)
                        if self._has_large_exponent_tower(answer)
                        else self._canonical_math_answer(answer)
                    )
                    if answer is not None
                    else None
                )
            except TimeoutError:
                self._operation_timed_out = True
                result = self._normalize_string(answer) if answer else None
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        if self._operation_timed_out:
            self._record_timeout(key, "extract")
        else:
            self._cache_put(key, result)
        return result

    # ------------------------------------------------------------------
    # GSM8K helpers
    # ------------------------------------------------------------------

    def _extract_gsm8k_answer(self, text: str) -> Optional[float]:
        """Extract the number after #### in GSM8K ground truth."""
        match = re.search(r'####\s*([\-\d,\.]+)', text)
        if match:
            return self._try_float(match.group(1).replace(',', ''))
        return None

    def _extract_last_number(self, text: str) -> Optional[float]:
        """Extract the last number appearing in model output."""
        # Match integers, floats, negatives, with optional comma separators
        numbers = re.findall(r'-?[\d,]+\.?\d*', text)
        for n in reversed(numbers):
            val = self._try_float(n.replace(',', ''))
            if val is not None:
                return val
        return None

    def _extract_gsm8k_prediction(self, text: str) -> Optional[float]:
        """Prefer the requested boxed answer, then fall back to the last number.

        Taking the last number unconditionally misreads outputs such as
        ``\\boxed{45} ... after 4 hours`` as 4 even though the model supplied an
        explicit final answer.
        """
        boxed = self._extract_boxed(text)
        if boxed is not None:
            boxed_num = self._try_numeric_expr(boxed)
            if boxed_num is None:
                boxed_num = self._extract_last_number(boxed)
            if boxed_num is not None:
                return boxed_num
        return self._extract_last_number(text)

    # ------------------------------------------------------------------
    # MATH helpers
    # ------------------------------------------------------------------

    @lru_cache(maxsize=8_192)
    def _extract_boxed(self, text: str) -> Optional[str]:
        """
        Extract the last content from \\boxed{...}, handling nested braces.
        Returns None if not found.
        """
        starts = [m.start() for m in re.finditer(r'\\boxed\{', text)]
        if not starts:
            # Try \boxed (without brace, rare)
            start = text.rfind(r'\boxed ')
            if start != -1:
                # Take next token
                rest = text[start + 7:].strip()
                return rest.split()[0] if rest else None
            return None

        for start in reversed(starts):
            # Walk forward counting braces
            idx = start + 7  # past \boxed{
            depth = 1
            while idx < len(text) and depth > 0:
                if text[idx] == '{':
                    depth += 1
                elif text[idx] == '}':
                    depth -= 1
                idx += 1

            if depth == 0:
                return text[start + 7: idx - 1].strip()

        return None

    @lru_cache(maxsize=8_192)
    def _sympy_eq(self, expr1: str, expr2: str) -> Optional[bool]:
        """
        Try to compare two LaTeX/sympy expressions symbolically.
        Returns None if parsing fails.
        """
        try:
            with self._time_limit():
                if LATEX2SYMPY_AVAILABLE:
                    s1 = latex2sympy(expr1)
                    s2 = latex2sympy(expr2)
                else:
                    s1 = parse_expr(expr1)
                    s2 = parse_expr(expr2)
                if s1 == s2:
                    return True
                if isinstance(s1, sympy.MatrixBase) or isinstance(
                    s2, sympy.MatrixBase
                ):
                    if not isinstance(s1, sympy.MatrixBase) or not isinstance(
                        s2, sympy.MatrixBase
                    ):
                        return False
                    if s1.shape != s2.shape:
                        return False
                    return all(
                        sympy.simplify(left - right) == 0
                        for left, right in zip(s1, s2)
                    )
                diff = sympy.simplify(s1 - s2)
                return diff == 0
        except TimeoutError:
            raise
        except Exception:
            return None

    @lru_cache(maxsize=8_192)
    def _canonical_math_answer(self, expr: str) -> str:
        """Canonicalize extracted math answers before consistency voting.

        Verification already treats many LaTeX-equivalent forms as equal, so
        voting on raw strings would incorrectly split votes such as ``1/2``
        and ``\\frac{1}{2}``.
        """
        # Textual class labels such as ``\text{odd}`` should vote together
        # with ``odd``. latex2sympy2 otherwise interprets the wrapper command
        # as symbolic multiplication (for example ``dd*o``).
        if re.fullmatch(
            r"\s*\\(?:text|mathrm|mathbf|mathit)\{[^{}]*\}\s*", expr
        ) or re.fullmatch(r"\s*[A-Za-z]{2,}\s*", expr):
            return self._normalize_string(expr)
        try:
            with self._time_limit():
                if LATEX2SYMPY_AVAILABLE:
                    parsed = latex2sympy(expr)
                elif SYMPY_AVAILABLE:
                    parsed = parse_expr(expr)
                else:
                    raise ValueError("symbolic parser unavailable")
                return str(sympy.simplify(parsed))
        except TimeoutError:
            raise
        except Exception:
            return self._normalize_string(expr)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _try_float(self, s: str) -> Optional[float]:
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def _try_numeric_expr(self, value: str) -> Optional[float]:
        """Parse a plain or LaTeX numeric expression into a finite float."""
        plain = self._try_float(value.replace(",", ""))
        if plain is not None:
            return plain
        if re.search(r"[A-Za-z]", re.sub(r"\\[A-Za-z]+", "", value)):
            return None
        try:
            with self._time_limit():
                if LATEX2SYMPY_AVAILABLE:
                    parsed = latex2sympy(value)
                elif SYMPY_AVAILABLE:
                    parsed = parse_expr(value)
                else:
                    return None
                if not bool(parsed.is_number):
                    return None
                numeric = float(parsed.evalf())
                return (
                    numeric
                    if numeric == numeric and abs(numeric) != float("inf")
                    else None
                )
        except TimeoutError:
            self._operation_timed_out = True
            return None
        except Exception:
            return None

    def _string_eq(self, a: str, b: str) -> bool:
        """Normalized string comparison."""
        return self._normalize_string(a) == self._normalize_string(b)

    @staticmethod
    def _has_large_exponent_tower(value: str) -> bool:
        exponents = [
            int(number)
            for number in re.findall(r"\^\{?\s*(\d+)", value)
        ]
        return value.count("^") >= 2 and any(
            exponent >= 100 for exponent in exponents
        )

    @staticmethod
    def _normalize_string(value: str) -> str:
        value = value.strip().lower()
        value = re.sub(
            r'\\(text|mathrm|mathbf|mathit|left|right)\{([^}]*)\}', r'\2', value
        )
        value = re.sub(r'\s+', '', value)
        return value.replace('$', '').replace('\\', '')
