import signal
import sqlite3
import sys
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from utils.math_verify import MathVerifier

# The symbolic-timeout guard is armed with signal.setitimer, which POSIX
# provides and Windows does not. Where it is absent the guard is a no-op, so
# these tests would not exercise what they name.
requires_setitimer = unittest.skipUnless(
    hasattr(signal, "setitimer"),
    "requires signal.setitimer (POSIX only)",
)


class MathVerifierTest(unittest.TestCase):
    def setUp(self):
        self.verifier = MathVerifier()

    def test_nested_box_is_extracted(self):
        self.assertEqual(
            self.verifier._extract_boxed(r"answer: \boxed{\frac{1}{2}}"),
            r"\frac{1}{2}",
        )

    def test_equivalent_fraction_answers_share_vote_key(self):
        plain = self.verifier.extract_prediction_answer(r"\boxed{1/2}", "math")
        latex = self.verifier.extract_prediction_answer(
            r"\boxed{\frac{1}{2}}", "math"
        )
        self.assertEqual(plain, latex)

    def test_repeated_symbolic_answers_use_cache(self):
        answer = r"\frac{1234567}{7654321}"
        self.verifier._canonical_math_answer.cache_clear()
        self.verifier._canonical_math_answer(answer)
        self.verifier._canonical_math_answer(answer)
        self.assertEqual(
            self.verifier._canonical_math_answer.cache_info().hits, 1
        )

    def test_disk_cache_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.sqlite"
            first = MathVerifier(cache_path=str(path))
            self.assertTrue(
                first.verify(r"\boxed{x+1}", r"\boxed{1+x}", "math")
            )
            first.close()

            second = MathVerifier(cache_path=str(path))
            second.verify_math = lambda *_: self.fail("cache was not reused")
            self.assertTrue(
                second.verify(r"\boxed{x+1}", r"\boxed{1+x}", "math")
            )
            second.close()

    @requires_setitimer
    def test_symbolic_timeout_is_recorded_and_next_answer_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.sqlite"
            verifier = MathVerifier(cache_path=str(path), timeout_s=0.01)
            with mock.patch(
                "utils.math_verify.latex2sympy",
                side_effect=lambda _: time.sleep(1),
            ):
                self.assertFalse(
                    verifier.verify(r"\boxed{x}", r"\boxed{y}", "math")
                )
            self.assertTrue(verifier.verify(r"\boxed{2}", "#### 2", "gsm8k"))
            verifier.close()

            connection = sqlite3.connect(path)
            timeout_count = connection.execute(
                "SELECT COUNT(*) FROM timeouts"
            ).fetchone()[0]
            connection.close()
            self.assertEqual(timeout_count, 1)

    @requires_setitimer
    def test_two_disk_cache_connections_do_not_lock_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.sqlite"
            first = MathVerifier(cache_path=str(path))
            second = MathVerifier(cache_path=str(path), timeout_s=0.01)
            self.assertTrue(first.verify(r"\boxed{2}", "#### 2", "gsm8k"))
            with mock.patch(
                "utils.math_verify.latex2sympy",
                side_effect=lambda _: time.sleep(1),
            ):
                self.assertFalse(
                    second.verify(r"\boxed{x}", r"\boxed{y}", "math")
                )
            first.close()
            second.close()

    def test_large_exponent_tower_skips_symbolic_parser(self):
        prediction = r"\boxed{1775^{2^{2002}} \cdot 1775 + 1}"
        with mock.patch(
            "utils.math_verify.latex2sympy",
            side_effect=AssertionError("huge tower reached symbolic parser"),
        ):
            self.assertFalse(
                self.verifier.verify(prediction, r"\boxed{1/2}", "math")
            )
            self.assertEqual(
                self.verifier.extract_prediction_answer(prediction, "math"),
                "1775^{2^{2002}}cdot1775+1",
            )

    @requires_setitimer
    def test_successful_retry_clears_timeout_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.sqlite"
            verifier = MathVerifier(cache_path=str(path), timeout_s=0.01)
            with mock.patch(
                "utils.math_verify.latex2sympy",
                side_effect=lambda _: time.sleep(1),
            ):
                verifier.verify(r"\boxed{x}", r"\boxed{y}", "math")
            verifier.timeout_s = 1
            verifier.verify(r"\boxed{x}", r"\boxed{y}", "math")
            verifier.close()

            connection = sqlite3.connect(path)
            timeout_count = connection.execute(
                "SELECT COUNT(*) FROM timeouts"
            ).fetchone()[0]
            connection.close()
            self.assertEqual(timeout_count, 0)

    def test_equivalent_matrix_answers_verify(self):
        ground_truth = (
            r"The answer is \boxed{\begin{pmatrix} "
            r"\frac{1}{5} \\ -\frac{3}{5} \end{pmatrix}}."
        )
        prediction = (
            r"\boxed{\begin{pmatrix} "
            r"\dfrac{1}{5} \\ -\dfrac{3}{5} \end{pmatrix}}"
        )
        self.assertTrue(self.verifier.verify_math(prediction, ground_truth))

    def test_text_wrapper_and_plain_label_share_vote_key(self):
        wrapped = self.verifier.extract_prediction_answer(
            r"\boxed{\text{odd}}", "math"
        )
        plain = self.verifier.extract_prediction_answer(r"\boxed{odd}", "math")
        self.assertEqual(wrapped, plain)

    def test_aime_integer_verification(self):
        self.assertTrue(
            self.verifier.verify(r"Therefore \boxed{023}.", r"\boxed{23}", "aime")
        )

    def test_gsm8k_prefers_boxed_answer_over_trailing_number(self):
        prediction = r"The distance is \boxed{45} miles after 4 hours."
        ground_truth = "work #### 45"
        self.assertTrue(self.verifier.verify(prediction, ground_truth, "gsm8k"))
        self.assertEqual(
            self.verifier.extract_prediction_answer(prediction, "gsm8k"), "45"
        )

    def test_gsm8k_boxed_fraction_is_not_misread_as_denominator(self):
        prediction = r"The answer is \boxed{\frac{1}{2}}."
        self.assertTrue(self.verifier.verify(prediction, "work #### 0.5", "gsm8k"))

    def test_gsm8k_units_do_not_poison_symbolic_parser(self):
        self.assertTrue(
            self.verifier.verify(r"\boxed{17 pounds}", "work #### 17", "gsm8k")
        )
        self.assertTrue(
            self.verifier.verify(
                r"\boxed{7x(x^2-3x+2)}",
                r"\boxed{7x(x-1)(x-2)}",
                "math",
            )
        )

    def test_reproducible_scoring_requires_latex_parser(self):
        import utils.math_verify as module

        available = module.LATEX2SYMPY_AVAILABLE
        module.LATEX2SYMPY_AVAILABLE = False
        try:
            with self.assertRaisesRegex(RuntimeError, "reproducible"):
                module.MathVerifier()
        finally:
            module.LATEX2SYMPY_AVAILABLE = available


if __name__ == "__main__":
    unittest.main()
