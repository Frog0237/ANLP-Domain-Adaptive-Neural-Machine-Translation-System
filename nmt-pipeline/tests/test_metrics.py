"""Unit tests for translation quality metrics."""

import unittest
from src.metrics import (
    compute_bleu,
    compute_chrf,
    compute_length_ratio,
    compute_bigram_repetition,
    compute_all_metrics,
)


class TestBLEU(unittest.TestCase):
    def test_perfect_match(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        score = compute_bleu(preds, refs)
        self.assertGreater(score, 0.9)

    def test_completely_wrong(self):
        preds = ["asdf qwer zxcv"]
        refs = ["the cat sat on the mat"]
        score = compute_bleu(preds, refs)
        self.assertLess(score, 0.05)

    def test_empty_prediction(self):
        preds = [""]
        refs = ["the cat sat on the mat"]
        score = compute_bleu(preds, refs)
        self.assertEqual(score, 0.0)


class TestChrF(unittest.TestCase):
    def test_perfect_match(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        score = compute_chrf(preds, refs)
        self.assertGreater(score, 95.0)

    def test_partial_match(self):
        preds = ["the cat sat"]
        refs = ["the cat sat on the mat"]
        score = compute_chrf(preds, refs)
        self.assertGreater(score, 30.0)
        self.assertLess(score, 90.0)


class TestLengthRatio(unittest.TestCase):
    def test_equal_length(self):
        preds = ["a b c d"]
        refs = ["x y z w"]
        ratio = compute_length_ratio(preds, refs)
        self.assertAlmostEqual(ratio, 1.0)

    def test_under_translation(self):
        preds = ["a b"]
        refs = ["x y z w"]
        ratio = compute_length_ratio(preds, refs)
        self.assertAlmostEqual(ratio, 0.5)

    def test_over_generation(self):
        preds = ["a b c d e f"]
        refs = ["x y z"]
        ratio = compute_length_ratio(preds, refs)
        self.assertAlmostEqual(ratio, 2.0)


class TestBigramRepetition(unittest.TestCase):
    def test_no_repetition(self):
        preds = ["the cat sat on a mat"]
        refs = ["dummy"]
        rate = compute_bigram_repetition(preds, refs)
        self.assertEqual(rate, 0.0)

    def test_high_repetition(self):
        preds = ["the the the the the"]
        refs = ["dummy"]
        rate = compute_bigram_repetition(preds, refs)
        self.assertGreater(rate, 0.5)

    def test_single_word(self):
        preds = ["hello"]
        refs = ["dummy"]
        rate = compute_bigram_repetition(preds, refs)
        self.assertEqual(rate, 0.0)


class TestComputeAll(unittest.TestCase):
    def test_all_metrics_returned(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        results = compute_all_metrics(preds, refs)
        self.assertIn("bleu", results)
        self.assertIn("chrf", results)
        self.assertIn("length_ratio", results)
        self.assertIn("bigram_repetition", results)

    def test_subset_metrics(self):
        preds = ["hello world"]
        refs = ["hello world"]
        results = compute_all_metrics(preds, refs, metric_names=["bleu"])
        self.assertIn("bleu", results)
        self.assertNotIn("chrf", results)


if __name__ == "__main__":
    unittest.main()

