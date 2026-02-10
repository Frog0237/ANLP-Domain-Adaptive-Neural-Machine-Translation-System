"""Unit tests for data loading and preprocessing."""

import unittest
from unittest.mock import MagicMock, patch


class TestPreprocessFunction(unittest.TestCase):
    def test_closure_returns_callable(self):
        from src.data import create_preprocess_fn

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [], "attention_mask": []}

        fn = create_preprocess_fn(mock_tokenizer)
        self.assertTrue(callable(fn))

    def test_extracts_correct_languages(self):
        from src.data import get_source_and_references

        # Mock dataset
        mock_ds = {
            "translation": [
                {"de": "Hallo Welt", "en": "Hello World"},
                {"de": "Guten Tag", "en": "Good day"},
            ]
        }

        sources, refs = get_source_and_references(mock_ds, "de", "en")
        self.assertEqual(sources, ["Hallo Welt", "Guten Tag"])
        self.assertEqual(refs, ["Hello World", "Good day"])

    def test_applies_source_prefix(self):
        from src.data import create_preprocess_fn

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [], "attention_mask": []}

        prefix = "translate German to English: "
        fn = create_preprocess_fn(mock_tokenizer, source_prefix=prefix)

        examples = {"translation": [{"de": "Hallo Welt", "en": "Hello World"}]}
        fn(examples)

        args, _ = mock_tokenizer.call_args
        self.assertTrue(args[0][0].startswith(prefix))


if __name__ == "__main__":
    unittest.main()

