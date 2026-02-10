"""Unit tests for ALiBi utilities."""

import unittest

import torch

from src.alibi import build_alibi_bias, get_alibi_slopes, ZeroPositionalEmbedding


class TestAlibiUtils(unittest.TestCase):
    def test_slopes_count(self):
        slopes = get_alibi_slopes(6)
        self.assertEqual(len(slopes), 6)

    def test_bias_shape(self):
        slopes = get_alibi_slopes(4)
        bias = build_alibi_bias(
            slopes=slopes,
            tgt_len=3,
            src_len=5,
            device=torch.device("cpu"),
            dtype=torch.float32,
            is_causal=False,
        )
        self.assertEqual(bias.shape, (1, 4, 3, 5))

    def test_zero_positional_embedding(self):
        zpe = ZeroPositionalEmbedding(embed_dim=8)
        out = zpe(torch.Size([2, 4]))
        self.assertEqual(out.shape, (4, 8))
        self.assertTrue(torch.all(out == 0))


if __name__ == "__main__":
    unittest.main()

