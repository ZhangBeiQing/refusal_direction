import unittest

import torch
import torch.nn.functional as F

from pipeline.submodules.geometry_refusal import (
    _ce_loss_from_logits,
    _find_prompt_end_mask,
    _kl_loss_from_logits,
    orthonormalize_basis,
    sample_cone_coefficients,
)


class GeometryRefusalTests(unittest.TestCase):
    def test_sample_cone_coefficients_are_positive_unit_vectors(self):
        coefficients = sample_cone_coefficients(n_samples=16, cone_dim=4)

        self.assertTrue(torch.all(coefficients >= 0))
        self.assertTrue(torch.allclose(coefficients.norm(dim=-1), torch.ones(16), atol=1e-6))

    def test_orthonormalize_basis_returns_row_orthonormal_basis(self):
        basis = torch.randn(3, 12)
        orthonormal = orthonormalize_basis(basis)
        gram = orthonormal @ orthonormal.T

        self.assertTrue(torch.allclose(gram, torch.eye(3), atol=1e-5))

    def test_ce_loss_uses_shifted_target_mask(self):
        logits = torch.tensor(
            [
                [
                    [0.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )
        input_ids = torch.tensor([[0, 1, 0]])
        loss_mask = torch.tensor([[0, 1, 0]])

        expected = F.cross_entropy(logits[:, 0, :], torch.tensor([1]))
        actual = _ce_loss_from_logits(logits, input_ids, loss_mask)

        self.assertTrue(torch.allclose(actual, expected))

    def test_kl_loss_is_zero_for_identical_logits(self):
        logits = torch.randn(2, 4, 5)
        loss_mask = torch.tensor(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
            ]
        )

        actual = _kl_loss_from_logits(logits, logits.clone(), loss_mask)

        self.assertLess(abs(actual.item()), 1e-8)

    def test_prompt_mask_prefers_eoi_tokens(self):
        input_ids = torch.tensor([[9, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)
        eoi_toks = torch.tensor([2, 3])

        loss_mask = _find_prompt_end_mask(input_ids, attention_mask, eoi_toks, prompt_lengths=[4])

        self.assertEqual(loss_mask.tolist(), [[0, 0, 1, 1, 0]])

    def test_prompt_mask_falls_back_to_prompt_length(self):
        input_ids = torch.tensor([[0, 0, 10, 11, 12, 13]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
        eoi_toks = torch.tensor([99])

        loss_mask = _find_prompt_end_mask(input_ids, attention_mask, eoi_toks, prompt_lengths=[2])

        self.assertEqual(loss_mask.tolist(), [[0, 0, 0, 0, 1, 0]])


if __name__ == "__main__":
    unittest.main()
