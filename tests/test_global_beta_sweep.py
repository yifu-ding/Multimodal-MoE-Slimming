import copy
import unittest

import numpy as np
import torch

from src.calibration.collect_global_beta_sweep import BETAS, measure_batch, summarize


class AffineBlock(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("contributions", torch.tensor([[2., 0.], [-1.9, 0.], [0., 1.], [0., 0.]], dtype=torch.float64))

    def forward(self):
        return (self.alpha @ self.contributions).reshape(1, 1, 2)


class GlobalBetaSweepTest(unittest.TestCase):
    def measure(self, loss="l2"):
        alpha = torch.ones(4, dtype=torch.float64, requires_grad=True)
        batch = measure_batch(AffineBlock(alpha), alpha, (), {}, torch.ones(1, 1), loss, (0, 1, 3))
        return batch

    def test_independent_vector_partials_at_equal_values(self):
        batch = self.measure()
        self.assertEqual([m["beta_global"] for m in batch["measurements"]], list(BETAS))
        for item in batch["measurements"]:
            expected = (item["beta_global"] - 1) * torch.tensor([.2, -.19, 1., 0.], dtype=torch.float64)
            torch.testing.assert_close(item["gradient_sum"], expected)
        self.assertEqual(batch["measurements"][0]["gradient_sum"].numel(), 4)
        for item in batch["finite_differences"]:
            self.assertAlmostEqual(item["gradient_sum"], item["central_difference_sum"], places=11)

    def test_token_sum_then_abs_and_invalid_zero_denominator(self):
        batch = self.measure()
        second = copy.deepcopy(batch)
        second["num_score_tokens"] = 3
        for key in ("measurements", "repeats"):
            for item in second[key]:
                item["gradient_sum"] *= -3
                item["loss_sum"] *= 3
        for item in second["finite_differences"]:
            item["gradient_sum"] *= -3
            item["central_difference_sum"] *= -3
        reference = -.5 * batch["measurements"][4]["gradient_sum"].numpy()
        rows, _, _, validation, linearity = summarize([batch, second], reference, loss_fn="l2")
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["invalid_ratio_experts"], [3])
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[3]["ratio_to_global_zero_pct"], "")
        self.assertAlmostEqual(rows[0]["gradient_signed"], .1)
        self.assertAlmostEqual(rows[0]["gradient_abs"], .1)
        self.assertIsNone(linearity["per_expert_fits"][3]["r_squared"])
        self.assertEqual(linearity["experts_with_fit_residual_above_threshold"], [])
        for stat, beta in zip(linearity["ratio_statistics"], BETAS):
            self.assertAlmostEqual(stat["mean_pct"], 100 * (1-beta), places=10)

    def test_kl_nonlinearity_is_measured_not_filled_from_theory(self):
        batch = self.measure("kl_div")
        reference = batch["measurements"][4]["gradient_sum"].numpy()
        _, _, _, validation, linearity = summarize([batch], reference, loss_fn="kl_div")
        self.assertTrue(validation["passed"])
        self.assertGreater(linearity["max_abs_ratio_deviation_percentage_points"], .1)
        self.assertTrue(linearity["experts_with_fit_residual_above_threshold"])


if __name__ == "__main__":
    unittest.main()
