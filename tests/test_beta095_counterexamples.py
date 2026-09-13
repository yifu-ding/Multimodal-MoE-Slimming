import unittest

import torch

from scripts.analyze_beta095_counterexamples import analyze, representative_experts
from src.calibration.collect_beta095_counterexamples import measure_batch


class AffineBlock(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("contributions", torch.tensor([[2., 0.], [-1.9, 0.], [0., 1.]], dtype=torch.float64))

    def forward(self):
        return (self.alpha @ self.contributions).reshape(1, 1, 2)


class Beta095Test(unittest.TestCase):
    def test_real_probes_preserve_teacher_and_reset_entire_background(self):
        alpha = torch.ones(3, dtype=torch.float64, requires_grad=True)
        block = AffineBlock(alpha)
        data = measure_batch(block, alpha, (), {}, torch.ones(1, 1), loss_fn="l2", beta_work=.95)
        self.assertTrue(torch.equal(data["identity_gradient"], torch.zeros(3, dtype=torch.float64)))
        torch.testing.assert_close(data["removal"], torch.tensor([2., 1.805, .5], dtype=torch.float64))
        torch.testing.assert_close(data["gradient"], torch.tensor([-.01, .0095, -.05], dtype=torch.float64))
        torch.testing.assert_close(alpha, torch.full_like(alpha, .95))
        self.assertLess(abs(data["gradient"][0]), abs(data["gradient"][2]))
        self.assertGreater(data["removal"][0], data["removal"][2])
        for item in data["finite_differences"]:
            self.assertAlmostEqual(item["gradient_sum"], item["central_difference_sum"], places=10)

    def test_kl_identity_and_fd(self):
        alpha = torch.ones(3, dtype=torch.float64, requires_grad=True)
        data = measure_batch(AffineBlock(alpha), alpha, (), {}, torch.ones(1, 1), loss_fn="kl_div", beta_work=.95)
        self.assertLess(float(data["identity_gradient"].abs().max()), 1e-14)
        self.assertGreater(float(data["gradient"].abs().max()), 0)
        for item in data["finite_differences"]:
            self.assertLess(abs(item["gradient_sum"] - item["central_difference_sum"]), 2e-5)

    def rows(self):
        return [{"expert": e, "true_removal_at_identity": float(e + 1), "gradient_at_work": float(9-e),
                 "first_order_signed": -float(9-e), "first_order_abs": float(9-e),
                 "hessian_half_at_identity": float(e + 1)} for e in range(9)]

    def test_signed_abs_are_separate_and_hessian_must_match_values(self):
        rows = self.rows()
        summary, pairs = analyze(rows, repeat_loss_error=0, repeat_gradient_error=0, hessian_rtol=1e-4)
        self.assertEqual(summary["counterexample_counts"], {"first_order_signed": 0, "first_order_abs": 36})
        self.assertEqual(len(summary["representatives"]), 9)
        self.assertEqual(pairs[0]["expert_high"], 8)
        rows[8]["hessian_half_at_identity"] = 10.
        _, pairs = analyze(rows, repeat_loss_error=0, repeat_gradient_error=0, hessian_rtol=1e-4)
        self.assertFalse(any(p["expert_high"] == 8 for p in pairs))

    def test_truth_and_score_ties_are_not_correct_predictions(self):
        rows = self.rows()
        for row in rows:
            row["first_order_abs"] = 1.
        rows[1]["true_removal_at_identity"] = rows[0]["true_removal_at_identity"]
        summary, pairs = analyze(rows, repeat_loss_error=0, repeat_gradient_error=0, hessian_rtol=1e-4)
        metric = summary["metrics"]["first_order_abs"]
        self.assertIsNone(metric["spearman"])
        self.assertEqual(metric["truth_ties_excluded"], 1)
        self.assertEqual(metric["predicted_ties"], 35)
        self.assertEqual(pairs, [])

    def test_representatives_use_expert_id_tiebreak(self):
        rows = self.rows()[::-1]
        for row in rows:
            row["true_removal_at_identity"] = 1.
        self.assertEqual([r["expert"] for r in representative_experts(rows)], list(range(9)))


if __name__ == "__main__":
    unittest.main()
