import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration.helpers.patches import patch_internvl_qwen3_moe_forward


class RoutingTest(unittest.TestCase):
    def test_native_outputs_gradients_and_routed_masks(self):
        for normalize in (True, False):
            torch.manual_seed(3)
            cfg = Qwen3MoeConfig(hidden_size=16, moe_intermediate_size=8,
                                num_experts=4, num_experts_per_tok=2, norm_topk_prob=normalize)
            original = Qwen3MoeSparseMoeBlock(cfg)
            patched = copy.deepcopy(original)
            patch_internvl_qwen3_moe_forward(SimpleNamespace(mlp=patched))
            masks = [torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),
                     torch.tensor([0, 1, 0, 1, 0, 0], dtype=torch.bool),
                     torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.bool)]
            for field, mask in zip(("moe_text_mask", "moe_media_mask", "moe_score_mask"), masks):
                setattr(patched, field, mask[:, None])
            x = torch.randn(2, 3, 16, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            ref, _ = original(x)
            actual, logits = patched(y)
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
            ref.square().sum().backward()
            actual.square().sum().backward()
            torch.testing.assert_close(x.grad, y.grad)
            for a, b in zip(original.parameters(), patched.parameters()):
                torch.testing.assert_close(a.grad, b.grad)
            weights, ids = logits.softmax(-1).topk(2, dim=-1)
            if normalize:
                weights = weights / weights.sum(-1, keepdim=True)
            for eid, expert in enumerate(patched.experts):
                slots, tokens = torch.where((ids == eid).T)
                if not tokens.numel():
                    self.assertIsNone(expert.saved_score_mask)
                    continue
                for field, mask in zip(("saved_text_mask", "saved_visual_mask", "saved_score_mask"), masks):
                    torch.testing.assert_close(getattr(expert, field), mask[tokens])
                torch.testing.assert_close(expert.saved_router_weights, weights[tokens, slots])
            for field, mask in zip(("saved_text_mask", "saved_visual_mask", "saved_score_mask"), masks):
                self.assertEqual(sum(int(getattr(e, field).sum()) for e in patched.experts if getattr(e, field) is not None), int(mask.sum()) * 2)

    def test_missing_masks_fail_before_collection(self):
        mlp = Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(hidden_size=16, moe_intermediate_size=8,
                                                 num_experts=4, num_experts_per_tok=2))
        patch_internvl_qwen3_moe_forward(SimpleNamespace(mlp=mlp))
        with self.assertRaisesRegex(RuntimeError, "aligned moe_text_mask"):
            mlp(torch.randn(1, 2, 16))


if __name__ == "__main__":
    unittest.main()
