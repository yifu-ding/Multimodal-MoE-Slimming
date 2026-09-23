import types

import torch

from src.mistral4_vllm_compat import expand_mistral4_fused_weights, install_deepseek_loader_into


def test_expand_fused_expert_weights_and_scales():
    gate_up = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
    down = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    scale = torch.tensor([[[2.0]], [[3.0]]])
    expanded = dict(
        expand_mistral4_fused_weights(
            [
                ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
                ("model.layers.0.mlp.experts.down_proj", down),
                ("model.layers.0.mlp.experts.down_proj_scale_inv", scale),
            ]
        )
    )
    assert expanded["model.layers.0.mlp.experts.0.gate_proj.weight"].shape == (4, 3)
    assert expanded["model.layers.0.mlp.experts.0.up_proj.weight"].shape == (4, 3)
    assert expanded["model.layers.0.mlp.experts.1.down_proj.weight"].shape == (3, 4)
    assert expanded["model.layers.0.mlp.experts.1.down_proj.weight_scale_inv"] == 3


def test_dense_activation_scale_is_renamed_to_input_scale():
    result = list(
        expand_mistral4_fused_weights(
            [("model.layers.0.self_attn.o_proj.activation_scale", torch.tensor(2.0))]
        )
    )
    assert result[0][0].endswith("o_proj.input_scale")


class _FakeParam:
    def __init__(self, shape):
        self.data = torch.ones(shape, dtype=torch.float32)

    def weight_loader(
        self, param, loaded_weight, target_name, shard_id, expert_id, return_success=False
    ):
        if shard_id in ("w1", "w3"):
            self.data[expert_id][0 if shard_id == "w1" else 1] = loaded_weight
        elif shard_id == "w2":
            self.data[expert_id] = loaded_weight
        return True if return_success else None


def test_fused_expert_scale_falls_back_to_per_tensor_param_name():
    """Mistral4-119B ships weight_block_size=null, so vLLM's Fp8MoEMethod
    registers w13_weight_scale/w2_weight_scale (no "_inv" suffix) instead of
    the block-quant DeepSeek-v2 names. The loader must still find them."""
    module = types.SimpleNamespace()

    class DeepseekV2ForCausalLM:
        @staticmethod
        def load_weights(self, weights):
            return {name for name, _ in weights}

    module.DeepseekV2ForCausalLM = DeepseekV2ForCausalLM
    install_deepseek_loader_into(module)

    w13_scale = _FakeParam((2, 2))
    w2_scale = _FakeParam((2,))
    params_dict = {
        "model.layers.0.mlp.experts.w13_weight_scale": w13_scale,
        "model.layers.0.mlp.experts.w2_weight_scale": w2_scale,
    }

    class FakeModel:
        config = types.SimpleNamespace(model_type="mistral4")

        def named_parameters(self):
            return params_dict.items()

    weights = [
        (
            "model.layers.0.mlp.experts.gate_up_proj_scale_inv",
            torch.tensor([[[2.0]], [[3.0]]]),
        ),
        (
            "model.layers.0.mlp.experts.down_proj_scale_inv",
            torch.tensor([[[4.0]], [[5.0]]]),
        ),
    ]

    loaded = module.DeepseekV2ForCausalLM.load_weights(FakeModel(), weights)

    assert "model.layers.0.mlp.experts.w13_weight_scale" in loaded
    assert "model.layers.0.mlp.experts.w2_weight_scale" in loaded
    assert torch.equal(w13_scale.data, torch.tensor([[2.0, 2.0], [3.0, 3.0]]))
    assert torch.equal(w2_scale.data, torch.tensor([4.0, 5.0]))
