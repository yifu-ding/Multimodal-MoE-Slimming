from .helpers import compute_block_loss, set_block_modality_masks, teacher_blocks, to_nested_expert_dict
from .hooks import register_copied_block_hooks, register_teacher_block_hook
from .patches import patch_grad_enabled_kimi_moe_infer, patch_qwen_fused_experts_forward
from .utils import clear_fused_saved_tensors, is_fused_expert_container
from .utils import clear_block_saved_tensors, enable_input_grads, move_to_device_dtype, unwrap_output

__all__ = [
    "compute_block_loss",
    "set_block_modality_masks",
    "teacher_blocks",
    "register_copied_block_hooks",
    "register_teacher_block_hook",
    "patch_grad_enabled_kimi_moe_infer",
    "patch_qwen_fused_experts_forward",
    "clear_fused_saved_tensors",
    "is_fused_expert_container",
    "to_nested_expert_dict",
    "clear_block_saved_tensors",
    "enable_input_grads",
    "move_to_device_dtype",
    "unwrap_output",
]
