from typing import Dict

import datetime as dt
import torch

from observations.common import resolve_model_name_or_path
from src.calibration.helpers.helpers import to_nested_expert_dict
from src.calibration.helpers.utils import is_fused_expert_container

from src.calibration.helpers.score_namespace import ACTIVE_CHANNEL_METRICS as CHANNEL_METRICS, EXPERT_METRICS

def _normalize_per_layer_counts(counts_map: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    output = {}
    for layer_idx, counts in counts_map.items():
        counts = counts.detach().cpu().float()
        denom = counts.sum().clamp_min(1.0)
        output[layer_idx] = counts / denom
    return output


class ScoreAccumulator:
    def __init__(
        self,
        layer_to_num_experts: Dict[int, int],
        layer_to_num_channels: Dict[int, int],
    ) -> None:
        self.layer_to_num_experts = layer_to_num_experts
        self.layer_to_num_channels = layer_to_num_channels
        self.layers = sorted(layer_to_num_experts.keys())
        self.channel_metrics: Dict[str, Dict[int, torch.Tensor]] = {}
        for metric in CHANNEL_METRICS:
            self.channel_metrics[metric] = {}
        self.expert_scores: Dict[str, Dict[int, torch.Tensor]] = {}
        for metric in EXPERT_METRICS:
            self.expert_scores[metric] = {}

        self.hit_counts: Dict[int, torch.Tensor] = {}
        self.layerwise_loss: Dict[int, float] = {}
        self.layerwise_second_order_sum: Dict[int, float] = {}

        for layer_idx in self.layers:
            e = layer_to_num_experts[layer_idx]
            i = layer_to_num_channels[layer_idx]
            for metric in CHANNEL_METRICS:
                self.channel_metrics[metric][layer_idx] = torch.zeros(e, i, dtype=torch.float32)
            for metric in EXPERT_METRICS:
                self.expert_scores[metric][layer_idx] = torch.zeros(e, dtype=torch.float32)
            self.hit_counts[layer_idx] = torch.zeros(e, dtype=torch.int64)

    def absorb_layer_scores(self, layer_idx: int, copied_block) -> None:
        expert_container = copied_block.mlp.experts
        if is_fused_expert_container(expert_container):
            num_experts = self.layer_to_num_experts[layer_idx]
            for metric in CHANNEL_METRICS:
                value = getattr(expert_container, metric, None)
                if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == num_experts:
                    self.channel_metrics[metric][layer_idx] = value.detach().cpu().float()
            for metric in EXPERT_METRICS:
                value = getattr(expert_container, metric, None)
                if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == num_experts:
                    self.expert_scores[metric][layer_idx] = value.detach().cpu().float().view(-1)
            activation = getattr(expert_container, "activation", None)
            if isinstance(activation, torch.Tensor) and activation.ndim >= 2 and activation.shape[0] == num_experts:
                self.hit_counts[layer_idx] = (activation.detach().cpu().float().abs().sum(dim=1) > 0).to(torch.int64)
            return

        experts = list(expert_container)
        for eid, expert in enumerate(experts):
            for metric in CHANNEL_METRICS:
                value = getattr(expert, metric, None)
                if value is None:
                    continue
                self.channel_metrics[metric][layer_idx][eid] = value.detach().cpu().float()
            for metric in EXPERT_METRICS:
                value = getattr(expert, metric, None)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    self.expert_scores[metric][layer_idx][eid] = value.detach().cpu().float().reshape(()).item()
                else:
                    self.expert_scores[metric][layer_idx][eid] = float(value)
            if getattr(expert, "activation", None) is not None:
                self.hit_counts[layer_idx][eid] = 1

    def build_scores_payload(self, args) -> dict:
        channel_scores = {
            metric: to_nested_expert_dict(self.channel_metrics[metric], scalar=False)
            for metric in CHANNEL_METRICS
        }
        expert_scores = {
            metric: to_nested_expert_dict(self.expert_scores[metric], scalar=True)
            for metric in EXPERT_METRICS
        }
        normalized_token_count_text = _normalize_per_layer_counts(
            {layer_idx: self.expert_scores["token_count_text"][layer_idx] for layer_idx in self.layers}
        )
        normalized_token_count_visual = _normalize_per_layer_counts(
            {layer_idx: self.expert_scores["token_count_visual"][layer_idx] for layer_idx in self.layers}
        )
        expert_scores["token_count_text"] = to_nested_expert_dict(
            normalized_token_count_text, scalar=True
        )
        expert_scores["token_count_visual"] = to_nested_expert_dict(
            normalized_token_count_visual, scalar=True
        )
        # Raw EMA matches observations/o1: compare per-modality normalized routing
        # frequencies directly, without an extra global prior correction.
        text_count_map = self.expert_scores["token_count_text"]      # {layer_idx: Tensor[E]}
        visual_count_map = self.expert_scores["token_count_visual"]  # {layer_idx: Tensor[E]}

        total_text = sum(
            float(layer_counts.float().sum().item())
            for layer_counts in text_count_map.values()
        )
        total_visual = sum(
            float(layer_counts.float().sum().item())
            for layer_counts in visual_count_map.values()
        )

        gamma = 1.0
        prior_ratio = (total_visual + 1e-8) / (total_text + 1e-8)

        ema_matrix = {}
        ema_matrix_prior_corrected = {}

        for layer_idx in self.layers:
            text_freq_tensor = text_count_map[layer_idx].float()      # shape [E]
            visual_freq_tensor = visual_count_map[layer_idx].float()  # shape [E]

            ema_tensor = (visual_freq_tensor - text_freq_tensor) / (
                visual_freq_tensor + text_freq_tensor + 1e-8
            )
            visual_freq_corr = visual_freq_tensor / (prior_ratio ** gamma)
            ema_prior_corrected_tensor = (visual_freq_corr - text_freq_tensor) / (
                visual_freq_corr + text_freq_tensor + 1e-8
            )

            ema_matrix[layer_idx] = {
                expert_id: float(ema_tensor[expert_id].item())
                for expert_id in range(ema_tensor.shape[0])
            }
            ema_matrix_prior_corrected[layer_idx] = {
                expert_id: float(ema_prior_corrected_tensor[expert_id].item())
                for expert_id in range(ema_prior_corrected_tensor.shape[0])
            }


        payload = {
            "channel_scores": channel_scores,
            "expert_scores": expert_scores,
            "ema_matrix": ema_matrix,
            "ema_matrix_prior_corrected": ema_matrix_prior_corrected,
            "layerwise_loss": dict(self.layerwise_loss),
            "layerwise_second_order_sum": dict(self.layerwise_second_order_sum),
            "metadata": {
                "loss_fn": args.loss_fn,
                "num_samples": args.num_samples,
                "selected_num_samples": getattr(args, "selected_num_samples", args.num_samples),
                "batch_size": args.batch_size,
                "dataset": args.dataset,
                "start_idx": args.start_idx,
                "model_name_or_path": args.model_name_or_path,
                "resolved_model_name_or_path": resolve_model_name_or_path(args.model_name_or_path),
                "subset_seed": args.subset_seed,
                "ema": args.ema,
                "fill_zero_for_unrouted": getattr(args, "fill_zero_for_unrouted", False),
                "layers": self.layers,
                "layer_to_num_experts": self.layer_to_num_experts,
                "layer_to_num_channels": self.layer_to_num_channels,
                "available_channel_metrics": list(CHANNEL_METRICS),
                "available_expert_metrics": list(EXPERT_METRICS),
                "ema_matrix_definition": "raw modality affinity: (visual_freq - text_freq) / (visual_freq + text_freq + 1e-8)",
                "ema_matrix_prior_corrected_definition": "prior-corrected modality affinity: ((visual_freq / prior_ratio) - text_freq) / ((visual_freq / prior_ratio) + text_freq + 1e-8), prior_ratio = total_visual / total_text",
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "created_at_unix": dt.datetime.now(dt.timezone.utc).timestamp(),
            },
        }
        return payload
