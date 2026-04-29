import time
from typing import List, Optional, Tuple, Union

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from src.base.models.qwen3 import load_model
import src.base.models.qwen3 as models_qwen3
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.protocol import ChatMessages

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`"
    )
import lmms_eval.models
from lmms_eval.__main__ import cli_evaluate
import pickle

lmms_eval.models.AVAILABLE_CHAT_TEMPLATE_MODELS["qwen3_vl"] = "eval.qwen3.Qwen3_VL"


class Qwen3_VL(lmms):
    """lmms-eval model wrapper for Qwen3-VL-MoE with MoDES expert skipping support."""

    is_simple = False

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[
            float
        ] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[
            int
        ] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        # assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(
                f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}"
            )

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError(
                "max_image_size is only applicable if use_custom_video_loader is True"
            )

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
            "layer_gate_dict": kwargs.get("layer_gate_dict", None),
            "trust_remote_code": kwargs.get("trust_remote_code", True),
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        if batch_size == 1 or kwargs.get("tau_skip_path", None) is None:
            layer_gate_dict = {}
            config = AutoConfig.from_pretrained(pretrained)
            for i in range(config.text_config.num_hidden_layers):
                if (
                    i not in config.text_config.mlp_only_layers
                    and config.text_config.num_experts > 0
                    and (i + 1) % config.text_config.decoder_sparse_step == 0
                ):
                    layer_gate_dict[i] = {
                        "text": config.text_config.num_experts_per_tok,
                        "visual": config.text_config.num_experts_per_tok,
                    }

            if kwargs.get("topk", None) is not None:
                topk_text = kwargs.get(
                    "topk_text", config.text_config.num_experts_per_tok
                )
                topk_visual = kwargs.get(
                    "topk_visual", config.text_config.num_experts_per_tok
                )
                start = kwargs.get("start", 0)
                end = kwargs.get("end", -1)
                for i in range(start, end + 1):
                    if i in layer_gate_dict:
                        layer_gate_dict[i]["text"] = topk_text
                        layer_gate_dict[i]["visual"] = topk_visual

            model_kwargs["layer_gate_dict"] = layer_gate_dict

        layer_importance_path = kwargs.get("layer_importance_path", None)
        enable_load_layer_importance = layer_importance_path is not None
        tau_skip_path = kwargs.get("tau_skip_path", None)
        enable_tau_skip = tau_skip_path is not None
        tau = None
        self.tau_skip_path = tau_skip_path if tau_skip_path else ""
        if self.tau_skip_path != "":
            with open(tau_skip_path, "rb") as f:
                eval_logger.info(f"Loading tau skip dict from {tau_skip_path}")
                tau_skip_dict = pickle.load(f)
                tau = tau_skip_dict["tau"]
        self.tau = tau
        self.enable_load_layer_importance = enable_load_layer_importance
        self.enable_tau_skip = enable_tau_skip
        self.layer_importance_path = layer_importance_path

        model, processor = load_model(model_path=pretrained, **model_kwargs)

        # ── Optional structural pruning ──
        scores_path = kwargs.get("scores_path", None)
        prune_ratio = float(kwargs.get("prune_ratio", 0) or 0)
        if scores_path and prune_ratio > 0:
            from src.generate_mask import generate_masks
            from src.prune import apply_structural_pruning

            inter_method = kwargs.get("inter_method", "uniform")
            intra_method = kwargs.get("intra_method", "uniform")
            intra_expert_metric = kwargs.get("intra_expert_metric", "activation")
            modality_aware = bool(int(kwargs.get("modality_aware", 0)))
            normalize = bool(int(kwargs.get("normalize", 0)))
            smooth_fn = kwargs.get("smooth_fn", "sqrt")
            ema_source_key = kwargs.get("ema_source_key", "ema_matrix")
            align_inter = int(kwargs.get("align_inter", 0))
            min_per_expert = int(kwargs.get("min_per_expert", 0))
            thresholds_path = kwargs.get("thresholds_path", None)

            eval_logger.info(
                f"[Qwen3_VL] Generating masks: ratio={prune_ratio}, "
                f"inter={inter_method}, intra={intra_method}, metric={intra_expert_metric}"
            )
            mask_result = generate_masks(
                scores_dir=scores_path,
                prune_kwargs={
                    "prune_ratio": prune_ratio,
                    "thresholds_path": thresholds_path,
                    "mask_method_kwargs": {
                        "inter_layer_method": inter_method,
                        "intra_layer_method": intra_method,
                        "intra_expert_metric": intra_expert_metric,
                    },
                    "adjust_masks_kwargs": {
                        "align_inter": align_inter,
                        "min_per_expert": min_per_expert,
                    },
                    "modality_aware": modality_aware,
                    "normalize": normalize,
                    "ema_source_key": ema_source_key,
                    "prune_hidden": False,
                    "prune_gqa": False,
                    "smooth_fn": smooth_fn,
                },
                device="cpu",
                verbose=True,
            )
            mask_tensor = mask_result["intermediate_masks"]
            layers = [
                int(l)
                for l in mask_result.get(
                    "layers", list(range(mask_tensor.shape[0]))
                )
            ]
            masks = {
                layer_idx: mask_tensor[pos].detach().cpu().bool()
                for pos, layer_idx in enumerate(layers)
            }
            text_config = model.config.text_config
            apply_structural_pruning(model, masks, text_config)
            eval_logger.info(
                f"[Qwen3_VL] Structural pruning applied: ratio={prune_ratio}"
            )

        self._model = model.eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = processor
        self._tokenizer = processor.tokenizer
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [
                doc_to_messages[idx](self.task_dict[task][split][ids])
                for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))
            ]
            chat_messages: List[ChatMessages] = [
                ChatMessages(**{"messages": message}) for message in chat_messages
            ]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                video_kwargs["nframes"] = self.max_num_frames
            batched_messages = [
                chat_message.to_hf_messages(video_kwargs=video_kwargs)
                for chat_message in chat_messages
            ]
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batched_messages
            ]
            max_num_frames = self.max_num_frames
            try:
                image_inputs, video_inputs = process_vision_info(batched_messages)
            except Exception as e:
                import re

                # ValueError: nframes should in interval [\d+, \d+], but got \d+\.
                match = re.search(
                    r"nframes should in interval \[(\d+), (\d+)\], but got (\d+)\.",
                    str(e),
                )
                if match:
                    min_frames, max_frames, got_frames = map(int, match.groups())
                    frames = min(max_frames, got_frames)
                    frames = frames - (frames % 2)
                    eval_logger.warning(
                        f"Error processing vision info: {e}. "
                        f"Setting nframes to max_frames={frames}."
                    )
                    for chat_msg in batched_messages:
                        for msg in chat_msg:
                            for content in msg["content"]:
                                if content["type"] == "video":
                                    content["nframes"] = frames
                    max_num_frames = frames
                    image_inputs, video_inputs = process_vision_info(batched_messages)
                else:
                    raise e

            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
                enable_load_layer_importance=self.enable_load_layer_importance,
                enable_tau_skip=self.enable_tau_skip,
                tau=self.tau,
                layer_importance_path=self.layer_importance_path,
            )
            end_time = time.time()

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, texts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), clean_ans
                )
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")


if __name__ == "__main__":
    # Instantiate Qwen3_VL directly (bypasses model registry) and delegate to
    # lmms-eval's simple_evaluate so all task metrics are computed natively.
    import argparse as _ap
    from lmms_eval.utils import simple_parse_args_string
    from lmms_eval import evaluator

    _parser = _ap.ArgumentParser(description="Qwen3_VL prune + eval via lmms-eval")
    _parser.add_argument("--model", type=str, default="qwen3_vl")
    _parser.add_argument("--model_args", type=str, default="")
    _parser.add_argument("--tasks", type=str, required=True)
    _parser.add_argument("--batch_size", type=int, default=1)
    _parser.add_argument("--limit", type=int, default=None)
    _parser.add_argument("--offset", type=int, default=0)
    _parser.add_argument("--output_path", type=str, default=None)
    _parser.add_argument("--log_samples", action="store_true")
    _parser.add_argument("--gen_kwargs", type=str, default=None)
    _parser.add_argument("--verbosity", type=str, default="INFO")
    _args = _parser.parse_args()

    _model_kwargs = simple_parse_args_string(_args.model_args)
    _pretrained = _model_kwargs.pop("pretrained", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    _model_obj = Qwen3_VL(pretrained=_pretrained, batch_size=_args.batch_size, **_model_kwargs)

    from lmms_eval.evaluator import EvaluationTracker
    _tracker = EvaluationTracker(output_path=_args.output_path) if _args.output_path else None

    _results = evaluator.simple_evaluate(
        model=_model_obj,
        tasks=_args.tasks.split(","),
        batch_size=_args.batch_size,
        limit=_args.limit,
        offset=_args.offset,
        log_samples=_args.log_samples,
        evaluation_tracker=_tracker,
        gen_kwargs=_args.gen_kwargs,
        verbosity=_args.verbosity,
    )

    if _results is not None:
        from lmms_eval.utils import make_table
        print(make_table(_results))
        if "groups" in _results:
            print(make_table(_results, "groups"))
