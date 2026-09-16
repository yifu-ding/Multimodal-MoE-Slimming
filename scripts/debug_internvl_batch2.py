"""Replay the failing frozen batch with batch=2 and report CUDA peaks."""
import hashlib
import json
import sys
import importlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration import collect_scores_main as collection


def main():
    layer = int(sys.argv[1])
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "storage/calibration_manifests/internvl3_5-30b-a3b-mixed-512.json"
    manifest, source_hash = collection._load_selection_manifest(str(source))
    manifest.pop("candidates", None)
    samples = manifest["samples"][104:112]
    assert len(samples) == 8
    for index, sample in enumerate(samples):
        sample["original_selection_rank"] = sample["selection_rank"]
        sample["selection_rank"] = index
    manifest.update(samples=samples, num_samples=8,
                    score_token_budget=sum(s["score_token_count"] for s in samples),
                    source_summary={}, debug_source_sha256=source_hash,
                    debug_original_range=[104, 112])
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    collection._load_selection_manifest(str(manifest_path))
    args_list = [
        "--model_name_or_path", "OpenGVLab/InternVL3_5-30B-A3B-HF",
        "--selection_manifest", str(manifest_path), "--output_dir", str(out),
        "--batch_size", "2", "--layers", str(layer),
        "--attn_implementation", "flash_attention_2", "--aggregation", "mean",
    ]
    if layer == 0:
        args_list += ["--hessian_probe_layer", "0"]
    args = collection.build_arg_parser().parse_args(args_list)
    backward_records = []
    original_backward = torch.Tensor.backward
    original_load = collection.load_model_bundle
    block_module = importlib.import_module("src.calibration.block_forward")
    original_collect = block_module.collect_scores_from_moe_module
    routing_checks = []

    def checked_collect(block, *pos, **kw):
        context = kw["_kwargs"]
        fields = {"saved_text_mask": "moe_text_mask", "saved_visual_mask": "moe_media_mask",
                  "saved_score_mask": "attn_mask"}
        totals = {field: 0 for field in fields}
        for expert in block.mlp.experts:
            saved_input = getattr(expert.down_proj, "saved_input", None)
            if saved_input is None:
                continue
            for field in fields:
                value = getattr(expert, field, None)
                assert isinstance(value, torch.Tensor), f"Missing {field}"
                assert value.numel() == saved_input.shape[0], f"Misaligned {field}"
                totals[field] += int(value.sum())
            weights = getattr(expert, "saved_router_weights", None)
            assert weights is not None and weights.numel() == saved_input.shape[0]
        for field, global_field in fields.items():
            expected = int(context[global_field].sum()) * block.mlp.top_k
            assert totals[field] == expected, (field, totals[field], expected)
        routing_checks.append(totals)
        print("[debug] routing_masks PASS " + json.dumps(totals), flush=True)
        return original_collect(block, *pos, **kw)

    def traced_load(*pos, **kw):
        bundle = original_load(*pos, **kw)
        block = collection.teacher_block(bundle, layer)
        actual = getattr(block.self_attn.config, "_attn_implementation", None)
        print(f"[debug] layer={layer} attention={actual}", flush=True)
        if actual != "flash_attention_2":
            raise RuntimeError(f"Unexpected decoder attention: {actual}")
        return bundle

    def traced_backward(tensor, *pos, **kw):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        result = original_backward(tensor, *pos, **kw)
        torch.cuda.synchronize()
        record = dict(index=len(backward_records), before_allocated=before,
                      after_allocated=torch.cuda.memory_allocated(),
                      peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved())
        backward_records.append(record)
        print("[debug] backward_pass " + json.dumps(record), flush=True)
        return result

    torch.Tensor.backward = traced_backward
    collection.load_model_bundle = traced_load
    block_module.collect_scores_from_moe_module = checked_collect
    torch.cuda.reset_peak_memory_stats()
    try:
        collection.run_collection(args)
    finally:
        torch.Tensor.backward = original_backward
        collection.load_model_bundle = original_load
        block_module.collect_scores_from_moe_module = original_collect
    assert len(backward_records) == 4, backward_records
    summary = dict(layer=layer, batch_size=2, samples=8, backward_passes=4,
                   source_sha256=source_hash, original_range=[104, 112],
                   score_token_budget=manifest["score_token_budget"],
                   attention="flash_attention_2", backward_records=backward_records,
                   routing_mask_checks=len(routing_checks), routing_masks=routing_checks,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                   total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
    (out / "memory.json").write_text(json.dumps(summary, indent=2))
    print("[debug] SUCCESS " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
