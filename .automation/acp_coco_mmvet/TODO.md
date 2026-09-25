# ACP COCO and MMVet completion run

- Models: Kimi-VL-A3B-Instruct, Qwen3-VL-30B-A3B-Instruct, InternVL3.5-30B-A3B-HF.
- Ratios: p=0.3 and p=0.5.
- Tasks: `coco2017_cap_val_local` and `mmvet` plus the local MMVet judge.
- Sampling: deterministic 50% subset, seed 42, minimum one sample.
- Inference batches: COCO/light-image batch 64 and MMVet/image batch 32. The
  already completed Kimi p=0.3 COCO result used batch 32; batch size does not
  change the evaluation subset or scoring protocol.
- ACP: `first_attr_coverage`, non-modality-aware, `gateup_act`, align/min 128,
  `largest_channel`; remaining pruning settings use `build_ep4_pruning_plan.py`
  defaults.
- Runtime: padded EP4 on four H20 GPUs. Results are isolated from the old
  removed historical output under `results/vllm_acp_align128_completion/`.
