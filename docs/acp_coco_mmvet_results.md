# ACP COCO and MMVet completion results

Protocol: ACP `first_attr_coverage`, non-modality-aware, `gateup_act`, align/min=128, `largest_channel`; padded EP4 on 4 x H20; deterministic 50% subset with seed 42.

| Model | p | COCO CIDEr | COCO samples | MMVet judge score | MMVet samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kimi-VL-A3B-Instruct | 0.3 | 1.0151 | 2500 | 64.8624 | 109 |
| Qwen3-VL-30B-A3B-Instruct | 0.3 | 0.5034 | 2500 | 70.0000 | 109 |
| InternVL3.5-30B-A3B-HF | 0.3 | 1.2216 | 2500 | 70.3670 | 109 |
| Kimi-VL-A3B-Instruct | 0.5 | 1.0969 | 2500 | 37.8899 | 109 |
| Qwen3-VL-30B-A3B-Instruct | 0.5 | 0.5880 | 2500 | 61.9266 | 109 |
| InternVL3.5-30B-A3B-HF | 0.5 | 1.0843 | 2500 | 62.5688 | 109 |

Raw results: `/home/dyf/code/distill/MAES/results/vllm_acp_align128_completion`.
