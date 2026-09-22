# Kimi-VL method1-only router direct-mask campaign

- Wait until `.automation/kimi_direct/completion_check.sh` validates all 28 benchmark runs and both Judge stages.
- Evaluate `moonshotai/Kimi-VL-A3B-Instruct` at p30 and p50 on the same 14 benchmarks as the current Kimi direct-mask campaign.
- Keep `modality_aware=true` and replace the second-attribution intra-layer method with the Router's own output: `intra_layer_method=router`; do not use method2 or method3.
- Use the existing mixed-modality `gateup_act_text` and `gateup_act_visual` channel scores with `shared_protect=true`.
- Use direct 26-layer channel masks only: `align_inter=0`, `min_per_expert=0`, with no rounding, width tiers, or Rearrange.
- Run deterministic random half subsets with seed 42 and at least 500 samples (or all samples below 500), preserve completed artifacts, and run MMVet/MMBench/VideoMMMU Judge for both ratios.
