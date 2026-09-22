# Experiment C placement-grid partial snapshot

> This is an in-progress snapshot, not the final 144-cell result.

- Completed: 90/144 (62.5%)
- Proven optimal: 64
- Budget-ended/unproven: 26
- Reused E0/E0b artifacts: 12
- Median completed-cell wall time: 18.9487 s
- Maximum completed-cell wall time: 876.543 s
- The 300-second HiGHS limit excludes Python model construction and solver preprocessing; total wall time can exceed 300 seconds.
- E=128 is fixed. m is EP size/placement-group count; L is prefix depth.

## Available EP-size scan at L=48, p=30%

| m | Time (s) | Spread | Attempts | Status | Source |
| ---: | ---: | ---: | ---: | --- | --- |
| 4 | 0.368254 | 128 | 1 | proven optimal | reused |
| 6 | - | - | - | pending | - |
| 8 | - | - | - | pending | - |
| 12 | 11.6477 | 128 | 1 | proven optimal | computed |
| 16 | 35.056 | 128 | 1 | proven optimal | computed |
| 24 | - | - | - | pending | - |
| 32 | 300.305 | - | 1 | budget-ended/unproven | computed |
| 48 | 300.597 | - | 1 | budget-ended/unproven | computed |
| 64 | 876.543 | - | 1 | budget-ended/unproven | computed |

## Available EP-size scan at L=48, p=50%

| m | Time (s) | Spread | Attempts | Status | Source |
| ---: | ---: | ---: | ---: | --- | --- |
| 4 | - | - | - | pending | - |
| 6 | - | - | - | pending | - |
| 8 | - | - | - | pending | - |
| 12 | - | - | - | pending | - |
| 16 | - | - | - | pending | - |
| 24 | - | - | - | pending | - |
| 32 | - | - | - | pending | - |
| 48 | - | - | - | pending | - |
| 64 | - | - | - | pending | - |

## Files

- `partial-grid.csv`: completed cells only.
- `partial-heatmap-p30.png`
- `partial-heatmap-p50.png`
- `cases/`: validated per-cell JSON copied at snapshot time.
- `manifest.json`: full 144-cell experiment definition.
