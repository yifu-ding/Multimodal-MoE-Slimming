# Experiment C EP-size/depth grid

- Cases: 144
- Proven optimal: 101/144
- Reused exact E0/E0b artifacts: 18
- Newly computed: 126
- Median solve time: 22.0276 s
- Maximum solve time: 876.543 s
- Each newly computed case used one isolated four-core CPU affinity set.
- E=128 experts per layer is fixed; m is the EP rank/placement-group count.

## EP-size scan at L=48, p=30%

| m | Time (s) | Spread | Attempts | Status | Source |
| ---: | ---: | ---: | ---: | --- | --- |
| 4 | 0.368254 | 128 | 1 | proven optimal | reused |
| 6 | 1.3839 | 128 | 1 | proven optimal | computed |
| 8 | 2.13199 | 128 | 1 | proven optimal | computed |
| 12 | 11.6477 | 128 | 1 | proven optimal | computed |
| 16 | 35.056 | 128 | 1 | proven optimal | computed |
| 24 | 300.254 | OOT | 1 | budget_or_unproven | computed |
| 32 | 300.305 | OOT | 1 | budget_or_unproven | computed |
| 48 | 300.597 | OOT | 1 | budget_or_unproven | computed |
| 64 | 876.543 | OOT | 1 | budget_or_unproven | computed |

## EP-size scan at L=48, p=50%

| m | Time (s) | Spread | Attempts | Status | Source |
| ---: | ---: | ---: | ---: | --- | --- |
| 4 | 0.333438 | 0 | 1 | proven optimal | reused |
| 6 | 0.704694 | 0 | 1 | proven optimal | computed |
| 8 | 0.700582 | 0 | 1 | proven optimal | computed |
| 12 | 2.98383 | 0 | 1 | proven optimal | reused |
| 16 | 1.30474 | 0 | 1 | proven optimal | reused |
| 24 | 0.634503 | 0 | 1 | proven optimal | computed |
| 32 | 1.18383 | 0 | 1 | proven optimal | reused |
| 48 | 83.2531 | 0 | 1 | proven optimal | computed |
| 64 | 110.746 | 0 | 1 | proven optimal | reused |

## Figures

- `heatmap-p30.png`
- `heatmap-p50.png`
- `grid.csv` contains all cells and provenance.
