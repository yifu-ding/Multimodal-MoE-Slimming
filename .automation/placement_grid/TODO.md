# Experiment C placement grid TODO

- [x] Scan `m={4,6,8,12,16,24,32,48,64}` and `L={4,8,12,16,24,32,40,48}` at pruning ratios 30% and 50%.
- [x] Reuse configuration-identical E0/E0b artifacts; compute only missing cells.
- [x] Give every newly computed case four isolated physical CPU cores and a 300-second solver budget.
- [x] Record proven-optimal, retry, OOT, timing, and provenance for all 144 cells.
- [x] Generate the complete CSV, two EP-size tables, and time/attempt heatmaps.
- [x] Do not modify the paper or resume Qwen VideoMMMU.
