# Experiment C E0 TODO

- [x] Generate and validate the exact 31-case manifest from the existing depth and m-sweep artifacts.
- [x] Run Arm A at 0.5/1/2/5/10/30/60 seconds for every case, stopping each ladder after the arithmetic floor is reached.
- [x] Run Arm B as a floor-bounded feasibility MILP with a 300-second budget; retry one quantum higher only after proven infeasibility.
- [x] Use four concurrent workers pinned to four disjoint sets of four physical CPU cores.
- [x] Generate and validate the JSON and Markdown summaries.
- [x] Do not resume Qwen VideoMMMU, run the obsolete 475-case supplement, or edit the paper.
