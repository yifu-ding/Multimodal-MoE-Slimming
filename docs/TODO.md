# Current experiment TODO

Only fill missing cells in `docs/当前实验结果汇总.md`. Do not rerun a model, pruning ratio, task, or Judge result that is already present in the table or has a validated local artifact.

## Active pipeline: Qwen3-VL p = 0.3 video completion

Run directory: `results/vllm_ours/qwen3-vl-30b-a3b/ep4-p30-full`

- [ ] Complete Video-MME and replace the current missing value.
- [ ] Complete VideoMMMU direct evaluation and fill Adaptation, Comprehension, and Perception.
- [ ] Run the local VideoMMMU Judge from the completed predictions and fill VideoMMMU overall.
- [ ] Validate coverage and update `docs/当前实验结果汇总.md` with the final metrics.

The active dispatcher is `.automation/qwen_p30_video/pipeline.sh`. It must preserve existing response caches and skip every validated completed task.

## Completed or reusable: do not rerun

- Kimi-VL-A3B-Instruct baseline, p = 0.3, and p = 0.5 results in the summary table.
- InternVL3.5-30B-A3B-HF p = 0.3 and p = 0.5 results in the summary table.
- Qwen3-VL-30B-A3B-Instruct baseline and p = 0.5 results in the summary table.
- Qwen3-VL-30B-A3B-Instruct p = 0.3 non-video results already present in the summary table.
- Qwen baseline VideoMMMU Judge: reuse the validated 900/900 local artifact with score 63.3333; do not run the Judge again.

Before adding any future task, compare its model, pruning ratio, benchmark, Judge protocol, and scientific settings with the summary table and validated artifacts. Runtime-only settings such as GPU memory utilization do not define a new experiment and must reuse compatible cached responses.

## Queued after the active video pipeline: Experiment C supplement

- [ ] Run the five-case 60 s versus 300 s MILP pilot.
- [ ] If the pilot preserves solution quality, add stride-4 depth windows for Qwen, Kimi, and InternVL p = 0.3/p = 0.5 plans.
- [ ] Add p = 0.3 multi-model cases at m = 5, 7, 9, 10, and 11 while retaining the existing m grid.
- [ ] Compare `pairwise_swap` and `full_bijection` at m = 5, 6, and 7.
- [ ] Validate and summarize the new artifacts without rerunning existing Experiment C records.

Dispatcher: `.automation/placement_ablation_supplement/pipeline.sh`.
