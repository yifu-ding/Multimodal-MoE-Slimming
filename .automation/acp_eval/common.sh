#!/usr/bin/env bash
# Shared config for the ACP (first_attr_coverage, non-modality-aware, align=128)
# p=0.3 + p=0.5 eval pipeline scripts.
REPO_ROOT="/home/dyf/code/distill/MAES"

# video_mmmu_local and coco2017_cap_val_local are deferred to a later round
# (per user instruction); every other DEFAULT_TASKS entry runs now.
#
# NOTE: TASKS is what we pass in (the alias run_qwen3_vl_vllm_baseline.sh
# understands); TASK_LIST is what we use to check status/<name>.complete
# markers on disk. These differ for egoschema_subset: the script silently
# remaps it to "egoschema_subset_local" (a local parquet/media variant) and
# writes its completion marker under that remapped name. Using the alias in
# TASK_LIST would make count_complete permanently under-count by one and
# completion_check.sh would never report done even once every real task has
# finished.
#
# mmvet is a "deferred judge" task: run_qwen3_vl_vllm_baseline.sh runs it as
# --predict_only and DOES write status/mmvet.complete once the predict pass
# itself finishes (confirmed: results/.../status/mmvet.complete exists with
# no score fields, just like every other task's marker). That completion
# only means raw predictions were generated and cached; the real MMVet score
# still requires a separate `judge_vllm_predictions.py --predictions-dir
# <RUN_DIR>` pass afterwards (not tracked under status/ at all). Treat mmvet
# like any other TASK_LIST entry for "is this model's predict pass done";
# track the judge step separately once every model/ratio here is done.
# videomme now runs too (user: run it before coco/video_mmmu, which stay
# deferred). Qwen3-VL-30B's own baseline script defaults
# ENABLE_QWEN3_NATIVE_VIDEO=1 (Kimi/InternVL's wrappers force it to 0), which
# would remap videomme to "videomme_qwen3_vllm" only for that one model and
# break a shared TASK_LIST. pipeline.sh forces ENABLE_QWEN3_NATIVE_VIDEO=0 for
# every model so the completion marker is "videomme" everywhere.
TASKS="gqa,textvqa_val,chartqa,mmstar,mmbench_en_dev_static_local,mmvet,mme,realworldqa,videomme,longvideobench_val_v,egoschema_subset,mvbench_available_3800"
TASK_LIST=(gqa textvqa_val chartqa mmstar mmbench_en_dev_static_local mmvet mme realworldqa videomme longvideobench_val_v egoschema_subset_local mvbench_available_3800)

# name -> "MODEL_ID|PLAN_PATH|RUN_DIR"
model_spec() {
    case "$1" in
        kimi_p30)
            echo "moonshotai/Kimi-VL-A3B-Instruct|${REPO_ROOT}/runtime/ep4_plans/acp/kimi-vl-a3b-acp-p30.pt|${REPO_ROOT}/results/vllm_acp/kimi-vl-a3b/ep4-p30-padded/ep4-20260923-001917"
            ;;
        kimi_p50)
            echo "moonshotai/Kimi-VL-A3B-Instruct|${REPO_ROOT}/runtime/ep4_plans/acp/kimi-vl-a3b-acp-p50.pt|${REPO_ROOT}/results/vllm_acp/kimi-vl-a3b/ep4-p50-padded/run"
            ;;
        qwen3_p30)
            echo "Qwen/Qwen3-VL-30B-A3B-Instruct|${REPO_ROOT}/runtime/ep4_plans/acp/qwen3-vl-30b-a3b-acp-p30.pt|${REPO_ROOT}/results/vllm_acp/qwen3-vl-30b-a3b/ep4-p30-padded/run"
            ;;
        qwen3_p50)
            echo "Qwen/Qwen3-VL-30B-A3B-Instruct|${REPO_ROOT}/runtime/ep4_plans/acp/qwen3-vl-30b-a3b-acp-p50.pt|${REPO_ROOT}/results/vllm_acp/qwen3-vl-30b-a3b/ep4-p50-padded/run"
            ;;
        internvl_p30)
            echo "OpenGVLab/InternVL3_5-30B-A3B-HF|${REPO_ROOT}/runtime/ep4_plans/acp/internvl3_5-30b-a3b-acp-p30.pt|${REPO_ROOT}/results/vllm_acp/internvl3_5-30b-a3b/ep4-p30-padded/run"
            ;;
        internvl_p50)
            echo "OpenGVLab/InternVL3_5-30B-A3B-HF|${REPO_ROOT}/runtime/ep4_plans/acp/internvl3_5-30b-a3b-acp-p50.pt|${REPO_ROOT}/results/vllm_acp/internvl3_5-30b-a3b/ep4-p50-padded/run"
            ;;
        *)
            return 1
            ;;
    esac
}

# Order matters: finish p=0.3 across all 3 models before starting p=0.5
# (matches "these 3 models' p0.3 first, then p0.5" from the user).
MODEL_NAMES=(kimi_p30 qwen3_p30 internvl_p30 kimi_p50 qwen3_p50 internvl_p50)

# Number of TASK_LIST entries with a status/<task>.complete marker under $1 (a RUN_DIR).
count_complete() {
    local run_dir="$1" count=0 task
    for task in "${TASK_LIST[@]}"; do
        [[ -f "${run_dir}/status/${task}.complete" ]] && count=$((count + 1))
    done
    echo "${count}"
}
