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
# VideoMME uses Qwen3's native-video adapter for Qwen and the generic video
# path for Kimi/InternVL. The Qwen adapter writes videomme_qwen3_vllm.complete,
# so completion checks must resolve that model-specific marker without changing
# the experiment protocol.
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

native_video_for_model() {
    case "$1" in
        qwen3_*) echo 1 ;;
        *) echo 0 ;;
    esac
}

gpu_memory_utilization_for_model() {
    case "$1" in
        qwen3_*) echo 0.75 ;;
        *) echo 0.90 ;;
    esac
}

tasks_for_model() {
    case "$1" in
        qwen3_*) echo videomme ;;
        *) echo "${TASKS}" ;;
    esac
}

task_marker_for_model() {
    local name="$1" task="$2"
    if [[ "${name}" == qwen3_* && "${task}" == videomme ]]; then
        echo videomme_qwen3_vllm
    else
        echo "${task}"
    fi
}

# Number of TASK_LIST entries with a matching model-specific completion marker.
count_complete() {
    local name="$1" run_dir="$2" count=0 task marker
    for task in "${TASK_LIST[@]}"; do
        marker="$(task_marker_for_model "${name}" "${task}")"
        [[ -f "${run_dir}/status/${marker}.complete" ]] && count=$((count + 1))
    done
    echo "${count}"
}
