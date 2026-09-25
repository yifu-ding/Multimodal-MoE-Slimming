#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "${REPO_ROOT}"
mkdir -p "${PLAN_DIR}" "${RESULT_ROOT}"

wait_for_gpus() {
    while ! bash "${AUTOMATION_DIR}/resource_check.sh"; do
        echo "[pipeline] $(date -Is) GPUs busy; waiting"
        sleep 30
    done
}

for name in "${CONFIGS[@]}"; do
    IFS='|' read -r model scores_tag model_tag ratio ratio_tag <<< "$(config_spec "${name}")"
    IFS='|' read -r scores plan run_dir <<< "$(config_paths "${name}")"
    if [[ ! -s "${plan}" ]]; then
        echo "[pipeline] $(date -Is) building ${name} plan"
        conda run --no-capture-output -n maes python scripts/build_ep4_pruning_plan.py \
            --scores "${scores}" --output "${plan}" --model "${model}" --prune-ratio "${ratio}" \
            --intra-method first_attr_coverage --modality-aware 0 \
            --intra-expert-metric gateup_act --align-inter 128 --min-per-expert 128 \
            --adjust-method largest_channel
    fi
done

for name in "${CONFIGS[@]}"; do
    IFS='|' read -r model scores_tag model_tag ratio ratio_tag <<< "$(config_spec "${name}")"
    IFS='|' read -r scores plan run_dir <<< "$(config_paths "${name}")"
    if [[ ! -f "${run_dir}/status/coco2017_cap_val_local.complete" || ! -f "${run_dir}/status/mmvet.complete" ]]; then
        wait_for_gpus
        echo "[pipeline] $(date -Is) inference ${name}"
        gpu_memory=0.90
        task_csv=coco2017_cap_val_local,mmvet
        [[ "${name}" == qwen30_* ]] && gpu_memory=0.75
        # Preserve the already completed batch-32 COCO marker for this config.
        [[ "${name}" == kimi_p30 ]] && task_csv=mmvet
        EP4_PLAN="${plan}" MODEL="${model}" MAES_EP4_STRATEGY=padded PRUNING_LABEL=acp_align128 \
        OUTPUT_ROOT="$(dirname "${run_dir}")" RUN_DIR="${run_dir}" \
        TASKS="${task_csv}" RANDOM_SUBSET_FRACTION=0.5 \
        RANDOM_SUBSET_MIN_SAMPLES=1 RANDOM_SUBSET_SEED=42 GPU_MEMORY_UTILIZATION="${gpu_memory}" \
        LIGHT_IMAGE_BATCH_SIZE=64 IMAGE_BATCH_SIZE=32 \
        ENABLE_QWEN3_NATIVE_VIDEO=0 TASK_MAX_ATTEMPTS=2 \
            bash scripts/run_vllm_ep4_pruned.sh
        python "${AUTOMATION_DIR}/update_report.py"
    fi
done

for name in "${CONFIGS[@]}"; do
    IFS='|' read -r scores plan run_dir <<< "$(config_paths "${name}")"
    if [[ ! -s "${run_dir}/local_judge/mmvet_summary.json" ]]; then
        wait_for_gpus
        echo "[pipeline] $(date -Is) MMVet judge ${name}"
        PREDICTIONS_DIR="${run_dir}" TASKS=mmvet JUDGE_MAX_ATTEMPTS=2 JUDGE_WORKERS=8 PORT=8010 \
            bash scripts/run_vllm_judge_stage.sh
        python "${AUTOMATION_DIR}/update_report.py"
    fi
done

python "${AUTOMATION_DIR}/update_report.py"
bash "${AUTOMATION_DIR}/completion_check.sh"
printf '\n> [!DONE]\n> **COMPLETE (%s CST)**\n> All 12 ACP COCO/MMVet results passed validation.\n' "$(date '+%F %H:%M')" >> "${STATUS}"
echo "ACP COCO/MMVet completion campaign finished."
