#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dyf/code/distill/MAES
STATE="${ROOT}/.automation/kimi_method1_router"
cd "${ROOT}"
/home/dyf/miniconda/envs/vllm-maes/bin/python "${STATE}/campaign.py" validate-masks

tasks=(
    gqa textvqa_val coco2017_cap_val_local chartqa mmstar
    mmbench_en_dev_static_local mmvet mme realworldqa videomme
    longvideobench_val_v video_mmmu_local egoschema_subset mvbench_available_3800
)
for task in "${tasks[@]}"; do
    for ratio in p30 p50; do
        if /home/dyf/miniconda/envs/vllm-maes/bin/python "${STATE}/campaign.py" check --task "${task}" --ratio "${ratio}"; then
            echo "[skip] ${task} ${ratio}: validated artifact"
            continue
        fi
        printf '%s\n' "${task} ${ratio}" > "${STATE}/current.txt"
        echo "[run] $(date --iso-8601=seconds) ${task} ${ratio}"
        video_reader_env=()
        if [[ "${task}" == "video_mmmu_local" ]]; then
            video_reader_env=(FORCE_QWENVL_VIDEO_READER=opencv)
        fi
        env -u LIMIT -u FORCE \
            "${video_reader_env[@]}" \
            MASK_PLAN="${ROOT}/runtime/mask_plans/kimi-${ratio}-method1-router-direct.pt" \
            RUN_DIR="${ROOT}/results/vllm_ours/kimi/method1-router-direct/${ratio}-random-half-seed42" \
            RANDOM_SUBSET_FRACTION=0.5 RANDOM_SUBSET_MIN_SAMPLES=500 RANDOM_SUBSET_SEED=42 \
            BASELINE_LABEL="kimi_vl_method1_router_direct_${ratio}" \
            TASKS="${task}" FAIL_FAST=1 TASK_MAX_ATTEMPTS=2 \
            bash scripts/run_kimi_vl_vllm_mask_pruned.sh
        /home/dyf/miniconda/envs/vllm-maes/bin/python "${STATE}/campaign.py" check --task "${task}" --ratio "${ratio}"
        echo "[complete] $(date --iso-8601=seconds) ${task} ${ratio}"
    done
done

for ratio in p30 p50; do
    if /home/dyf/miniconda/envs/vllm-maes/bin/python -c \
        'import sys; sys.path.insert(0,".automation/kimi_method1_router"); from campaign import valid_judge; sys.exit(0 if valid_judge(sys.argv[1]) else 1)' "${ratio}"; then
        continue
    fi
    printf '%s\n' "judge ${ratio}" > "${STATE}/current.txt"
    echo "[judge] $(date --iso-8601=seconds) ${ratio}"
    PREDICTIONS_DIR="${ROOT}/results/vllm_ours/kimi/method1-router-direct/${ratio}-random-half-seed42" \
        TASKS=mmvet,mmbench,videommmu bash scripts/run_vllm_judge_stage.sh
done
printf '%s\n' complete > "${STATE}/current.txt"
/home/dyf/miniconda/envs/vllm-maes/bin/python "${STATE}/campaign.py" completion
