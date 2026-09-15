#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/dyf/code/distill/MAES"
REPORT="${REPO_ROOT}/docs/自动化执行结果.md"
AUTOMATION_LOG="${REPO_ROOT}/results/todo_scores.log"
KIMI_RUN="${REPO_ROOT}/results/vllm_baseline/kimi-vl-30b-a3b/ep4-full-all"
INTERNVL_RUN="${REPO_ROOT}/results/vllm_baseline/internvl3_5-30b-a3b-hf/ep4-full-all"
PLAN_ROWS=()
SCORE_MODELS="${SCORE_MODELS:-all}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
SCORE_SECOND_ORDER_CHUNK_SIZE="${SCORE_SECOND_ORDER_CHUNK_SIZE:-auto}"

append_record() {
    printf -- '- %s CST：%s\n' "$(date '+%F %H:%M')" "$1" >> "${REPORT}"
}

all_judges_ready() {
    local run_dir
    for run_dir in "${KIMI_RUN}" "${INTERNVL_RUN}"; do
        [[ -f "${run_dir}/run_summary.txt" ]] || return 1
        grep -qx 'exit_code=0' "${run_dir}/run_summary.txt" || return 1
        [[ -f "${run_dir}/local_judge/mmvet_summary.json" ]] || return 1
        [[ -f "${run_dir}/local_judge/mmbench_summary.json" ]] || return 1
        [[ -f "${run_dir}/local_judge/video_mmmu_summary.json" ]] || return 1
    done
}

run_logged() {
    local label="$1"
    shift
    append_record "开始 ${label}。"
    set +e
    "$@" 2>&1 | tee -a "${AUTOMATION_LOG}"
    local status=${PIPESTATUS[0]}
    set -e
    if (( status != 0 )); then
        append_record "${label} 失败，exit_code=${status}；详见 ${AUTOMATION_LOG}。跳过本项并继续下一模型。"
        return "${status}"
    fi
    append_record "${label} 完成。"
}

run_model_scores() {
    local artifact_tag="$1"
    local plan_tag="$2"
    local model="$3"
    local expected_layers="$4"
    local manifest="${REPO_ROOT}/storage/calibration_manifests/${artifact_tag}-mixed-512.json"
    local scores_dir="${REPO_ROOT}/storage/scores/${artifact_tag}-mixed-512"
    local score_status=complete

    if [[ ! -f "${manifest}" ]]; then
        if ! run_logged "${artifact_tag} 混合校准 manifest" \
            conda run --no-capture-output -n maes env \
                CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_MAP=balanced \
                HF_HOME=/home/data/dyf/hf_cache \
                HF_HUB_CACHE=/home/data/dyf/hf_cache/hub \
                HF_DATASETS_CACHE=/home/data/dyf/hf_cache/datasets \
                HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
                MPLCONFIGDIR=/tmp/maes-mpl \
                MODEL_PATH="${model}" \
                OUTPUT_MANIFEST="${manifest}" \
                CANDIDATE_POOL_SIZE=4096 NUM_SAMPLES=512 \
                TOTAL_SCORE_TOKENS=262144 MIN_SAMPLE_TOKENS=64 FEATURE_BATCH_SIZE=2 \
                bash scripts/prepare_mixed_calibration.sh; then
            printf '| %s | %s | %s | manifest failed |\n' "${model}" "${manifest}" "${scores_dir}/scores.pt" >> "${REPORT}"
            return 0
        fi
    else
        append_record "复用已有 ${artifact_tag} manifest：${manifest}（未覆盖）。"
    fi

    if [[ ! -f "${scores_dir}/scores.pt" ]]; then
        if ! run_logged "${artifact_tag} mixed scores（四卡按层并行）" \
            conda run --no-capture-output -n maes env \
                HF_HOME=/home/data/dyf/hf_cache \
                HF_HUB_CACHE=/home/data/dyf/hf_cache/hub \
                HF_DATASETS_CACHE=/home/data/dyf/hf_cache/datasets \
                HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
                MPLCONFIGDIR=/tmp/maes-mpl \
                MODEL_PATH="${model}" \
                SELECTION_MANIFEST="${manifest}" \
                OUTPUT_DIR="${scores_dir}" \
                GPUS=0,1,2,3 HESSIAN_PROBE_LAYER=0 \
                BATCH_SIZE="${SCORE_BATCH_SIZE}" \
                SECOND_ORDER_CHUNK_SIZE="${SCORE_SECOND_ORDER_CHUNK_SIZE}" \
                AGGREGATION=mean \
                bash scripts/run_collect_scores_4gpu.sh; then
            printf '| %s | %s | %s | scores failed |\n' "${model}" "${manifest}" "${scores_dir}/scores.pt" >> "${REPORT}"
            return 0
        fi
    else
        append_record "复用已有 ${artifact_tag} scores：${scores_dir}/scores.pt（未覆盖）。"
    fi

    if ! run_logged "${artifact_tag} scores 完整性校验" \
        conda run --no-capture-output -n maes \
            python scripts/validate_scores_artifact.py \
                --scores "${scores_dir}/scores.pt" \
                --expected-layers "${expected_layers}"; then
        printf '| %s | %s | %s | invalid; plans skipped |\n' "${model}" "${manifest}" "${scores_dir}/scores.pt" >> "${REPORT}"
        return 0
    fi

    local ratio ratio_tag plan_path
    for ratio in 0.3 0.5; do
        ratio_tag="p$(awk -v value="${ratio}" 'BEGIN { printf "%d", value * 100 }')"
        plan_path="${REPO_ROOT}/runtime/ep4_plans/${plan_tag}-${ratio_tag}-sparse-tier-v2.pt"
        if [[ ! -f "${plan_path}" ]]; then
            if ! run_logged "${plan_tag} ${ratio_tag} EP4 plan" \
                conda run --no-capture-output -n maes \
                    python scripts/build_ep4_pruning_plan.py \
                        --scores "${scores_dir}/scores.pt" \
                        --output "${plan_path}" \
                        --model "${model}" \
                        --prune-ratio "${ratio}"; then
                PLAN_ROWS+=("| ${model} | ${ratio} | ${plan_path} | failed; skipped |")
                score_status="valid; one or more plans failed"
                continue
            fi
        else
            append_record "复用已有 ${plan_tag} ${ratio_tag} EP4 plan：${plan_path}（未覆盖）。"
        fi
        PLAN_ROWS+=("| ${model} | ${ratio} | ${plan_path} | complete |")
    done
    printf '| %s | %s | %s | %s |\n' "${model}" "${manifest}" "${scores_dir}/scores.pt" "${score_status}" >> "${REPORT}"
}

model_selected() {
    local tag="$1"
    [[ "${SCORE_MODELS}" == "all" || ",${SCORE_MODELS}," == *",${tag},"* ]]
}

cd "${REPO_ROOT}"
append_record "scores 接力器已启动，等待 baseline/Judge/补跑接力器释放 GPU。"
while tmux has-session -t maes-todo-followup 2>/dev/null; do
    sleep 60
done
if all_judges_ready; then
    append_record "Kimi/InternVL baseline 与 Judge 产物完整，开始 scores。"
else
    append_record "部分 baseline/Judge 产物不完整；scores 生成与这些评测结果独立，继续执行三模型 scores。"
fi

cat >> "${REPORT}" <<'EOF'

### 三模型新生成 Scores

| Model | Manifest | Scores path | Status |
|---|---|---|---|
EOF

if model_selected "kimi"; then
    run_model_scores "kimi" "kimi" "moonshotai/Kimi-VL-A3B-Instruct" "1-26"
fi
if model_selected "qwen3"; then
    run_model_scores "qwen3" "qwen3-vl-30b-a3b" "Qwen/Qwen3-VL-30B-A3B-Instruct" "0-47"
fi
if model_selected "internvl3_5-30b-a3b"; then
    run_model_scores "internvl3_5-30b-a3b" "internvl3_5-30b-a3b" "OpenGVLab/InternVL3_5-30B-A3B-HF" "0-47"
fi
{
    printf '\n### EP4 Plans\n\n'
    printf '| Model | Prune ratio | Plan path | Status |\n'
    printf '|---|---:|---|---|\n'
    printf '%s\n' "${PLAN_ROWS[@]}"
} >> "${REPORT}"
append_record "三个模型的 scores/plan 阶段已逐项处理；失败项已记录并跳过。"
