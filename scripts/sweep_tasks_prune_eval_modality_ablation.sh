#!/usr/bin/env bash
# Sweep TASK × INTER_METHOD × INTRA_METHOD × MODALITY_AWARE × SHARED_PROTECT × INTRA_EXPERT_METRIC
# with modality ablation flags passed via command line:
#   bash scripts/sweep_tasks_prune_eval_modality_ablation.sh 1 0   # text_only
#   bash scripts/sweep_tasks_prune_eval_modality_ablation.sh 0 1   # visual_only
#
# Appends one markdown table row per run to a per-batch summary.md under:
#   results/prune_eval_p50/sweep_tasks-<MODEL_NAME>-<SUFFIX>/

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <text_only:0|1> <visual_only:0|1>" >&2
  exit 1
fi

TEXT_ONLY="$1"
VISUAL_ONLY="$2"

if [[ "${TEXT_ONLY}" != "0" && "${TEXT_ONLY}" != "1" ]]; then
  echo "error: text_only must be 0 or 1; got ${TEXT_ONLY}" >&2
  exit 1
fi

if [[ "${VISUAL_ONLY}" != "0" && "${VISUAL_ONLY}" != "1" ]]; then
  echo "error: visual_only must be 0 or 1; got ${VISUAL_ONLY}" >&2
  exit 1
fi

if [[ "${TEXT_ONLY}" == "1" && "${VISUAL_ONLY}" == "1" ]]; then
  echo "error: text_only and visual_only cannot both be 1" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PREFIX="${PREFIX:-${REPO_ROOT}}"
export PYTHONPATH="${PREFIX}"
PRUNE_RATIO="${PRUNE_RATIO:-0.5}"
export SCORES_PATH="${SCORES_PATH:-}"

NUM_SAMPLES="${NUM_SAMPLES:-0}"
USE_LMMS_EVAL=${USE_LMMS_EVAL:-0}

SWEEP_TASKS="${SWEEP_TASKS:-chartqa coco2017cap mmstar mmbench realworldqa gqa mme textvqa}"
SWEEP_INTER_METHODS="${SWEEP_INTER_METHODS:-uniform}"
SWEEP_INTRA_METHODS="${SWEEP_INTRA_METHODS:-second_attr_fillzero_coverage}"

SWEEP_MODALITY_AWARE="${SWEEP_MODALITY_AWARE:-1}"
SWEEP_SHARED_PROTECT="${SWEEP_SHARED_PROTECT:-1}"
EXPERTWISE_BUDGET_NORMALIZE="${EXPERTWISE_BUDGET_NORMALIZE:-1}"
USE_EMA="${USE_EMA:-1}"

NORMALIZE="${NORMALIZE:-0}"
EMA_SOURCE_KEY="${EMA_SOURCE_KEY:-ema_matrix_prior_corrected}"
LAYERWISE_LOSS_KEY="${LAYERWISE_LOSS_KEY:-layerwise_loss}"
SMOOTH_FN="${SMOOTH_FN:-cbrt}"
SWEEP_INTRA_EXPERT_METRICS="${SWEEP_INTRA_EXPERT_METRICS:-gateup_act}"

SWEEP_TS="${SWEEP_TS:-$(date +%m%d%H%M)}"
MODEL_NAME="${MODEL_NAME:-kimi-vl-a3b}"
case "${MODEL_NAME}" in
  deepseek-vl2-small) ;;
  kimi-vl-a3b-instruct) MODEL_NAME="kimi-vl-a3b" ;;
  kimi-vl-a3b) ;;
  qwen3-vl-30b-a3b-instruct) MODEL_NAME="qwen3-vl-30b-a3b" ;;
  qwen3-vl-30b-a3b) ;;
  internvl3_5-30b-a3b-hf) ;;
  gemma-4-26b-a4b) ;;
  qwen3.5-35b-a3b) ;;
  *)
    echo "error: MODEL_NAME must be one of: deepseek-vl2-small kimi-vl-a3b kimi-vl-a3b-instruct qwen3-vl-30b-a3b qwen3-vl-30b-a3b-instruct internvl3_5-30b-a3b-hf gemma-4-26b-a4b qwen3.5-35b-a3b. Got: ${MODEL_NAME}" >&2
    exit 1
    ;;
esac

if [[ -n "${SWEEP_MODEL_PATH:-}" ]]; then
  export MODEL_PATH="${SWEEP_MODEL_PATH}"
else
  case "${MODEL_NAME}" in
    deepseek-vl2-small) export MODEL_PATH="deepseek-ai/deepseek-vl2-small" ;;
    kimi-vl-a3b) export MODEL_PATH="moonshotai/Kimi-VL-A3B-Instruct" ;;
    qwen3-vl-30b-a3b) export MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Instruct" ;;
    internvl3_5-30b-a3b-hf) export MODEL_PATH="OpenGVLab/InternVL3_5-30B-A3B-HF" ;;
    gemma-4-26b-a4b) export MODEL_PATH="google/gemma-4-26B-A4B" ;;
    qwen3.5-35b-a3b) export MODEL_PATH="Qwen/Qwen3.5-35B-A3B" ;;
  esac
fi

SUFFIX="${SUFFIX:-}"
ABLATION_TAG="t${TEXT_ONLY}_v${VISUAL_ONLY}"
SWEEP_BASE="${REPO_ROOT}/results/prune_eval_p${PRUNE_RATIO}/sweep_tasks-${MODEL_NAME}-${SUFFIX}${SUFFIX:+-}${ABLATION_TAG}"
export OUTPUT_DIR="${SWEEP_BASE}"

if [[ -d "${SWEEP_BASE}" ]]; then
  echo "warning: SWEEP_BASE already exists: ${SWEEP_BASE}" >&2
  if [[ ! -t 0 ]]; then
    echo "error: need interactive confirmation but stdin is not a terminal; exiting." >&2
    exit 1
  fi
  while true; do
    read -r -p "Continue reusing this directory? [y/n]: " reply
    case "${reply}" in
      [yY]) break ;;
      [nN]) echo "Aborted." >&2; exit 1 ;;
      *) echo "Please enter y or n." >&2 ;;
    esac
  done
fi

SUMMARY_FILE="${SUMMARY_FILE:-${SWEEP_BASE}/summary.md}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${SWEEP_BASE}/logs}"
SWEEP_SKIP_DONE="${SWEEP_SKIP_DONE:-1}"

mkdir -p "${SWEEP_BASE}"
mkdir -p "${SWEEP_LOG_DIR}"

RUN_IDX=0
SWEEP_SKIPPED=0
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "${SUMMARY_FILE}" ]]; then
  {
    echo "# Prune + multi-task eval sweep summary"
    echo ""
    echo "Auto-generated table; new runs are **appended** (this file is not overwritten)."
    echo ""
    echo "Output directory: \`${SWEEP_BASE}\`"
    echo "Use lmms eval: ${USE_LMMS_EVAL}"
    echo "Text only: ${TEXT_ONLY}"
    echo "Visual only: ${VISUAL_ONLY}"
    echo ""
  } >> "${SUMMARY_FILE}"
fi

echo "SUMMARY_FILE has been created: ${SUMMARY_FILE}"

{
  echo ""
  echo "## Sweep batch \`${SWEEP_ID}\`"
  echo ""
  echo "Started: $(date -Iseconds)"
  echo ""
  echo "SCORES_PATH: \`${SCORES_PATH}\`"
  echo ""
  echo "| # | task | inter_method | intra_method | modality_aware | shared_protect | text_only | visual_only | use_ema | expertwise_budget_normalize | ema_source_key | layerwise_loss_key | intra_expert_metric | smooth_fn | metric | detail | status | log |"
  echo "|---|------|--------------|--------------|----------------|----------------|-----------|-------------|---------|-----------------------------|----------------|--------------------|---------------------|-----------|--------|--------|--------|-----|"
} >> "${SUMMARY_FILE}"

for TASK in ${SWEEP_TASKS}; do
  for INTER_METHOD in ${SWEEP_INTER_METHODS}; do
    for INTRA_METHOD in ${SWEEP_INTRA_METHODS}; do
      for MODALITY_AWARE in ${SWEEP_MODALITY_AWARE}; do
        for SHARED_PROTECT in ${SWEEP_SHARED_PROTECT}; do
          for INTRA_EXPERT_METRIC in ${SWEEP_INTRA_EXPERT_METRICS}; do
            if [[ "${SWEEP_SKIP_DONE}" == "1" ]] && [[ -f "${SUMMARY_FILE}" ]]; then
              if grep -F "| ${TASK} | ${INTER_METHOD} | ${INTRA_METHOD} | ${MODALITY_AWARE} | ${SHARED_PROTECT} | ${TEXT_ONLY} | ${VISUAL_ONLY} | ${USE_EMA} | ${EXPERTWISE_BUDGET_NORMALIZE} | ${EMA_SOURCE_KEY} | ${LAYERWISE_LOSS_KEY} | ${INTRA_EXPERT_METRIC} | ${SMOOTH_FN} |" "${SUMMARY_FILE}" 2>/dev/null \
                   | grep -qF '| ok |'; then
                echo "[sweep] Skip (already ok): TASK=${TASK} INTER=${INTER_METHOD} INTRA=${INTRA_METHOD} MODALITY=${MODALITY_AWARE} SHARED=${SHARED_PROTECT} TEXT_ONLY=${TEXT_ONLY} VISUAL_ONLY=${VISUAL_ONLY} METRIC=${INTRA_EXPERT_METRIC}"
                SWEEP_SKIPPED=$((SWEEP_SKIPPED + 1))
                continue
              fi
            fi

            RUN_IDX=$((RUN_IDX + 1))
            TAG="$(printf '%04d' "${RUN_IDX}")_${SWEEP_ID}_${TASK}_${INTER_METHOD}_${INTRA_METHOD}_m${MODALITY_AWARE}_s${SHARED_PROTECT}_t${TEXT_ONLY}_v${VISUAL_ONLY}_${INTRA_EXPERT_METRIC}"
            TAG="${TAG//[^a-zA-Z0-9._-]/_}"
            RUN_LOG="${SWEEP_LOG_DIR}/stdout_${TAG}.log"

            echo ""
            echo "========== sweep run ${RUN_IDX}: ${MODEL_NAME} TASK=${TASK} INTER=${INTER_METHOD} INTRA=${INTRA_METHOD} MODALITY=${MODALITY_AWARE} SHARED=${SHARED_PROTECT} TEXT_ONLY=${TEXT_ONLY} VISUAL_ONLY=${VISUAL_ONLY} METRIC=${INTRA_EXPERT_METRIC} =========="

            set +e
            TASK="${TASK}" \
              INTER_METHOD="${INTER_METHOD}" \
              INTRA_METHOD="${INTRA_METHOD}" \
              MODALITY_AWARE="${MODALITY_AWARE}" \
              SHARED_PROTECT="${SHARED_PROTECT}" \
              TEXT_ONLY="${TEXT_ONLY}" \
              VISUAL_ONLY="${VISUAL_ONLY}" \
              NORMALIZE="${NORMALIZE}" \
              EXPERTWISE_BUDGET_NORMALIZE="${EXPERTWISE_BUDGET_NORMALIZE}" \
              USE_EMA="${USE_EMA}" \
              EMA_SOURCE_KEY="${EMA_SOURCE_KEY}" \
              LAYERWISE_LOSS_KEY="${LAYERWISE_LOSS_KEY}" \
              PRUNE_RATIO="${PRUNE_RATIO}" \
              INTRA_EXPERT_METRIC="${INTRA_EXPERT_METRIC}" \
              SMOOTH_FN="${SMOOTH_FN}" \
              MODEL_NAME="${MODEL_NAME}" \
              PREFIX="${PREFIX}" \
              NUM_SAMPLES="${NUM_SAMPLES}" \
              USE_LMMS_EVAL="${USE_LMMS_EVAL}" \
              bash "${SCRIPT_DIR}/run_prune_eval_kimi_gqa.sh" 2>&1 | tee "${RUN_LOG}"
            EXIT_CODE=${PIPESTATUS[0]}
            set -e

            METRIC_LINE=""
            METRIC_VAL=""
            METRIC_DETAIL=""

            if METRIC_LINE="$(grep -E '\[Run\] (Accuracy|Score|CIDEr|F1|Recall|mAP):' "${RUN_LOG}" | tail -n 1)"; then
              :
            else
              METRIC_LINE=""
            fi

            if [[ -n "${METRIC_LINE}" ]]; then
              METRIC_VAL="$(echo "${METRIC_LINE}" | sed -n 's/.*: \([0-9.]*\).*/\1/p')"
              METRIC_DETAIL="$(echo "${METRIC_LINE}" | sed -n 's/.*(\([^)]*\)).*/\1/p')"
              STATUS="ok"
            else
              if [[ "${EXIT_CODE}" -eq 0 ]]; then
                STATUS="no_metric_line"
              else
                STATUS="exit_${EXIT_CODE}"
              fi
            fi

            REL_LOG="logs/$(basename "${RUN_LOG}")"
            {
              echo "| ${RUN_IDX} | ${TASK} | ${INTER_METHOD} | ${INTRA_METHOD} | ${MODALITY_AWARE} | ${SHARED_PROTECT} | ${TEXT_ONLY} | ${VISUAL_ONLY} | ${USE_EMA} | ${EXPERTWISE_BUDGET_NORMALIZE} | ${EMA_SOURCE_KEY} | ${LAYERWISE_LOSS_KEY} | ${INTRA_EXPERT_METRIC} | ${SMOOTH_FN} | ${METRIC_VAL:-—} | ${METRIC_DETAIL:-—} | ${STATUS} | \`${REL_LOG}\` |"
            } >> "${SUMMARY_FILE}"
          done
        done
      done
    done
  done
done

{
  echo ""
  echo "Finished: $(date -Iseconds)"
  echo ""
  echo "**Batch stats:** appended_rows=${RUN_IDX}, skipped_already_ok=${SWEEP_SKIPPED}"
  echo ""
} >> "${SUMMARY_FILE}"

echo ""
echo "[sweep] This batch: appended ${RUN_IDX} row(s), skipped ${SWEEP_SKIPPED} (already ok) -> ${SUMMARY_FILE}"
