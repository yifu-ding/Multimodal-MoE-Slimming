#!/usr/bin/env bash
# Sweep TASK × INTER_METHOD × INTRA_METHOD × MODALITY_AWARE × INTRA_EXPERT_METRIC
# by calling run_prune_eval_kimi_gqa.sh (with TASK env var forwarded).
#
# Appends one markdown table row per run to a per-batch summary.md under:
#   results/prune_eval_kimi_gqa_p50/sweep_tasks-kimi-gqa-rell2-<SWEEP_TS>/
#
# Override grids (space-separated):
#   SWEEP_TASKS="gqa textvqa"          # subset of default task list
#   SWEEP_INTER_METHODS="uniform loss"
#   SWEEP_INTRA_METHODS="uniform second_attr_coverage"
#   SWEEP_MODALITY_AWARE="0 1"
#   SWEEP_INTRA_EXPERT_METRICS="gateup_act 3proj_act"
#
# Resume / skip already-done runs (default on):
#   If summary.md already has a row with the same task, inter_method, intra_method,
#   modality_aware, intra_expert_metric and status ``ok``, that combination is skipped.
#   SWEEP_SKIP_DONE=0  — run everything (ignore summary)
#
# Other env vars are passed through to run_prune_eval_kimi_gqa.sh
# (SCORES_PATH, PRUNE_RATIO, NUM_SAMPLES, ...).
# MODEL_NAME selects the model; this script also sets MODEL_PATH to the default HF id unless you set
# SWEEP_MODEL_PATH (so export MODEL_PATH=... in your shell is ignored in favor of MODEL_NAME).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PREFIX="${PREFIX:-${REPO_ROOT}}"
export PYTHONPATH="${PREFIX}"

# export SCORES_PATH="storage/prune/scores/kimi-vl-a3b_gqa-rell2-041513.pt"
# export SCORES_PATH="storage/prune/scores/kimi-vl-a3b_coco-rell2-fill1-0416-115847.pt"
# export SCORES_PATH="storage/data_distill_kimi/gqa-sample_at1.0-0418234400/distilled-0419142618/scores-step4000.pt"
# export SCORES_PATH="storage/data_distill_kimi/gqa-sample_at1.0-0418234400/distilled-0419182016/distilled_hidden-scores.pt"
# export SCORES_PATH="storage/data_distill_kimi/gqa-sample_at1.0-0418234400/distilled-0419164811/distilled_hidden-step7000-scores.pt"
export SCORES_PATH="${SCORES_PATH:-}"

NUM_SAMPLES="${NUM_SAMPLES:-0}"
USE_LMMS_EVAL=${USE_LMMS_EVAL:-0}

# ── Task grid ──────────────────────────────────────────────────────────────────
# All 14 tasks requested; override via SWEEP_TASKS env var.
SWEEP_TASKS="${SWEEP_TASKS:-textvqa chartqa coco2017cap mmstar mmbench realworldqa gqa mme}"
# mmvet video_mmmu videomme mvbench egoschema
# ── Setting grids (same defaults as sweep_prune_eval_kimi_gqa.sh) ──────────────
SWEEP_INTER_METHODS="${SWEEP_INTER_METHODS:-uniform_coverage}"
SWEEP_INTRA_METHODS="${SWEEP_INTRA_METHODS:-second_attr_coverage}"
SWEEP_MODALITY_AWARE="${SWEEP_MODALITY_AWARE:-1}"
NORMALIZE="${NORMALIZE:-0}"
SMOOTH_FN="${SMOOTH_FN:-cbrt}" # sqrt cbrt fourth_root log
SWEEP_INTRA_EXPERT_METRICS="${SWEEP_INTRA_EXPERT_METRICS:-gateup_act}"
# 3proj_second_order down_saliency 3proj_saliency 3proj_grad wg

# ── Output paths ───────────────────────────────────────────────────────────────
# Timestamp is fixed at script start so all runs share the same directory.
SWEEP_TS="${SWEEP_TS:-$(date +%m%d%H%M)}"
MODEL_NAME="${MODEL_NAME:-kimi-vl-a3b}"
# Short family tags (no -instruct), same as run_prune_eval_kimi_gqa.sh
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
    echo "error: MODEL_NAME must be one of: deepseek-vl2-small kimi-vl-a3b kimi-vl-a3b-instruct qwen3-vl-30b-a3b qwen3-vl-30b-a3b-instruct internvl3_5-30b-a3b-hf gemma-4-26b-a4b qwen3.5-35b-a3b (see run_prune_eval_kimi_gqa.sh). Got: ${MODEL_NAME}" >&2
    exit 1
    ;;
esac
# Pin MODEL_PATH from MODEL_NAME (HF repo ids, aligned with download_hf_benchmarks.MODELS) so a
# stale export MODEL_PATH=... in the shell does not load a different model than MODEL_NAME.
# Use SWEEP_MODEL_PATH=/path/or/hf-id to use a local copy or non-default id instead.
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
SWEEP_BASE="${REPO_ROOT}/results/prune_eval_p50/sweep_tasks-${MODEL_NAME}-mixed-num_342-token_2048-sample_at1.0-0421175527-teacher-mean"  # -${SWEEP_TS}
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

# ── Counters ───────────────────────────────────────────────────────────────────
RUN_IDX=0
SWEEP_SKIPPED=0
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"

# ── Init summary file (only write header once) ─────────────────────────────────
if [[ ! -f "${SUMMARY_FILE}" ]]; then
  {
    echo "# Prune + multi-task eval sweep summary"
    echo ""
    echo "Auto-generated table; new runs are **appended** (this file is not overwritten)."
    echo ""
    echo "Output directory: \`${SWEEP_BASE}\`"
    echo "Use lmms eval: ${USE_LMMS_EVAL}"
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
  echo "| # | task | inter_method | intra_method | modality_aware | normalize | intra_expert_metric | smooth_fn | metric | detail | status | log |"
  echo "|---|------|--------------|--------------|----------------|----------------|---------------------|-----------|--------|--------|--------|-----|"
} >> "${SUMMARY_FILE}"

# ── Main sweep ─────────────────────────────────────────────────────────────────
for TASK in ${SWEEP_TASKS}; do
  for INTER_METHOD in ${SWEEP_INTER_METHODS}; do
    for INTRA_METHOD in ${SWEEP_INTRA_METHODS}; do
      for MODALITY_AWARE in ${SWEEP_MODALITY_AWARE}; do
        for INTRA_EXPERT_METRIC in ${SWEEP_INTRA_EXPERT_METRICS}; do

          # ── Skip-done check ──────────────────────────────────────────────────
          if [[ "${SWEEP_SKIP_DONE}" == "1" ]] && [[ -f "${SUMMARY_FILE}" ]]; then
            if grep -F "| ${TASK} | ${INTER_METHOD} | ${INTRA_METHOD} | ${MODALITY_AWARE} | ${INTRA_EXPERT_METRIC} |" "${SUMMARY_FILE}" 2>/dev/null \
                 | grep -qF '| ok |'; then
              echo "[sweep] Skip (already ok): TASK=${TASK} INTER=${INTER_METHOD} INTRA=${INTRA_METHOD} MODALITY=${MODALITY_AWARE} METRIC=${INTRA_EXPERT_METRIC}"
              SWEEP_SKIPPED=$((SWEEP_SKIPPED + 1))
              continue
            fi
          fi

          # ── Run ──────────────────────────────────────────────────────────────
          RUN_IDX=$((RUN_IDX + 1))
          TAG="$(printf '%04d' "${RUN_IDX}")_${SWEEP_ID}_${TASK}_${INTER_METHOD}_${INTRA_METHOD}_m${MODALITY_AWARE}_${INTRA_EXPERT_METRIC}"
          TAG="${TAG//[^a-zA-Z0-9._-]/_}"
          RUN_LOG="${SWEEP_LOG_DIR}/stdout_${TAG}.log"

          echo ""
          echo "========== sweep run ${RUN_IDX}: ${MODEL_NAME} TASK=${TASK} INTER=${INTER_METHOD} INTRA=${INTRA_METHOD} MODALITY=${MODALITY_AWARE} METRIC=${INTRA_EXPERT_METRIC} =========="

          set +e
          TASK="${TASK}" \
            INTER_METHOD="${INTER_METHOD}" \
            INTRA_METHOD="${INTRA_METHOD}" \
            MODALITY_AWARE="${MODALITY_AWARE}" \
            NORMALIZE="${NORMALIZE}" \
            INTRA_EXPERT_METRIC="${INTRA_EXPERT_METRIC}" \
            SMOOTH_FN="${SMOOTH_FN}" \
            MODEL_NAME="${MODEL_NAME}" \
            PREFIX="${PREFIX}" \
            NUM_SAMPLES="${NUM_SAMPLES}" \
            USE_LMMS_EVAL="${USE_LMMS_EVAL}" \
            bash "${SCRIPT_DIR}/run_prune_eval_kimi_gqa.sh" 2>&1 | tee "${RUN_LOG}"
          EXIT_CODE=${PIPESTATUS[0]}
          set -e

          # ── Extract metric from log ──────────────────────────────────────────
          # Try various metric line patterns (task-specific output formats):
          #   [Run] Accuracy: 0.6120  (612/1000)   — GQA / VQA-style tasks
          #   [Run] Score: 0.7234  (detail)          — score-based tasks
          #   [Run] CIDEr: 1.2345                    — captioning tasks
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
            echo "| ${RUN_IDX} | ${TASK} | ${INTER_METHOD} | ${INTRA_METHOD} | ${MODALITY_AWARE} | ${NORMALIZE} | ${INTRA_EXPERT_METRIC} | ${SMOOTH_FN} | ${METRIC_VAL:-—} | ${METRIC_DETAIL:-—} | ${STATUS} | \`${REL_LOG}\` |"
          } >> "${SUMMARY_FILE}"

        done
      done
    done
  done
done

# ── Batch footer ───────────────────────────────────────────────────────────────
{
  echo ""
  echo "Finished: $(date -Iseconds)"
  echo ""
  echo "**Batch stats:** appended_rows=${RUN_IDX}, skipped_already_ok=${SWEEP_SKIPPED}"
  echo ""
} >> "${SUMMARY_FILE}"

echo ""
echo "[sweep] This batch: appended ${RUN_IDX} row(s), skipped ${SWEEP_SKIPPED} (already ok) -> ${SUMMARY_FILE}"
