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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PREFIX="${PREFIX:-${REPO_ROOT}}"
export PYTHONPATH="${PREFIX}"

export SCORES_PATH="./storage/prune/scores/kimi-vl-a3b_gqa-rell2-041513.pt"

# ── Task grid ──────────────────────────────────────────────────────────────────
# All 14 tasks requested; override via SWEEP_TASKS env var.
SWEEP_TASKS="${SWEEP_TASKS:-textvqa gqa chartqa mmstar mmbench mmvet mme realworldqa coco2017cap mvbench egoschema videomme longvideobench video_mmmu}"

# ── Setting grids (same defaults as sweep_prune_eval_kimi_gqa.sh) ──────────────
SWEEP_INTER_METHODS="${SWEEP_INTER_METHODS:-loss_smooth_2 uniform uniform_coverage}"
SWEEP_INTRA_METHODS="${SWEEP_INTRA_METHODS:-uniform second_attr_coverage second_attr_fillzero_coverage}"
SWEEP_MODALITY_AWARE="${SWEEP_MODALITY_AWARE:-0 1}"
SMOOTH_FN="${SMOOTH_FN:-sqrt}"
SWEEP_INTRA_EXPERT_METRICS="${SWEEP_INTRA_EXPERT_METRICS:-gateup_act down_second_order_exact 3proj_second_order}"
# 3proj_second_order down_saliency 3proj_saliency 3proj_grad wg

# ── Output paths ───────────────────────────────────────────────────────────────
# Timestamp is fixed at script start so all runs share the same directory.
SWEEP_TS="${SWEEP_TS:-$(date +%m%d%H%M)}"
MODEL_NAME="${MODEL_NAME:-kimi}"
SWEEP_BASE="${REPO_ROOT}/results/prune_eval_p50/sweep_tasks-${MODEL_NAME}-gqa-rell2-041513-${SWEEP_TS}"
export OUTPUT_DIR="${SWEEP_BASE}"

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
    echo ""
  } >> "${SUMMARY_FILE}"
fi

{
  echo ""
  echo "## Sweep batch \`${SWEEP_ID}\`"
  echo ""
  echo "Started: $(date -Iseconds)"
  echo ""
  echo "| # | task | inter_method | intra_method | modality_aware | intra_expert_metric | smooth_fn | metric | detail | status | log |"
  echo "|---|------|--------------|--------------|----------------|---------------------|-----------|--------|--------|--------|-----|"
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
          echo "========== sweep run ${RUN_IDX}: TASK=${TASK} INTER=${INTER_METHOD} INTRA=${INTRA_METHOD} MODALITY=${MODALITY_AWARE} METRIC=${INTRA_EXPERT_METRIC} =========="

          set +e
          TASK="${TASK}" \
            INTER_METHOD="${INTER_METHOD}" \
            INTRA_METHOD="${INTRA_METHOD}" \
            MODALITY_AWARE="${MODALITY_AWARE}" \
            INTRA_EXPERT_METRIC="${INTRA_EXPERT_METRIC}" \
            SMOOTH_FN="${SMOOTH_FN}" \
            MODEL_NAME="${MODEL_NAME}" \
            PREFIX="${PREFIX}" \
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
            echo "| ${RUN_IDX} | ${TASK} | ${INTER_METHOD} | ${INTRA_METHOD} | ${MODALITY_AWARE} | ${INTRA_EXPERT_METRIC} | ${SMOOTH_FN} | ${METRIC_VAL:-—} | ${METRIC_DETAIL:-—} | ${STATUS} | \`${REL_LOG}\` |"
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
