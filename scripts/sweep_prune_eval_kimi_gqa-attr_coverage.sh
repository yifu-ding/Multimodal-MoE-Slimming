#!/usr/bin/env bash
# Sweep INTER_METHOD × INTRA_METHOD × MODALITY_AWARE × INTRA_EXPERT_METRIC by calling run_prune_eval_kimi_gqa.sh.
# Appends one markdown table row per run to results/prune_eval_kimi_gqa_p50/summary.md (never overwrites).
#
# Override grids (space-separated):
#   SWEEP_INTER_METHODS="uniform loss"
#   SWEEP_INTRA_METHODS="uniform second_attr_coverage"
#   SWEEP_MODALITY_AWARE="0 1"
#   SWEEP_INTRA_EXPERT_METRICS="gateup_act 3proj_act"   # shorten as needed
#
# Default SWEEP_INTRA_EXPERT_METRICS matches run_prune_eval_kimi_gqa.sh (INTRA_EXPERT_METRIC / channel_scores keys):
#   gateup_act, gateup_act_text, gateup_act_visual
#   3proj_act, 3proj_act_text, 3proj_act_visual
#   down_second_order, down_second_order_text, down_second_order_visual
#   3proj_second_order, 3proj_second_order_text, 3proj_second_order_visual
#   down_saliency, down_saliency_text, down_saliency_visual
#   3proj_saliency, 3proj_saliency_text, 3proj_saliency_visual
#   wa, wa_text, wa_visual
#   3proj_grad, 3proj_grad_text, 3proj_grad_visual
#   wg, weight
#
# Other env vars are passed through to run_prune_eval_kimi_gqa.sh (SCORES_PATH, PRUNE_RATIO, NUM_SAMPLES, ...).
#
# Skip already-finished settings (default on):
#   If summary.md already has a row with the same inter_method, intra_method, modality_aware,
#   intra_expert_metric and status ``ok``, that combination is skipped.
#   SWEEP_SKIP_DONE=0  — run everything (ignore summary)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PREFIX="${PREFIX:-${REPO_ROOT}}"
export PYTHONPATH="${PREFIX}"

# Default grids (edit or override via env)
# INTRA_METHOD = --intra_method (intra-layer planner); see run_prune_eval_kimi_gqa.sh
SWEEP_INTER_METHODS="${SWEEP_INTER_METHODS:-uniform uniform_coverage loss_smooth_1 loss_smooth_2 loss_coverage}"
SWEEP_INTRA_METHODS="${SWEEP_INTRA_METHODS:-attr_coverage}"  # usage usage_coverage router router_coverage attr_coverage 
SWEEP_MODALITY_AWARE="${SWEEP_MODALITY_AWARE:-0}"  # 0, 1
# Not swept; passed through to run_prune_eval_kimi_gqa.sh (see SMOOTH_FN there).
SMOOTH_FN="${SMOOTH_FN:-sqrt}"
# Default: full list from run_prune_eval_kimi_gqa.sh (long run); override to shorten.
SWEEP_INTRA_EXPERT_METRICS="${SWEEP_INTRA_EXPERT_METRICS:-gateup_act 3proj_act down_second_order 3proj_second_order down_saliency 3proj_saliency 3proj_grad}"

SUMMARY_FILE="${SUMMARY_FILE:-${REPO_ROOT}/results/prune_eval_kimi_gqa_p50/summary-kimi_gqa-rell2-04131052.md}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${REPO_ROOT}/results/prune_eval_kimi_gqa_p50/sweep_runs-kimi_gqa-rell2-04131052}"
SWEEP_SKIP_DONE="${SWEEP_SKIP_DONE:-1}"
mkdir -p "$(dirname "${SUMMARY_FILE}")"
mkdir -p "${SWEEP_LOG_DIR}"

RUN_IDX=0
SWEEP_SKIPPED=0
SWEEP_ID="$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "${SUMMARY_FILE}" ]]; then
  {
    echo "# Prune + GQA eval sweep summary"
    echo ""
    echo "Auto-generated table; new runs are **appended** (this file is not overwritten)."
    echo ""
  } >> "${SUMMARY_FILE}"
fi

{
  echo ""
  echo "## Sweep batch \`${SWEEP_ID}\`"
  echo ""
  echo "Started: $(date -Iseconds)"
  echo ""
  echo "| # | inter_method | intra_method | modality_aware | intra_expert_metric | smooth_fn | accuracy | correct/total | status | log |"
  echo "|---|--------------|--------------|----------------|---------------------|-----------|----------|---------------|--------|-----|"
} >> "${SUMMARY_FILE}"

for INTER_METHOD in ${SWEEP_INTER_METHODS}; do
  for INTRA_METHOD in ${SWEEP_INTRA_METHODS}; do
    for MODALITY_AWARE in ${SWEEP_MODALITY_AWARE}; do
      for INTRA_EXPERT_METRIC in ${SWEEP_INTRA_EXPERT_METRICS}; do
        if [[ "${SWEEP_SKIP_DONE}" == "1" ]] && [[ -f "${SUMMARY_FILE}" ]]; then
          if grep -F "| ${INTER_METHOD} | ${INTRA_METHOD} | ${MODALITY_AWARE} | ${INTRA_EXPERT_METRIC} |" "${SUMMARY_FILE}" 2>/dev/null | grep -qF '| ok |'; then
            echo "[sweep] Skip (already in summary with ok): INTER=${INTER_METHOD} INTRA=${INTRA_METHOD} MODALITY=${MODALITY_AWARE} METRIC=${INTRA_EXPERT_METRIC}"
            SWEEP_SKIPPED=$((SWEEP_SKIPPED + 1))
            continue
          fi
        fi

        RUN_IDX=$((RUN_IDX + 1))
        TAG="$(printf '%04d' "${RUN_IDX}")_${SWEEP_ID}_${INTER_METHOD}_${INTRA_METHOD}_m${MODALITY_AWARE}_${INTRA_EXPERT_METRIC}"
        # sanitize filename
        TAG="${TAG//[^a-zA-Z0-9._-]/_}"
        RUN_LOG="${SWEEP_LOG_DIR}/stdout_${TAG}.log"

        echo ""
        echo "========== sweep run ${RUN_IDX}: INTER=${INTER_METHOD} INTRA=${INTRA_METHOD} MODALITY=${MODALITY_AWARE} METRIC=${INTRA_EXPERT_METRIC} =========="

        set +e
        INTER_METHOD="${INTER_METHOD}" \
          INTRA_METHOD="${INTRA_METHOD}" \
          MODALITY_AWARE="${MODALITY_AWARE}" \
          INTRA_EXPERT_METRIC="${INTRA_EXPERT_METRIC}" \
          SMOOTH_FN="${SMOOTH_FN}" \
          PREFIX="${PREFIX}" \
          bash "${SCRIPT_DIR}/run_prune_eval_kimi_gqa.sh" 2>&1 | tee "${RUN_LOG}"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e

        ACC_LINE="$(grep '\[Run\] Accuracy:' "${RUN_LOG}" | tail -n 1 || true)"
        # Example: [Run] Accuracy: 0.6120  (612/1000)
        if [[ -n "${ACC_LINE}" ]]; then
          ACC="$(echo "${ACC_LINE}" | sed -n 's/.*Accuracy: \([0-9.]*\).*/\1/p')"
          DETAIL="$(echo "${ACC_LINE}" | sed -n 's/.*(\([^)]*\)).*/\1/p')"
          STATUS="ok"
        else
          ACC=""
          DETAIL=""
          if [[ "${EXIT_CODE}" -eq 0 ]]; then
            STATUS="no_accuracy_line"
          else
            STATUS="exit_${EXIT_CODE}"
          fi
        fi

        REL_LOG="sweep_runs/$(basename "${RUN_LOG}")"
        {
          echo "| ${RUN_IDX} | ${INTER_METHOD} | ${INTRA_METHOD} | ${MODALITY_AWARE} | ${INTRA_EXPERT_METRIC} | ${SMOOTH_FN} | ${ACC:-—} | ${DETAIL:-—} | ${STATUS} | \`${REL_LOG}\` |"
        } >> "${SUMMARY_FILE}"

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
