#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${REPO_ROOT}/lmms-eval/lmms_eval/models/simple/vllm.py"
PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_disable_inner_tqdm.patch"

if [[ ! -f "${TARGET}" ]]; then
    echo "error: local lmms-eval checkout is missing: ${TARGET}" >&2
    exit 2
fi

marker_count="$(grep -c '^[[:space:]]*use_tqdm=False,$' "${TARGET}" || true)"
if [[ "${marker_count}" == "2" ]]; then
    exit 0
fi
if [[ "${marker_count}" != "0" ]]; then
    echo "error: lmms-eval vLLM progress patch is only partially applied (${marker_count}/2 markers)." >&2
    exit 2
fi

if ! patch --dry-run --silent --forward -d "${REPO_ROOT}" -p1 < "${PATCH_FILE}"; then
    echo "error: ${PATCH_FILE} does not apply to the current lmms-eval checkout." >&2
    exit 2
fi

patch --silent --forward -d "${REPO_ROOT}" -p1 < "${PATCH_FILE}"
echo "Applied lmms-eval vLLM progress patch (inner vLLM tqdm disabled)."
