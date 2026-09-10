#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${REPO_ROOT}/lmms-eval/lmms_eval/models/simple/vllm.py"
PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_disable_inner_tqdm.patch"
CALLABLE_PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_silent_tqdm_callable.patch"
PARITY_PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_modes_parity.patch"

if [[ ! -f "${TARGET}" ]]; then
    echo "error: local lmms-eval checkout is missing: ${TARGET}" >&2
    exit 2
fi

silent_marker_count="$(grep -c '^[[:space:]]*use_tqdm=SILENT_VLLM_TQDM,$' "${TARGET}" || true)"
if [[ "${silent_marker_count}" == "2" ]]; then
    :
elif [[ "${silent_marker_count}" != "0" ]]; then
    echo "error: lmms-eval vLLM callable progress patch is only partially applied (${silent_marker_count}/2 markers)." >&2
    exit 2
else
    legacy_marker_count="$(grep -c '^[[:space:]]*use_tqdm=False,$' "${TARGET}" || true)"
    if [[ "${legacy_marker_count}" == "0" ]]; then
        if ! patch --dry-run --silent --forward -d "${REPO_ROOT}" -p1 < "${PATCH_FILE}"; then
            echo "error: ${PATCH_FILE} does not apply to the current lmms-eval checkout." >&2
            exit 2
        fi
        patch --silent --forward -d "${REPO_ROOT}" -p1 < "${PATCH_FILE}"
    elif [[ "${legacy_marker_count}" != "2" ]]; then
        echo "error: lmms-eval vLLM legacy progress patch is only partially applied (${legacy_marker_count}/2 markers)." >&2
        exit 2
    fi

    if ! patch --dry-run --silent --forward -d "${REPO_ROOT}" -p1 < "${CALLABLE_PATCH_FILE}"; then
        echo "error: ${CALLABLE_PATCH_FILE} does not apply to the current lmms-eval checkout." >&2
        exit 2
    fi
    patch --silent --forward -d "${REPO_ROOT}" -p1 < "${CALLABLE_PATCH_FILE}"
fi

parity_marker_count="$(grep -Ec '^[[:space:]]*(params = None|return request_max_new_tokens|request_params = self\._build_sampling_params_dict\(gen_kwargs\)|raise ValueError\("All requests in a vLLM batch must use identical generation parameters\."\))$' "${TARGET}" || true)"
if [[ "${parity_marker_count}" == "4" ]]; then
    :
elif [[ "${parity_marker_count}" != "0" ]]; then
    echo "error: lmms-eval vLLM MoDES parity patch is only partially applied (${parity_marker_count}/4 markers)." >&2
    exit 2
else
    if ! patch --dry-run --silent --forward -d "${REPO_ROOT}" -p1 < "${PARITY_PATCH_FILE}"; then
        echo "error: ${PARITY_PATCH_FILE} does not apply to the current lmms-eval checkout." >&2
        exit 2
    fi
    patch --silent --forward -d "${REPO_ROOT}" -p1 < "${PARITY_PATCH_FILE}"
fi
