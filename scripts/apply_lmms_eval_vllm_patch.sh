#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${REPO_ROOT}/lmms-eval/lmms_eval/models/simple/vllm.py"
CHAT_TARGET="${REPO_ROOT}/lmms-eval/lmms_eval/models/chat/vllm.py"
PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_disable_inner_tqdm.patch"
CALLABLE_PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_silent_tqdm_callable.patch"
PARITY_PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_modes_parity.patch"
VIDEOMME_PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_qwen3_videomme.patch"
VIDEOMMMU_PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_qwen3_videommmu.patch"
SHORT_VIDEO_PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_chat_short_video.patch"
PREFILL_METRICS_PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_prefill_metrics.patch"
MM_CACHE_PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_vllm_mm_processor_cache.patch"
RESPONSE_CACHE_IDENTITY_PATCH_FILE="${REPO_ROOT}/patches/lmms_eval_response_cache_identity.patch"
RESPONSE_CACHE_TARGET="${REPO_ROOT}/lmms-eval/lmms_eval/caching/response_cache.py"

if [[ ! -f "${TARGET}" ]]; then
    echo "error: local lmms-eval checkout is missing: ${TARGET}" >&2
    exit 2
fi
if [[ ! -f "${CHAT_TARGET}" ]]; then
    echo "error: local lmms-eval chat vLLM backend is missing: ${CHAT_TARGET}" >&2
    exit 2
fi
if [[ ! -f "${RESPONSE_CACHE_TARGET}" ]]; then
    echo "error: local lmms-eval response cache is missing: ${RESPONSE_CACHE_TARGET}" >&2
    exit 2
fi

if ! grep -q '^RUNTIME_ONLY_MODEL_ARGS = frozenset(' "${RESPONSE_CACHE_TARGET}"; then
    if ! patch --dry-run --silent --forward -d "${REPO_ROOT}" -p1 < "${RESPONSE_CACHE_IDENTITY_PATCH_FILE}"; then
        echo "error: ${RESPONSE_CACHE_IDENTITY_PATCH_FILE} does not apply to the current lmms-eval checkout." >&2
        exit 2
    fi
    patch --silent --forward -d "${REPO_ROOT}" -p1 < "${RESPONSE_CACHE_IDENTITY_PATCH_FILE}"
fi

silent_marker_count="$(grep -c '^[[:space:]]*use_tqdm=SILENT_VLLM_TQDM,$' "${TARGET}" || true)"
if ! grep -q 'MAES_DISABLE_MM_PROCESSOR_CACHE' "${TARGET}"; then
    patch --dry-run --silent --forward -d "${REPO_ROOT}" -p1 < "${MM_CACHE_PATCH_FILE}"
    patch --silent --forward -d "${REPO_ROOT}" -p1 < "${MM_CACHE_PATCH_FILE}"
fi
if [[ "${silent_marker_count}" == "2" || "${silent_marker_count}" == "3" ]]; then
    :
elif [[ "${silent_marker_count}" != "0" ]]; then
    echo "error: lmms-eval vLLM callable progress patch has an unexpected marker count (${silent_marker_count})." >&2
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

videomme_marker_count="$(grep -c '^from eval\.vllm_qwen3_videomme import prepare_qwen3_videomme_input$' "${TARGET}" || true)"
if [[ "${videomme_marker_count}" == "1" ]]; then
    :
elif [[ "${videomme_marker_count}" != "0" ]]; then
    echo "error: lmms-eval Qwen3 VideoMME patch has duplicate markers (${videomme_marker_count})." >&2
    exit 2
else
    if ! patch --dry-run --silent --forward -d "${REPO_ROOT}" -p1 < "${VIDEOMME_PATCH_FILE}"; then
        echo "error: ${VIDEOMME_PATCH_FILE} does not apply to the current lmms-eval checkout." >&2
        exit 2
    fi
    patch --silent --forward -d "${REPO_ROOT}" -p1 < "${VIDEOMME_PATCH_FILE}"
fi

silent_marker_count="$(grep -c '^[[:space:]]*use_tqdm=SILENT_VLLM_TQDM,$' "${TARGET}" || true)"
native_no_tqdm_count="$(grep -c '^[[:space:]]*use_tqdm=False,$' "${TARGET}" || true)"
if [[ "${silent_marker_count}" != "2" || "${native_no_tqdm_count}" != "1" ]]; then
    echo "error: lmms-eval vLLM progress configuration is inconsistent (chat callable=${silent_marker_count}/2, native disabled=${native_no_tqdm_count}/1)." >&2
    exit 2
fi

videommmu_marker_count="$(grep -c '^from eval\.vllm_qwen3_videommmu import prepare_qwen3_videommmu_input$' "${TARGET}" || true)"
if [[ "${videommmu_marker_count}" == "1" ]]; then
    :
elif [[ "${videommmu_marker_count}" != "0" ]]; then
    echo "error: lmms-eval Qwen3 VideoMMMU patch has duplicate markers (${videommmu_marker_count})." >&2
    exit 2
else
    if ! patch --dry-run --silent --forward -d "${REPO_ROOT}" -p1 < "${VIDEOMMMU_PATCH_FILE}"; then
        echo "error: ${VIDEOMMMU_PATCH_FILE} does not apply to the current lmms-eval checkout." >&2
        exit 2
    fi
    patch --silent --forward -d "${REPO_ROOT}" -p1 < "${VIDEOMMMU_PATCH_FILE}"
fi

short_video_marker_count="$(grep -c '^from eval\.vllm_short_video import to_openai_messages_with_nframe_fallback$' "${CHAT_TARGET}" || true)"
if [[ "${short_video_marker_count}" == "1" ]]; then
    :
elif [[ "${short_video_marker_count}" != "0" ]]; then
    echo "error: lmms-eval short-video patch has duplicate markers (${short_video_marker_count})." >&2
    exit 2
else
    if ! patch --dry-run --silent --forward -d "${REPO_ROOT}" -p1 < "${SHORT_VIDEO_PATCH_FILE}"; then
        echo "error: ${SHORT_VIDEO_PATCH_FILE} does not apply to the current lmms-eval checkout." >&2
        exit 2
    fi
    patch --silent --forward -d "${REPO_ROOT}" -p1 < "${SHORT_VIDEO_PATCH_FILE}"
fi

prefill_metrics_marker_count="$(grep -Ec '^[[:space:]]*if "time_(to_first_token|per_output_token_seconds)" in name and metric\.count:$' "${CHAT_TARGET}" || true)"
if [[ "${prefill_metrics_marker_count}" == "2" ]]; then
    :
elif [[ "${prefill_metrics_marker_count}" != "0" ]]; then
    echo "error: lmms-eval prefill metric patch is only partially applied (${prefill_metrics_marker_count}/2 markers)." >&2
    exit 2
else
    if ! patch --dry-run --silent --forward -d "${REPO_ROOT}" -p1 < "${PREFILL_METRICS_PATCH_FILE}"; then
        echo "error: ${PREFILL_METRICS_PATCH_FILE} does not apply to the current lmms-eval checkout." >&2
        exit 2
    fi
    patch --silent --forward -d "${REPO_ROOT}" -p1 < "${PREFILL_METRICS_PATCH_FILE}"
fi
