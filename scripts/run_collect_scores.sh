#!/usr/bin/env bash
set -euo pipefail

# Phase 1: Collect per-channel importance scores for Kimi-VL MoE experts.
#
# Quick smoke test (weight scoring, no GPU data needed):
#   SCORE_TYPE=weight bash scripts/run_collect_scores.sh
#
# Activation scoring on 128 GQA samples:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Activation scoring on 128 COCO2017-Capval samples:
#   DATASET=coco CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Activation scoring on 128 VMMMU samples:
#   DATASET=VMMMU CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Activation scoring with text/visual split on 128 GQA samples:
#   MODALITY_AWARE=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_collect_scores.sh
#
# Full run:
#   CUDA_VISIBLE_DEVICES=0 NUM_SAMPLES=512 bash scripts/run_collect_scores.sh
#
# Frozen mixed-source run (sample count and score-token quota come from manifest):
#   SELECTION_MANIFEST=storage/calibration_manifests/mixed-512.json \
#     BATCH_SIZE=16 SECOND_ORDER_CHUNK_SIZE=auto AGGREGATION=mean \
#     bash scripts/run_collect_scores.sh

source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu

if [[ "${CONDA_DEFAULT_ENV:-}" == "ds-vl2-h20" ]]; then
    export LD_LIBRARY_PATH="/home/dyf/miniconda/envs/ds-vl2-h20/lib:${LD_LIBRARY_PATH:-}"
fi

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
# MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
# deepseek-ai/deepseek-vl2-small

case "${MODEL_PATH,,}" in
    kimi-vl-a3b|kimi-vl-a3b-instruct)
        export MODEL_PATH="moonshotai/Kimi-VL-A3B-Instruct"
        ;;
    qwen3-vl-30b-a3b|qwen3-vl-30b-a3b-instruct)
        export MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Instruct"
        ;;
    deepseek-vl2-small)
        export MODEL_PATH="deepseek-ai/deepseek-vl2-small"
        ;;
    internvl3_5-30b-a3b-hf)
        export MODEL_PATH="OpenGVLab/InternVL3_5-30B-A3B-HF"
        ;;
esac

DATASET="${DATASET:-video_mmmu}"  # DATASET=gqa, coco, video_mmmu, m4_instruct, star
SELECTION_MANIFEST="${SELECTION_MANIFEST:-}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
TOKEN_PER_SAMPLE="${TOKEN_PER_SAMPLE:-2048}"
BATCH_SIZE="${BATCH_SIZE:-8}"
START_IDX="${START_IDX:-0}"
SUBSET_SEED="${SUBSET_SEED:-42}"
LOSS_FN="${LOSS_FN:-rel_l2}"
SECOND_ORDER_IMPL="${SECOND_ORDER_IMPL:-vectorized}"
SECOND_ORDER_CHUNK_SIZE="${SECOND_ORDER_CHUNK_SIZE:-auto}"
SECOND_ORDER_MAX_CHUNK_SIZE="${SECOND_ORDER_MAX_CHUNK_SIZE:-0}"
SECOND_ORDER_PROFILE="${SECOND_ORDER_PROFILE:-0}"
LAYERS="${LAYERS:-}"  # 默认不传值，全部层calibration
# LAYERS="${LAYERS:-1-6}"
# LAYERS="${LAYERS:-7-13}"
# LAYERS="${LAYERS:-14-20}"
# LAYERS="${LAYERS:-20-26}"

EMA="${EMA:-0.9}"
AGGREGATION="${AGGREGATION:-mean}"
FILL_ZERO_FOR_UNROUTED="${FILL_ZERO_FOR_UNROUTED:-0}"

export SECOND_ORDER_IMPL SECOND_ORDER_CHUNK_SIZE SECOND_ORDER_MAX_CHUNK_SIZE SECOND_ORDER_PROFILE

if [[ -n "${SELECTION_MANIFEST}" ]]; then
    if [[ ! -f "${SELECTION_MANIFEST}" ]]; then
        echo "Selection manifest does not exist: ${SELECTION_MANIFEST}" >&2
        exit 1
    fi
    DATASET="mixed"
    DATASET_TAG="mixed-$(basename "${SELECTION_MANIFEST}" .json)"
else
case "${DATASET,,}" in
    gqa)
        DATASET="gqa"
        DATASET_TAG="gqa"
        ;;
    coco)
        DATASET="coco"
        DATASET_TAG="coco"
        ;;
    vmmmu|video_mmmu)
        DATASET="video_mmmu"
        DATASET_TAG="video_mmmu"
        ;;
    m4|m4_instruct|m4-instruct|m4_instruct_data)
        DATASET="m4_instruct"
        DATASET_TAG="m4_instruct"
        ;;
    star|star_train_subset|star_train_subset_256)
        DATASET="star"
        DATASET_TAG="star"
        ;;
    *)
        echo "Unsupported DATASET=${DATASET}. Supported values: gqa, coco, VMMMU (video_mmmu), m4 (m4_instruct), star." >&2
        exit 1
        ;;
esac
fi

MODEL_TAG_RAW="${MODEL_PATH##*/}"
MODEL_TAG="${MODEL_TAG_RAW,,}"
case "${MODEL_TAG}" in
    qwen3-vl-30b-a3b-instruct)
        MODEL_TAG="qwen3-vl-30b-a3b"
        ;;
    kimi-vl-a3b-instruct)
        MODEL_TAG="kimi-vl-a3b"
        ;;
    deepseek-vl2-small)
        MODEL_TAG="deepseek-vl2-small"
        ;;
    internvl3_5-30b-a3b-hf)
        MODEL_TAG="internvl3_5-30b-a3b"
        ;;
    internvl-3.5-gpt-oss-20b-a4b-preview-hf)
        MODEL_TAG="internvl-3.5-20b-a4b"
        ;;
    *)
        MODEL_TAG="${MODEL_TAG%-instruct}"
        ;;
esac

if [[ -n "${SELECTION_MANIFEST}" ]]; then
    DEFAULT_OUTPUT_DIR="${PREFIX}/storage/scores/${MODEL_TAG}_${DATASET_TAG}-${LOSS_FN}-$(date +%m%d-%H%M%S)"
else
    DEFAULT_OUTPUT_DIR="${PREFIX}/storage/scores/${MODEL_TAG}_${DATASET_TAG}-num_${NUM_SAMPLES}-token_${TOKEN_PER_SAMPLE}-${LOSS_FN}-$(date +%m%d-%H%M%S)"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.collect_scores_main
    --model_name_or_path "${MODEL_PATH}"
    --output_dir         "${OUTPUT_DIR}"
    --dataset            "${DATASET}"
    --num_samples        "${NUM_SAMPLES}"
    --token_per_sample   "${TOKEN_PER_SAMPLE}"
    --batch_size         "${BATCH_SIZE}"
    --start_idx          "${START_IDX}"
    --subset_seed        "${SUBSET_SEED}"
    --loss_fn            "${LOSS_FN}"
    --ema                "${EMA}"
    --aggregation        "${AGGREGATION}"
)

if [[ -n "${SELECTION_MANIFEST}" ]]; then
    CMD+=(--selection_manifest "${SELECTION_MANIFEST}")
fi

if [[ "${FILL_ZERO_FOR_UNROUTED}" == "1" ]]; then
    CMD+=(--fill_zero_for_unrouted)
fi

# Optional explicit layer list, e.g.:
#   LAYERS="4 5 6 7"
#   LAYERS="4,5,6,7"
#   LAYERS="17-32"
#   LAYERS="4,6-8,10"
if [[ -n "${LAYERS// }" ]]; then
    LAYERS_NORM="${LAYERS//,/ }"
    # shellcheck disable=SC2206
    LAYERS_TOKENS=(${LAYERS_NORM})
    LAYERS_LIST=()
    for token in "${LAYERS_TOKENS[@]}"; do
        if [[ "${token}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start="${BASH_REMATCH[1]}"
            end="${BASH_REMATCH[2]}"
            if (( start > end )); then
                echo "Invalid LAYERS range: ${token} (start > end)" >&2
                exit 1
            fi
            for ((i = start; i <= end; i++)); do
                LAYERS_LIST+=("${i}")
            done
        elif [[ "${token}" =~ ^[0-9]+$ ]]; then
            LAYERS_LIST+=("${token}")
        else
            echo "Invalid LAYERS token: ${token}" >&2
            echo "Supported format examples: 4,5,6 or 17-32 or 4,6-8,10" >&2
            exit 1
        fi
    done
    CMD+=(--layers "${LAYERS_LIST[@]}")
fi

# --force if you want to recompute the scores and overwrite the existing ones

CMD+=("${EXTRA_ARGS[@]}")

echo "Model      : ${MODEL_PATH}"
if [[ -n "${SELECTION_MANIFEST}" ]]; then
    echo "Dataset    : mixed (sample count and score-token quota loaded from manifest)"
    echo "Manifest   : ${SELECTION_MANIFEST} (sample count and score-token quota loaded from JSON)"
else
    echo "Dataset    : ${DATASET} (${NUM_SAMPLES} samples, ${TOKEN_PER_SAMPLE} tokens per sample)"
fi
echo "Layers     : ${LAYERS:-<all MoE layers>}"
echo "Output     : ${OUTPUT_DIR}"
echo "Aggregation: ${AGGREGATION}"
echo "2nd order  : ${SECOND_ORDER_IMPL}, expert group=${SECOND_ORDER_CHUNK_SIZE}, max=${SECOND_ORDER_MAX_CHUNK_SIZE}"
echo "Fill zero for unrouted: ${FILL_ZERO_FOR_UNROUTED}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "CMD: ${CMD[@]}"
echo ""
echo "Running command..."
echo ""
"${CMD[@]}"
