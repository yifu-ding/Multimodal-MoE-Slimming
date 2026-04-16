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

source scripts/select_least_used_gpu.sh # 自动选择显存使用量最少的 gpu

PREFIX="${PREFIX:-$(pwd)}"
export PYTHONPATH="${PREFIX}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-VL-A3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
# MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF}"

DATASET="${DATASET:-gqa}"
NUM_SAMPLES="${NUM_SAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
START_IDX="${START_IDX:-0}"
SUBSET_SEED="${SUBSET_SEED:-42}"
LAYERS="${LAYERS:-}"  # 默认不传值，全部层calibration
# LAYERS="${LAYERS:-1-6}"
# LAYERS="${LAYERS:-7-13}"
# LAYERS="${LAYERS:-14-20}"
# LAYERS="${LAYERS:-20-26}"

EMA="${EMA:-0.9}"
FILL_ZERO_FOR_UNROUTED="${FILL_ZERO_FOR_UNROUTED:-1}"

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
    *)
        echo "Unsupported DATASET=${DATASET}. Supported values: gqa, coco, VMMMU (video_mmmu)." >&2
        exit 1
        ;;
esac

MODEL_TAG_RAW="${MODEL_PATH##*/}"
MODEL_TAG="${MODEL_TAG_RAW,,}"
case "${MODEL_TAG}" in
    qwen3-vl-30b-a3b-instruct)
        MODEL_TAG="qwen3-vl-30b-a3b"
        ;;
    kimi-vl-a3b-instruct)
        MODEL_TAG="kimi-vl-a3b"
        ;;
    internvl-3.5-gpt-oss-20b-a4b-preview-hf)
        MODEL_TAG="internvl-3.5-20b-a4b"
        ;;
    *)
        MODEL_TAG="${MODEL_TAG%-instruct}"
        ;;
esac

# OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/prune/scores/${MODEL_TAG}_${DATASET_TAG}-second_order}"
OUTPUT_DIR="${OUTPUT_DIR:-${PREFIX}/storage/prune/scores/${MODEL_TAG}_${DATASET_TAG}-rell2-fill${FILL_ZERO_FOR_UNROUTED}-$(date +%m%d-%H%M%S)}"

EXTRA_ARGS=("$@")

CMD=(
    python -m src.calibration.collect_scores_main
    --model_name_or_path "${MODEL_PATH}"
    --output_dir         "${OUTPUT_DIR}"
    --dataset            "${DATASET}"
    --num_samples        "${NUM_SAMPLES}"
    --batch_size         "${BATCH_SIZE}"
    --start_idx          "${START_IDX}"
    --subset_seed        "${SUBSET_SEED}"
    --ema                "${EMA}"
)

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
echo "Dataset    : ${DATASET} (${NUM_SAMPLES} samples)"
echo "Layers     : ${LAYERS:-<all MoE layers>}"
echo "Output     : ${OUTPUT_DIR}"
echo "Fill zero for unrouted: ${FILL_ZERO_FOR_UNROUTED}"
echo "GPU        : ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "CMD: ${CMD[@]}"
echo ""
echo "Running command..."
echo ""
"${CMD[@]}"
