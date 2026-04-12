#!/usr/bin/env bash
set -euo pipefail

# Download evaluation datasets for calibration & benchmarking.
#
# Calibration datasets (used by collect_scores):
#   - GQA
#   - COCO-Caption2017
#   - VideoMMMU
#
# Image understanding benchmarks:
#   - TextVQA (val)          文字视觉问答
#   - ChartQA                图表理解
#   - MMStar                 多模态综合评测
#   - MMBench (dev, EN)      全面多模态评测
#   - MMVet                  多模态综合能力
#   - MME                    多模态全面评测
#   - RealWorldQA            真实场景视觉问答
#   - COCO2017-Cap (val)     图像描述，CIDEr 指标
#
# Video understanding benchmarks:
#   - MVBench               多模态视频理解
#   - EgoSchema             长视频第一人称理解
#   - VideoMME              多模态视频分析
#   - LongVideoBench (val)  长上下文视频语言理解
#   - Video-MMMU            多学科专业视频知识获取
#
# Usage:
#   conda activate modes
#   bash scripts/download_datasets.sh                        # download all
#   BENCHMARKS="gqa coco textvqa" bash scripts/download_datasets.sh  # selected only

if ! command -v hf >/dev/null 2>&1; then
    echo "error: 'hf' command not found. Activate the 'modes' environment first." >&2
    exit 1
fi

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
DATASET_ROOT="${HF_HOME}/datasets"
mkdir -p "${DATASET_ROOT}"

# repo_id -> local_dir_name
declare -A DATASET_MAP=(
    # --- calibration ---
    ["gqa"]="lmms-lab/GQA|GQA"
    ["coco"]="lmms-lab/COCO-Caption2017|COCO-Caption2017"
    ["video-mmmu"]="lmms-lab/VideoMMMU|VideoMMMU"
    # --- image understanding ---
    ["textvqa"]="lmms-lab/textvqa|textvqa"
    ["chartqa"]="lmms-lab/ChartQA|ChartQA"
    ["mmstar"]="Lin-Chen/MMStar|MMStar"
    ["mmbench"]="lmms-lab/MMBench|MMBench"
    ["mmvet"]="lmms-lab/MMVet|MMVet"
    ["mme"]="lmms-lab/MME|MME"
    ["realworldqa"]="lmms-lab/RealWorldQA|RealWorldQA"
    # --- video understanding ---
    ["mvbench"]="OpenGVLab/MVBench|MVBench"
    ["egoschema"]="lmms-lab/egoschema|egoschema"
    ["videomme"]="lmms-lab/Video-MME|Video-MME"
    ["longvideobench"]="longvideobench/LongVideoBench|LongVideoBench"
)

ALL_KEYS=(
    gqa coco video-mmmu
    textvqa chartqa mmstar mmbench mmvet mme realworldqa
    mvbench egoschema videomme longvideobench
)

if [[ -n "${BENCHMARKS:-}" ]]; then
    IFS=' ' read -ra SELECTED <<< "${BENCHMARKS}"
else
    SELECTED=("${ALL_KEYS[@]}")
fi

echo "Download root: ${DATASET_ROOT}"
echo "Benchmarks:    ${SELECTED[*]}"
echo ""

for key in "${SELECTED[@]}"; do
    entry="${DATASET_MAP[${key}]:-}"
    if [[ -z "${entry}" ]]; then
        echo "warning: unknown benchmark '${key}', skipping." >&2
        continue
    fi
    repo_id="${entry%%|*}"
    local_name="${entry##*|}"
    target_dir="${DATASET_ROOT}/${local_name}"
    echo "==> [${key}] Downloading ${repo_id}"
    mkdir -p "${target_dir}"
    hf download --repo-type dataset "${repo_id}" --local-dir "${target_dir}"
done

cat <<EOF

Downloads finished.
Datasets are under:
  ${DATASET_ROOT}
EOF
