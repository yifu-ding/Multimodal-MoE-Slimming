#!/usr/bin/env bash
set -euo pipefail

# Download the official Hugging Face snapshots for the video benchmarks to
# the roomier data5 filesystem. Benchmark names may be supplied as arguments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HF_DOWNLOAD_HOME="${HF_DOWNLOAD_HOME:-/home/data5/dyf/hf_cache}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm-maes}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

if (( $# == 0 )); then
    BENCHMARKS=(egoschema video-mmmu mvbench)
else
    BENCHMARKS=("$@")
fi

for benchmark in "${BENCHMARKS[@]}"; do
    case "${benchmark}" in
        egoschema|video-mmmu|mvbench) ;;
        *)
            echo "error: unsupported benchmark '${benchmark}'." >&2
            echo "supported: egoschema video-mmmu mvbench" >&2
            exit 2
            ;;
    esac
done

mkdir -p "${HF_DOWNLOAD_HOME}/hub" "${HF_DOWNLOAD_HOME}/datasets"

echo "HF_DOWNLOAD_HOME=${HF_DOWNLOAD_HOME}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
df -h "${HF_DOWNLOAD_HOME}"

conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    python "${REPO_ROOT}/download_hf_benchmarks.py" \
    --hf-home "${HF_DOWNLOAD_HOME}" \
    --benchmarks "${BENCHMARKS[@]}" \
    --resume-download

if [[ " ${BENCHMARKS[*]} " == *" mvbench "* ]]; then
    cat <<'EOF'

MVBench note:
  The official Hugging Face snapshot excludes the NTU RGB+D videos because
  of their license. The 200 fine_grained_pose questions remain unavailable
  until the required NTU videos are obtained manually from ROSE Lab.
EOF
fi
