#!/usr/bin/env bash
set -euo pipefail

# Run this script on your local machine.
# It pulls A3 plotting code plus lightweight redraw data from server alias H20_new.

REMOTE_HOST="H20_new"
REMOTE_ROOT="/home/dyf/code/distill/MAES"
LOCAL_ROOT="${1:-./MAES_a3_sync}"

mkdir -p "${LOCAL_ROOT}/observations/a3"
mkdir -p "${LOCAL_ROOT}/observations/a3/split-hidden"

FILES=(
  "observations/a3/plot_hidden_tsne.py"
  "observations/a3/split_teacher_hidden_by_tsne.py"
  "observations/a3/legacy-split_teacher_hidden_by_tsne_density.py"
  "observations/a3/split_teacher_hidden_by_tsne_2d_select.py"
  "observations/a3/redraw_from_plot_data.py"
  "observations/a3/scp_plot_artifacts_from_h20.sh"
  "observations/a3/split-hidden/gqa_1024_tsne2d_per_modality_fixed100_plot_data.pt"
  "observations/a3/split-hidden/gqa_1024_tsne2d_per_modality_fixed100_summary.json"
  "observations/a3/split-hidden/gqa_1024_tsne2d_per_modality_fixed100_per_modality_tsne.png"
  "observations/a3/split-hidden/gqa_1024_tsne2d_per_modality_core20_out20_plot_data.pt"
  "observations/a3/split-hidden/gqa_1024_tsne2d_per_modality_core20_out20_summary.json"
  "observations/a3/split-hidden/gqa_1024_tsne2d_per_modality_core20_out20_per_modality_tsne.png"
  "observations/a3/split-hidden/gqa_1024_tsne2d_core20_out20_summary.json"
  "observations/a3/split-hidden/gqa_1024_tsne2d_core20_out20_core_outlier_tsne.png"
)

for relpath in "${FILES[@]}"; do
  mkdir -p "${LOCAL_ROOT}/$(dirname "${relpath}")"
  scp "${REMOTE_HOST}:${REMOTE_ROOT}/${relpath}" "${LOCAL_ROOT}/${relpath}"
done

echo "Synced A3 plotting files to: ${LOCAL_ROOT}"
