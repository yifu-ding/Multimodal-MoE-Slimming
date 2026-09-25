#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
completed="$(find artifacts/efficiency_batch512_three_models/runs -path '*/bs_512/COMPLETE' -type f 2>/dev/null | wc -l)"
failed="$(find artifacts/efficiency_batch512_three_models/runs -path '*/bs_512/FAILED' -type f 2>/dev/null | wc -l)"
latest="$(find artifacts/efficiency_batch512_three_models/runs -path '*/bs_512/runner.log' -type f -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)"
[[ -n "${latest}" ]] || latest=waiting
printf 'batch512 p0=3/3 new=%s/24 failed=%s current=%s\n' "${completed}" "${failed}" "${latest#artifacts/efficiency_batch512_three_models/runs/}"
