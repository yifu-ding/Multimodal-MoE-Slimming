"""Validate a plan against its source scores; build it only when absent."""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.vllm_ep4_plan import validate_ep4_plan

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--scores", required=True)
parser.add_argument("--layers", required=True)
parser.add_argument("--ratio", type=float, default=0.5)
args = parser.parse_args()
scores = Path(args.scores).resolve()
subprocess.run([sys.executable, str(ROOT / "scripts/validate_scores_artifact.py"),
                "--scores", str(scores), "--expected-layers", args.layers], check=True)
plan_path = ROOT / f"runtime/ep4_plans/{args.tag}-p{round(args.ratio*100)}-sparse-tier-v2.pt"
if not plan_path.exists():
    subprocess.run([sys.executable, str(ROOT / "scripts/build_ep4_pruning_plan.py"),
                    "--scores", str(scores), "--output", str(plan_path),
                    "--model", args.model, "--prune-ratio", str(args.ratio)], check=True)
plan = torch.load(plan_path, map_location="cpu", weights_only=False)
validate_ep4_plan(plan)
if plan.get("source_scores_sha256") != hashlib.sha256(scores.read_bytes()).hexdigest():
    raise RuntimeError(f"Plan/source hash mismatch: {plan_path}; preserve and Debug before reuse")
if plan.get("model") != args.model or abs(float(plan.get("pruning_config", {}).get("prune_ratio", -1)) - args.ratio) > 1e-9:
    raise RuntimeError(f"Plan model/prune ratio mismatch: {plan_path}")
print(f"VALID PLAN {plan_path}")
