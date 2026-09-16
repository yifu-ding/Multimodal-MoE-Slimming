"""Read aggregate campaign artifacts; write explicit active-stage metadata."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "artifacts/ours-active-stage.json"
TASKS = "gqa coco2017_cap_val_local textvqa_val chartqa mmstar mmbench_en_dev_static_local mmvet mme realworldqa videomme longvideobench_val_v video_mmmu_local egoschema_subset_local mvbench_available_3800".split()
MODELS = ["qwen3-vl-30b-a3b", "kimi", "internvl3_5-30b-a3b"]


def run_status(model, ratio):
    root = ROOT / "results/vllm_ours" / model / f"ep4-{ratio}-full"
    complete = []
    for task in TASKS:
        names = [task, "videomme_qwen3_vllm"] if task == "videomme" else [task]
        markers = [root / "status" / f"{name}.complete" for name in names]
        if not any(p.is_file() and "signature=" in p.read_text() for p in markers):
            continue
        files = [p for name in names for p in (root / "tasks" / name).rglob("*_results.json")]
        if not files:
            continue
        try:
            payload = json.loads(max(files, key=lambda p: p.stat().st_mtime).read_text())
            if payload.get("results"):
                complete.append(task)
        except (OSError, ValueError):
            pass
    judges = []
    for task in ["mmvet", "mmbench", "video_mmmu"]:
        try:
            data = json.loads((root / "local_judge" / f"{task}_summary.json").read_text())
            if (data.get("num_failed") == 0 and data.get("num_source_samples", 0) > 0
                    and data.get("num_scored") == data["num_source_samples"]):
                judges.append(task)
        except (OSError, ValueError):
            pass
    return {"model": model, "ratio": ratio, "benchmarks": len(complete), "total": 14,
            "judge": len(judges), "missing": [t for t in TASKS if t not in complete],
            "complete": len(complete) == 14 and len(judges) == 3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "summary", "stage"])
    parser.add_argument("--model")
    parser.add_argument("--ratio", default="p50")
    parser.add_argument("--phase")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    if args.command == "stage":
        STATE.parent.mkdir(exist_ok=True)
        payload = dict(model=args.model, ratio=args.ratio, phase=args.phase,
                       run_dir=args.run_dir, started_at=time.time())
        temp = STATE.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(payload))
        temp.replace(STATE)
        return 0
    scopes = [(args.model, args.ratio)] if args.model else [
        ("internvl3_5-30b-a3b", "p30"), *[(m, "p50") for m in MODELS]]
    rows = [run_status(*scope) for scope in scopes]
    print("; ".join(f"{r['model']} {r['ratio']}={r['benchmarks']}/14, Judge={r['judge']}/3" for r in rows))
    return 0 if args.command == "summary" or all(r["complete"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
