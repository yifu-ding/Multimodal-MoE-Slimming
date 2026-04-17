from pathlib import Path
import re
from datetime import datetime

base = Path("/home/dyf/code/distill/MoDES/results/prune_eval_p50/sweep_tasks-kimi-coco-rell2-fill1-0416-115847/logs")
log_root = base / "logs"
out_path = base / "summary-new.md"

logs = sorted(log_root.glob("stdout_*.log")) + sorted((log_root / "logs").glob("stdout_*.log"))

acc_re = re.compile(r"\[Run\] Accuracy:\s*([0-9.]+)\s*\(([^)]*)\)")
runidx_re = re.compile(r"^stdout_(\d{4})_")
modality_re = re.compile(r"_m([01])_")
header_patterns = {
    "inter_method": re.compile(r"^Inter\s*:\s*(.+)$", re.M),
    "intra_method": re.compile(r"^Intra\s*:\s*(.+)$", re.M),
    "intra_expert_metric": re.compile(r"^Metric\s*:\s*(.+)$", re.M),
    "smooth_fn": re.compile(r"^Smooth fn\s*:\s*(.+)$", re.M),
    "modality_text": re.compile(r"^Modality\s*:\s*(.+)$", re.M),
}

rows = []
for path in logs:
    text = path.read_text(errors="replace")
    name = path.name

    row = {
        "#": "—",
        "inter_method": "—",
        "intra_method": "—",
        "modality_aware": "—",
        "intra_expert_metric": "—",
        "smooth_fn": "—",
        "accuracy": "—",
        "detail": "—",
        "status": "no_accuracy_line",
        "log": path.relative_to(base).as_posix(),
        "_sort_key": (name, path.relative_to(base).as_posix()),
    }

    m = runidx_re.match(name)
    if m:
        row["#"] = str(int(m.group(1)))
        ts_match = re.search(r"^stdout_\d{4}_(\d{8}_\d{6})_", name)
        if ts_match:
            row["_sort_key"] = (ts_match.group(1), int(m.group(1)), row["log"])
    else:
        ts_match = re.search(r"^stdout_(\d{8}_\d{6})", name)
        if ts_match:
            row["_sort_key"] = (ts_match.group(1), 10**9, row["log"])

    mm = modality_re.search(name)
    if mm:
        row["modality_aware"] = mm.group(1)

    for key, pat in header_patterns.items():
        m = pat.search(text)
        if m:
            val = m.group(1).strip()
            if key == "modality_text":
                row["modality_aware"] = "1" if val == "text+visual" else "0" if val == "disabled" else val
            else:
                row[key] = val

    m = acc_re.search(text)
    if m:
        row["accuracy"] = m.group(1)
        row["detail"] = m.group(2)
        row["status"] = "ok"

    rows.append(row)

rows.sort(key=lambda r: r["_sort_key"])

with out_path.open("w", encoding="utf-8") as f:
    f.write("# Prune + GQA eval sweep summary\n\n")
    f.write("Auto-generated from existing log files under `logs/` and `logs/logs/`.\n\n")
    f.write(f"Started: {datetime.now().astimezone().isoformat(timespec='seconds')}\n\n")
    f.write("| # | inter_method | intra_method | modality_aware | intra_expert_metric | smooth_fn | accuracy | correct/total | status | log |\n")
    f.write("|---|--------------|--------------|----------------|---------------------|-----------|----------|---------------|--------|-----|\n")
    for r in rows:
        f.write(
            f"| {r['#']} | {r['inter_method']} | {r['intra_method']} | {r['modality_aware']} | "
            f"{r['intra_expert_metric']} | {r['smooth_fn']} | {r['accuracy']} | {r['detail']} | "
            f"{r['status']} | `{r['log']}` |\n"
        )
