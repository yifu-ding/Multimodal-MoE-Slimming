"""Order measured counterexamples for visual selection; this is a post-hoc display aid."""

import argparse
import csv
from pathlib import Path

from scripts.analyze_beta095_counterexamples import write_csv


def select(directory):
    with (directory / "counterexamples.csv").open() as handle:
        pairs = [row for row in csv.DictReader(handle) if row["baseline"] == "first_order_abs"]
    with (directory / "beta_curves.csv").open() as handle:
        scanned = {int(row["expert"]) for row in csv.DictReader(handle)}
    candidates = []
    for pair in pairs:
        high, low = int(pair["expert_high"]), int(pair["expert_low"])
        truth_low, first_low = float(pair["truth_low"]), float(pair["first_low"])
        if truth_low <= 0 or first_low <= 0:
            continue
        increase = float(pair["truth_high"]) / truth_low - 1
        decrease = 1 - float(pair["first_high"]) / first_low
        candidates.append({**pair, "truth_increase_percent": 100 * increase,
                           "first_decrease_percent": 100 * decrease,
                           "balanced_display_score": min(increase, decrease),
                           "both_differences_at_least_20_percent": increase >= .2 and decrease >= .2,
                           "both_full_curves_available": high in scanned and low in scanned})
    candidates.sort(key=lambda row: (-row["balanced_display_score"], int(row["expert_high"]), int(row["expert_low"])))
    fields = list(candidates[0]) if candidates else [
        "expert_high", "expert_low", "truth_increase_percent", "first_decrease_percent",
        "balanced_display_score", "both_differences_at_least_20_percent", "both_full_curves_available",
    ]
    write_csv(directory / "plot_candidates.csv", candidates, fields)
    print(f"{directory}: {len(candidates)} absolute-score pairs; "
          f"{sum(row['both_differences_at_least_20_percent'] for row in candidates)} exceed 20% in both directions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    select(parser.parse_args().data_dir)
