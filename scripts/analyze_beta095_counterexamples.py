"""Compare a displaced gradient ranking with identity-point removal sensitivity."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr


SCORE_KEYS = ("first_order_signed", "first_order_abs", "hessian_half_at_identity")
PAIR_FIELDS = (
    "baseline", "expert_high", "expert_low", "truth_high", "truth_low", "truth_gap",
    "first_high", "first_low", "hessian_high", "hessian_low",
    "hessian_relative_error_high", "hessian_relative_error_low",
)


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def representative_experts(rows):
    ordered = sorted(rows, key=lambda row: (row["true_removal_at_identity"], row["expert"]))
    if len(ordered) < 9:
        raise ValueError("Nine representatives require at least nine active experts.")
    selected = []
    for group, indices in zip(("low", "medium", "high"), np.array_split(np.arange(len(ordered)), 3)):
        used = set()
        for quantile in (0.25, 0.5, 0.75):
            target = round(quantile * (len(indices) - 1))
            position = min((p for p in range(len(indices)) if p not in used), key=lambda p: (abs(p - target), p))
            used.add(position)
            selected.append({"expert": ordered[indices[position]]["expert"], "group": group, "quantile": quantile})
    return selected


def analyze(rows, *, repeat_loss_error, repeat_gradient_error, hessian_rtol):
    if not rows:
        raise ValueError("No active experts.")
    truth = np.array([r["true_removal_at_identity"] for r in rows])
    scores = {key: np.array([r[key] for r in rows]) for key in SCORE_KEYS}
    if not all(np.isfinite(values).all() for values in (truth, *scores.values())):
        raise ValueError("Nonfinite measured values must be investigated before ranking.")
    # Predeclared numerical floor, independent of the number of counterexamples.
    floor = 32 * np.finfo(np.float32).eps
    truth_tol = max(8 * repeat_loss_error, floor * float(np.max(np.abs(truth))))
    tolerances = {key: max(floor * float(np.max(np.abs(value))),
                           8 * repeat_gradient_error if key.startswith("first_") else 0.0)
                  for key, value in scores.items()}
    rel_error = np.abs(scores[SCORE_KEYS[-1]] - truth) / np.maximum(np.abs(truth), max(truth_tol, np.finfo(float).tiny))
    metrics, pairs = {}, []
    for key, value in scores.items():
        correct = inverted = tied = excluded = 0
        for i in range(len(rows)):
            for j in range(i):
                if abs(truth[i] - truth[j]) <= truth_tol:
                    excluded += 1
                    continue
                hi, lo = (i, j) if truth[i] > truth[j] else (j, i)
                gap = value[hi] - value[lo]
                if gap > tolerances[key]:
                    correct += 1
                elif gap < -tolerances[key]:
                    inverted += 1
                    h = scores[SCORE_KEYS[-1]]
                    if key.startswith("first_") and h[hi] > h[lo] + tolerances[SCORE_KEYS[-1]] and max(rel_error[hi], rel_error[lo]) <= hessian_rtol:
                        pairs.append(dict(zip(PAIR_FIELDS, (
                            key, rows[hi]["expert"], rows[lo]["expert"], float(truth[hi]), float(truth[lo]),
                            float(truth[hi] - truth[lo]), float(value[hi]), float(value[lo]),
                            float(h[hi]), float(h[lo]), float(rel_error[hi]), float(rel_error[lo]),
                        ))))
                else:
                    tied += 1
        count = correct + inverted + tied
        rho = float(spearmanr(truth, value).statistic) if np.ptp(value) and np.ptp(truth) else None
        tau = float(kendalltau(truth, value).statistic) if np.ptp(value) and np.ptp(truth) else None
        metrics[key] = {"spearman": rho, "kendall": tau, "correct_pairs": correct,
                        "inverted_pairs": inverted, "predicted_ties": tied,
                        "truth_ties_excluded": excluded, "eligible_pairs": count,
                        "inversion_rate": inverted / count if count else None}
    error = scores[SCORE_KEYS[-1]] - truth
    metrics[SCORE_KEYS[-1]].update(mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                                  median_relative_error=float(np.median(rel_error)), max_relative_error=float(np.max(rel_error)))
    pairs.sort(key=lambda p: (-p["truth_gap"], p["expert_high"], p["expert_low"], p["baseline"]))
    representatives = representative_experts(rows)
    # Keep the nine representatives intact; add endpoints of three unique pairs.
    display_pairs, seen = [], set()
    for pair in pairs:
        identity = (pair["expert_high"], pair["expert_low"])
        if identity not in seen:
            display_pairs.append(pair)
            seen.add(identity)
        if len(display_pairs) == 3:
            break
    summary = {
        "target": "single removal from all-ones background; gradient is a ranking proxy from all-0.95 background",
        "num_experts": len(rows), "negative_removal_count": int(np.sum(truth < 0)),
        "gradient_range": [min(r["gradient_at_work"] for r in rows), max(r["gradient_at_work"] for r in rows)],
        "truth_tolerance": truth_tol, "score_tolerances": tolerances,
        "tolerance_rule": "max(8 * measured repeat discrepancy, 32 * float32 epsilon * maximum absolute score)",
        "hessian_match_rtol": hessian_rtol, "metrics": metrics,
        "counterexample_counts": {key: sum(p["baseline"] == key for p in pairs) for key in SCORE_KEYS[:2]},
        "representatives": representatives, "display_pairs": display_pairs,
        "selection_rule": "truth ascending, expert-ID ties, array_split thirds, distinct nearest within-group P25/P50/P75; extra top three unique counterexample pairs by truth gap",
    }
    return summary, pairs


def export_analysis(output_dir, rows, **kwargs):
    summary, pairs = analyze(rows, **kwargs)
    write_csv(output_dir / "scores.csv", rows)
    write_csv(output_dir / "counterexamples.csv", pairs, PAIR_FIELDS)
    (output_dir / "statistics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
