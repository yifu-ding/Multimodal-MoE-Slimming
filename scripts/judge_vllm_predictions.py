#!/usr/bin/env python3
"""Score saved lmms-eval MMVet/MMBench predictions with a local judge."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests


MMVET_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.
gpt_query_prompt | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0"""

MMBENCH_PROMPT = """You match a model answer to one option of a multiple-choice question.
Return exactly one uppercase letter from A, B, C, D, or E. Return E only when the answer is meaningfully different from every listed option. Do not explain your answer.

Question and choices:
{question}

Model answer:
{prediction}

Your output:"""

_thread_local = threading.local()
_MMVET_INPUT_PREFIX = "First please perform reasoning, and think step by step to provide best answer to the following question:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True, help="Baseline RUN_DIR containing lmms-eval sample JSONL files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to PREDICTIONS_DIR/local_judge.")
    parser.add_argument("--tasks", default="mmvet,mmbench", help="Comma-separated subset of: mmvet,mmbench.")
    parser.add_argument("--api-base", default=os.getenv("JUDGE_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--model", default=os.getenv("MODEL_VERSION", "local-mm-judge"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Discard an existing per-sample judge file instead of resuming it.")
    return parser.parse_args()


def chat_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def request_judge(prompt: str, args: argparse.Namespace) -> str:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    error: Exception | None = None
    for attempt in range(args.retries):
        try:
            response = session().post(chat_url(args.api_base), headers=headers, json=payload, timeout=args.timeout)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if content and content.strip():
                return content.strip()
            raise ValueError("judge returned empty content")
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            error = exc
            if attempt + 1 < args.retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"judge request failed after {args.retries} attempts: {error}")


def prediction_text(sample: dict[str, Any]) -> str:
    value: Any = sample.get("filtered_resps", sample.get("resps", ""))
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return "" if value is None else str(value).strip()


def unwrap_metric(sample: dict[str, Any], key: str) -> dict[str, Any]:
    value = sample.get(key)
    return value if isinstance(value, dict) else {}


def mmvet_record(sample: dict[str, Any], args: argparse.Namespace, source: Path) -> dict[str, Any]:
    metric = unwrap_metric(sample, "gpt_eval_score")
    question = str(metric.get("question") or sample.get("input") or "").strip()
    if question.startswith(_MMVET_INPUT_PREFIX):
        question = question[len(_MMVET_INPUT_PREFIX) :].strip()
    target = str(metric.get("gt_answer") or sample.get("target") or "").strip()
    prediction = str(metric.get("pred_answer") or prediction_text(sample)).strip()
    if not question or not target:
        raise ValueError("MMVet sample is missing its question or target")
    prompt = f"{MMVET_PROMPT}\n{question} | {target.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> ')} | {prediction} |"
    raw_judgment = request_judge(prompt, args)
    match = re.search(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", raw_judgment)
    if not match:
        raise ValueError(f"invalid MMVet judge response: {raw_judgment!r}")
    score = float(match.group(1))
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"MMVet score outside [0, 1]: {score}")
    return {
        "task": "mmvet",
        "doc_id": sample.get("doc_id"),
        "question": question,
        "target": target,
        "prediction": prediction,
        "score": score,
        "raw_judgment": raw_judgment,
        "judge_model": args.model,
        "source_file": str(source),
    }


def explicit_choice(text: str) -> str | None:
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE).strip()
    patterns = (
        r"^(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)?\s*\(?([A-E])\)?(?:\b|[.)])",
        r"^\(?([A-E])\)?(?:\b|[.)])",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def mmbench_record(sample: dict[str, Any], args: argparse.Namespace, source: Path) -> dict[str, Any]:
    metric = unwrap_metric(sample, "gpt_eval_score")
    prediction = str(metric.get("prediction") or prediction_text(sample)).strip()
    target = str(metric.get("answer") or sample.get("target") or "").strip().upper()
    question = str(metric.get("question") or sample.get("input") or "").strip()
    if target not in set("ABCDE"):
        raise ValueError(f"invalid MMBench target: {target!r}")

    choices = []
    for letter in "ABCDE":
        value = metric.get(letter)
        if value is not None and str(value).lower() != "nan":
            choices.append(f"{letter}. {value}")
    if choices:
        question = f"{question}\n" + "\n".join(choices)

    choice = explicit_choice(prediction)
    method = "static"
    raw_judgment = ""
    if choice is None:
        raw_judgment = request_judge(MMBENCH_PROMPT.format(question=question, prediction=prediction), args)
        choice = explicit_choice(raw_judgment)
        method = "judge"
    if choice is None:
        raise ValueError(f"invalid MMBench judge response: {raw_judgment!r}")

    return {
        "task": "mmbench",
        "doc_id": sample.get("doc_id"),
        "index": metric.get("index", sample.get("doc_id")),
        "question": question,
        "target": target,
        "prediction": prediction,
        "predicted_choice": choice,
        "score": float(choice == target),
        "method": method,
        "raw_judgment": raw_judgment,
        "judge_model": args.model if method == "judge" else None,
        "source_file": str(source),
    }


def find_sources(root: Path, task: str) -> list[Path]:
    patterns = {
        "mmvet": "*_samples_mmvet.jsonl",
        "mmbench": "*_samples_mmbench_en_dev*.jsonl",
    }
    return sorted(path for path in root.rglob(patterns[task]) if "local_judge" not in path.parts)


def load_samples(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    samples = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    samples.append((path, json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return samples


def result_key(record: dict[str, Any]) -> str:
    return f"{record.get('source_file', '')}\0{record.get('doc_id')}"


def score_task(task: str, sources: list[Path], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_path = output_dir / f"{task}_judged.jsonl"
    if args.overwrite and output_path.exists():
        output_path.unlink()

    existing: list[dict[str, Any]] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
    # Failed rows are deliberately omitted here so a rerun retries them. Also
    # collapse any duplicate successful rows left by an interrupted rerun.
    successful_by_key = {
        result_key(record): record
        for record in existing
        if isinstance(record.get("score"), (int, float))
    }
    existing = list(successful_by_key.values())
    completed = set(successful_by_key)
    if output_path.exists():
        with output_path.open("w", encoding="utf-8") as handle:
            for record in existing:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_samples = load_samples(sources)
    pending = []
    for source, sample in source_samples:
        key = f"{source}\0{sample.get('doc_id')}"
        if key not in completed:
            pending.append((source, sample))

    scorer = mmvet_record if task == "mmvet" else mmbench_record
    failures = 0
    with output_path.open("a", encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(scorer, sample, args, source): (source, sample) for source, sample in pending}
            for index, future in enumerate(concurrent.futures.as_completed(future_map), 1):
                source, sample = future_map[future]
                try:
                    record = future.result()
                except Exception as exc:
                    failures += 1
                    record = {
                        "task": task,
                        "doc_id": sample.get("doc_id"),
                        "source_file": str(source),
                        "error": str(exc),
                    }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                if index % 25 == 0 or index == len(pending):
                    print(f"[{task}] completed {index}/{len(pending)} new samples", flush=True)

    with output_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    valid = [record for record in records if isinstance(record.get("score"), (int, float))]
    score = (sum(record["score"] for record in valid) / len(valid) * 100.0) if valid else None
    num_groups = None
    if task == "mmbench" and valid:
        # Match lmms-eval's MMBench circular evaluation: one original question
        # is correct only when every rotated-option variant is correct.
        indexed_records = []
        for record in valid:
            try:
                indexed_records.append((int(record["index"]), record))
            except (KeyError, TypeError, ValueError):
                indexed_records = []
                break
        if indexed_records:
            base_indices = sorted(index for index, _ in indexed_records if index < 1_000_000)
            grouped_hits = []
            for base_index in base_indices:
                group = [record for index, record in indexed_records if index % 1_000_000 == base_index]
                if group:
                    grouped_hits.append(float(all(record["score"] == 1.0 for record in group)))
            if grouped_hits:
                score = sum(grouped_hits) / len(grouped_hits) * 100.0
                num_groups = len(grouped_hits)
    summary = {
        "task": task,
        "score": score,
        "num_scored": len(valid),
        "num_failed": len(records) - len(valid),
        "num_source_samples": len(source_samples),
        "judge_model": args.model,
        "source_files": [str(path) for path in sources],
        "per_sample_output": str(output_path),
    }
    if task == "mmbench":
        summary["num_static"] = sum(record.get("method") == "static" for record in valid)
        summary["num_judged"] = sum(record.get("method") == "judge" for record in valid)
        summary["num_question_groups"] = num_groups
    summary_path = output_dir / f"{task}_summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary_path, summary_path)
    if failures:
        print(f"warning: {failures} new {task} samples failed; rerun to retry them", file=sys.stderr)
    return summary


def check_server(args: argparse.Namespace) -> None:
    models_url = args.api_base.rstrip("/")
    if models_url.endswith("/chat/completions"):
        models_url = models_url[: -len("/chat/completions")]
    response = requests.get(f"{models_url}/models", timeout=min(args.timeout, 15.0))
    response.raise_for_status()
    available = [item.get("id") for item in response.json().get("data", [])]
    if args.model not in available:
        raise RuntimeError(f"judge model {args.model!r} is not served; available models: {available}")


def main() -> int:
    args = parse_args()
    root = args.predictions_dir.resolve()
    if not root.is_dir():
        print(f"error: predictions directory does not exist: {root}", file=sys.stderr)
        return 2
    selected = [item.strip().lower() for item in args.tasks.split(",") if item.strip()]
    unknown = sorted(set(selected) - {"mmvet", "mmbench"})
    if unknown:
        print(f"error: unknown tasks: {', '.join(unknown)}", file=sys.stderr)
        return 2

    task_sources = {task: find_sources(root, task) for task in selected}
    missing = [task for task, sources in task_sources.items() if not sources]
    if missing:
        print(f"error: no prediction JSONL found for: {', '.join(missing)} under {root}", file=sys.stderr)
        return 2

    output_dir = (args.output_dir or root / "local_judge").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    check_server(args)

    summaries = []
    for task in selected:
        summaries.append(score_task(task, task_sources[task], output_dir, args))
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0 if all(summary["num_failed"] == 0 for summary in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
