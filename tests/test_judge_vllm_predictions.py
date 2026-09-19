import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "judge_vllm_predictions",
    ROOT / "scripts/judge_vllm_predictions.py",
)
JUDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGE)


class BoundedPredictionTest(unittest.TestCase):
    def test_mmvet_keeps_prediction_start(self):
        prediction = "start-" + "x" * 20 + "-final"
        bounded = JUDGE.bounded_prediction(prediction, 10, keep_tail=False)
        self.assertTrue(bounded.startswith(prediction[:10]))
        self.assertTrue(bounded.endswith("[Later response omitted]"))
        self.assertNotIn("-final", bounded)

    def test_videommmu_keeps_prediction_end(self):
        prediction = "start-" + "x" * 20 + "-final"
        bounded = JUDGE.bounded_prediction(prediction, 10, keep_tail=True)
        self.assertTrue(bounded.startswith("[Earlier reasoning omitted]"))
        self.assertTrue(bounded.endswith(prediction[-10:]))
        self.assertNotIn("start-", bounded)

    def test_zero_disables_truncation(self):
        self.assertEqual(
            JUDGE.bounded_prediction("answer", 0, keep_tail=False),
            "answer",
        )

    def test_negative_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            JUDGE.bounded_prediction("answer", -1, keep_tail=False)

    def test_mmvet_request_is_bounded_and_requires_numeric_output(self):
        sample = {
            "doc_id": 1,
            "input": "question",
            "target": "target",
            "filtered_resps": "start-" + "x" * 30 + "-final",
        }
        args = SimpleNamespace(mmvet_prediction_max_chars=10, model="judge")
        with patch.object(JUDGE, "request_judge", return_value="0.6") as request:
            record = JUDGE.mmvet_record(sample, args, Path("samples.jsonl"))
        prompt = request.call_args.args[0]
        self.assertIn("start-xxxx", prompt)
        self.assertNotIn("-final", prompt)
        self.assertIn("Return only one score", prompt)
        self.assertEqual(record["score"], 0.6)


if __name__ == "__main__":
    unittest.main()
