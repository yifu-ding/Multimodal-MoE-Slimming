import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lmms-eval"))

from lmms_eval.api.instance import Instance
from lmms_eval.caching.response_cache import ResponseCache


def make_request(doc_id: int) -> Instance:
    return Instance(
        request_type="generate_until",
        arguments=(f"prompt-{doc_id}", {"temperature": 0}),
        idx=0,
        metadata={"task": "write_through_test", "doc_id": doc_id, "repeats": 1},
    )


class FakeLM:
    batch_size_per_gpu = 1

    def __init__(self, fail_doc_id=None):
        self.fail_doc_id = fail_doc_id
        self.calls = []

    def generate_until(self, requests):
        doc_ids = [request.doc_id for request in requests]
        self.calls.extend(doc_ids)
        if self.fail_doc_id in doc_ids:
            raise RuntimeError("synthetic interruption")
        return [f"response-{doc_id}" for doc_id in doc_ids]


class ResponseCacheWriteThroughTest(unittest.TestCase):
    def test_pre_fix_runtime_fingerprint_remains_readable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            run_dir = cache_root / "runs" / "legacy-runtime"
            run_dir.mkdir(parents=True)
            legacy_cache = ResponseCache(
                str(run_dir / "rank_0.db"),
                str(run_dir / "rank_0.audit.jsonl"),
                model_fingerprint='fake-model|{"gpu_memory_utilization":0.7,"temperature":0}',
                eval_version="test-version",
            )
            request = make_request(0)
            legacy_cache.execute(FakeLM(), "generate_until", [request])
            legacy_cache.close()

            old_env = {
                key: os.environ.get(key)
                for key in ("LMMS_CACHE_RUN_ID", "LMMS_CACHE_FINGERPRINT_SALT")
            }
            os.environ["LMMS_CACHE_RUN_ID"] = "legacy-runtime"
            os.environ["LMMS_CACHE_FINGERPRINT_SALT"] = "plan-a"
            try:
                with patch(
                    "lmms_eval.utils.get_lmms_eval_cache_version",
                    return_value="test-version",
                ):
                    resumed_cache = ResponseCache.create(
                        str(cache_root),
                        model="fake-model",
                        model_args="temperature=0,gpu_memory_utilization=0.7",
                    )
                resumed_model = FakeLM()
                self.assertEqual(
                    resumed_cache.execute(resumed_model, "generate_until", [request]),
                    ["response-0"],
                )
                self.assertEqual(resumed_model.calls, [])
                resumed_cache.finalize(success=False)
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_runtime_memory_settings_share_cache_but_semantic_salt_does_not(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            old_env = {
                key: os.environ.get(key)
                for key in ("LMMS_CACHE_RUN_ID", "LMMS_CACHE_FINGERPRINT_SALT")
            }
            os.environ["LMMS_CACHE_RUN_ID"] = "runtime-settings"
            try:
                request = make_request(0)
                os.environ["LMMS_CACHE_FINGERPRINT_SALT"] = "plan-a"
                first_cache = ResponseCache.create(
                    str(cache_root),
                    model="fake-model",
                    model_args="temperature=0,gpu_memory_utilization=0.85,max_num_batched_tokens=1024",
                )
                first_model = FakeLM()
                first_cache.execute(first_model, "generate_until", [request])
                first_cache.finalize(success=False)

                resumed_cache = ResponseCache.create(
                    str(cache_root),
                    model="fake-model",
                    model_args="temperature=0,gpu_memory_utilization=0.70,max_num_batched_tokens=2048",
                )
                resumed_model = FakeLM()
                self.assertEqual(
                    resumed_cache.execute(resumed_model, "generate_until", [request]),
                    ["response-0"],
                )
                self.assertEqual(resumed_model.calls, [])
                resumed_cache.finalize(success=False)

                os.environ["LMMS_CACHE_FINGERPRINT_SALT"] = "plan-b"
                isolated_cache = ResponseCache.create(
                    str(cache_root),
                    model="fake-model",
                    model_args="temperature=0,gpu_memory_utilization=0.70,max_num_batched_tokens=2048",
                )
                isolated_model = FakeLM()
                isolated_cache.execute(isolated_model, "generate_until", [request])
                self.assertEqual(isolated_model.calls, [0])
                isolated_cache.finalize(success=False)
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_interrupted_run_resumes_from_persisted_responses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            watchdog_dir = Path(temp_dir) / "watchdog"
            old_env = {
                key: os.environ.get(key)
                for key in (
                    "LMMS_CACHE_RUN_ID",
                    "LMMS_CACHE_WRITE_THROUGH_BATCH_SIZE",
                    "LMMS_WATCHDOG_DIR",
                )
            }
            os.environ["LMMS_CACHE_RUN_ID"] = "stable-test-run"
            os.environ["LMMS_CACHE_WRITE_THROUGH_BATCH_SIZE"] = "1"
            os.environ["LMMS_WATCHDOG_DIR"] = str(watchdog_dir)
            try:
                requests = [make_request(doc_id) for doc_id in range(4)]
                cache = ResponseCache.create(
                    str(cache_root),
                    model="fake-model",
                    model_args="temperature=0",
                )
                first_model = FakeLM(fail_doc_id=2)
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    cache.execute(first_model, "generate_until", requests)
                cache.finalize(success=False)

                shard = cache_root / "runs" / "stable-test-run" / "rank_0.db"
                with sqlite3.connect(shard) as connection:
                    count = connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
                self.assertEqual(count, 2)

                resumed_cache = ResponseCache.create(
                    str(cache_root),
                    model="fake-model",
                    model_args="temperature=0",
                )
                resumed_model = FakeLM()
                responses = resumed_cache.execute(
                    resumed_model,
                    "generate_until",
                    [make_request(doc_id) for doc_id in range(4)],
                )
                self.assertEqual(resumed_model.calls, [2, 3])
                self.assertEqual(
                    responses,
                    [f"response-{doc_id}" for doc_id in range(4)],
                )
                resumed_cache.finalize(success=True)
                self.assertTrue((watchdog_dir / "response_cache.json").is_file())
            finally:
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
