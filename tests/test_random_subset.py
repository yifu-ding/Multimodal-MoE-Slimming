import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).parents[1] / "lmms-eval/lmms_eval/evaluator_utils.py"
sys.path.insert(0, str(MODULE_PATH.parents[1]))
SPEC = importlib.util.spec_from_file_location("maes_evaluator_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_random_subset_doc_ids = MODULE.build_random_subset_doc_ids


class RandomSubsetTests(unittest.TestCase):
    def test_exact_reproducible_and_stratified(self):
        sizes = {"part_a": 300, "part_b": 300, "part_c": 300}
        groups = {name: "benchmark" for name in sizes}

        first = build_random_subset_doc_ids(sizes, groups, 0.5, 500, 42)
        second = build_random_subset_doc_ids(sizes, groups, 0.5, 500, 42)

        self.assertEqual(first, second)
        self.assertEqual(sum(map(len, first.values())), 500)
        self.assertEqual(sorted(map(len, first.values())), [166, 167, 167])
        self.assertTrue(all(len(ids) == len(set(ids)) for ids in first.values()))
        self.assertTrue(all(0 <= doc_id < sizes[name] for name, ids in first.items() for doc_id in ids))

    def test_uses_all_samples_below_minimum(self):
        selected = build_random_subset_doc_ids({"small": 218}, {"small": None}, 0.5, 500, 42)
        self.assertEqual(selected["small"], list(range(218)))

    def test_keeps_mme_pairs(self):
        selected = build_random_subset_doc_ids({"mme": 2374}, {"mme": None}, 0.5, 500, 42)["mme"]
        self.assertEqual(len(selected), 1188)
        self.assertTrue(all((doc_id + 1 if doc_id % 2 == 0 else doc_id - 1) in selected for doc_id in selected))


if __name__ == "__main__":
    unittest.main()
