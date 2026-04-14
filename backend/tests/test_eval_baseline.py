import tempfile
import unittest
from pathlib import Path

from evals.baseline import (
    build_baseline_snapshot,
    compare_aggregate_scores,
    load_baseline_snapshot,
    resolve_baseline_path,
    save_baseline_snapshot,
)


class EvalBaselineTests(unittest.TestCase):
    def test_default_baseline_path_uses_dataset_stem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            dataset_path = base_dir / "evals" / "datasets" / "docutalk_benchmark_v1.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text("[]", encoding="utf-8")

            self.assertIsNone(
                resolve_baseline_path(
                    base_dir=base_dir,
                    dataset_path=dataset_path,
                    baseline_arg=None,
                    create_default=False,
                )
            )

            expected_path = base_dir / "evals" / "baselines" / "docutalk_benchmark_v1.json"
            self.assertEqual(
                resolve_baseline_path(
                    base_dir=base_dir,
                    dataset_path=dataset_path,
                    baseline_arg=None,
                    create_default=True,
                ),
                expected_path.resolve(),
            )

    def test_explicit_relative_baseline_path_resolves_from_base_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            dataset_path = base_dir / "evals" / "datasets" / "docutalk_benchmark_v1.jsonl"

            resolved = resolve_baseline_path(
                base_dir=base_dir,
                dataset_path=dataset_path,
                baseline_arg="custom/baseline.json",
                create_default=False,
            )

            self.assertEqual(resolved, (base_dir / "custom" / "baseline.json").resolve())

    def test_snapshot_round_trip_and_metric_comparison(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            snapshot_path = temp_path / "baseline.json"
            dataset_path = temp_path / "evals" / "datasets" / "docutalk_benchmark_v1.jsonl"

            snapshot = build_baseline_snapshot(
                dataset_path=dataset_path,
                judge_provider="anthropic",
                judge_model="claude-test",
                metrics=["faithfulness", "answer_correctness"],
                aggregate_scores={
                    "faithfulness": 0.81,
                    "answer_correctness": 0.72,
                },
                sample_count=15,
            )
            save_baseline_snapshot(snapshot_path, snapshot)
            loaded = load_baseline_snapshot(snapshot_path)

            self.assertEqual(loaded["dataset_name"], "docutalk_benchmark_v1")
            self.assertEqual(loaded["sample_count"], 15)

            comparisons = compare_aggregate_scores(
                current_scores={
                    "faithfulness": 0.84,
                    "answer_correctness": 0.70,
                },
                baseline_scores=loaded["aggregate_scores"],
                metrics=["faithfulness", "answer_correctness", "context_recall"],
            )

            self.assertEqual(
                comparisons,
                [
                    {
                        "metric": "faithfulness",
                        "current": 0.84,
                        "baseline": 0.81,
                        "delta": 0.03,
                    },
                    {
                        "metric": "answer_correctness",
                        "current": 0.7,
                        "baseline": 0.72,
                        "delta": -0.02,
                    },
                    {
                        "metric": "context_recall",
                        "current": None,
                        "baseline": None,
                        "delta": None,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
