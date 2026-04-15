from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


def resolve_baseline_path(
    *,
    base_dir: Path,
    dataset_path: Path,
    baseline_arg: str | None,
    create_default: bool,
) -> Path | None:
    if baseline_arg:
        candidate = Path(baseline_arg)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return candidate

    default_path = (base_dir / "evals" / "baselines" / f"{dataset_path.stem}.json").resolve()
    if create_default or default_path.exists():
        return default_path

    return None


def build_baseline_snapshot(
    *,
    dataset_path: Path,
    judge_provider: str,
    judge_model: str,
    metrics: Sequence[str],
    aggregate_scores: Mapping[str, float],
    sample_count: int,
) -> dict[str, object]:
    return {
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.stem,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "metrics": list(metrics),
        "sample_count": sample_count,
        "aggregate_scores": {
            metric: float(score)
            for metric, score in aggregate_scores.items()
        },
    }


def load_baseline_snapshot(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline_snapshot(path: Path, snapshot: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def compare_aggregate_scores(
    *,
    current_scores: Mapping[str, float],
    baseline_scores: Mapping[str, object],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []

    for metric in metrics:
        current_value = current_scores.get(metric)
        baseline_value = baseline_scores.get(metric)
        normalized_baseline = None if baseline_value is None else float(baseline_value)
        normalized_current = None if current_value is None else float(current_value)
        delta = None

        if normalized_current is not None and normalized_baseline is not None:
            delta = round(normalized_current - normalized_baseline, 6)

        comparisons.append(
            {
                "metric": metric,
                "current": normalized_current,
                "baseline": normalized_baseline,
                "delta": delta,
            }
        )

    return comparisons
