from __future__ import annotations

import argparse
import json
import logging
import mimetypes
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
# RAGAS 0.4.x still validates metrics against the legacy Metric base class in evaluate(),
# so use the compatible implementations from the private legacy modules here.
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig

from config import BASE_DIR, Settings, get_settings
from evals.baseline import (
    build_baseline_snapshot,
    compare_aggregate_scores,
    load_baseline_snapshot,
    resolve_baseline_path,
    save_baseline_snapshot,
)
from evals.dataset import load_eval_dataset, resolve_document_path
from logging_config import configure_logging
from schemas import ChatRequest
from services.rag import RagService

LOGGER = logging.getLogger("docutalk.ragas")

DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision_with_reference",
    "answer_correctness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline RAGAS evaluation against a benchmark dataset for DocuTalk."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSONL or JSON dataset file.",
    )
    parser.add_argument(
        "--report-dir",
        default="evals/reports",
        help="Directory where CSV and JSON evaluation reports should be written.",
    )
    parser.add_argument(
        "--judge-provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="Provider for the RAGAS judge model.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model to use for RAGAS. Defaults to the app chat model.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated subset of metrics to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional RAGAS batch size.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout used by RAGAS during evaluation calls.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum worker count used by RAGAS.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Optional baseline snapshot JSON to compare against. "
            "If omitted, evals/baselines/<dataset_stem>.json is used when present."
        ),
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Write the current run's aggregate scores as the baseline snapshot.",
    )
    return parser.parse_args()


def create_eval_service(settings: Settings, temp_dir: Path) -> RagService:
    eval_settings = replace(
        settings,
        chroma_directory=temp_dir / "chroma_db",
        documents_registry_path=temp_dir / "documents.json",
    )
    return RagService(eval_settings)


def infer_content_type(path: Path) -> str:
    guessed_content_type, _ = mimetypes.guess_type(path.name)
    return guessed_content_type or "application/octet-stream"


def create_ragas_llm(settings: Settings, provider: str, model: str | None):
    judge_model = model or settings.chat_model

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic-based RAGAS evaluation.")
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        # RAGAS 0.4.x defaults to both temperature and top_p for Anthropic, but
        # current Claude models reject requests that set both together.
        llm = llm_factory(
            judge_model,
            provider="anthropic",
            client=client,
            temperature=0.01,
            max_tokens=1024,
            top_p=None,
        )
        llm.model_args.pop("top_p", None)
        return llm

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI-based RAGAS evaluation.")
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    return llm_factory(judge_model, provider="openai", client=client)


def create_ragas_embeddings(settings: Settings):
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for RAGAS embedding-based metrics.")

    # RAGAS 0.4.x legacy metrics still expect the old LangChain-style embedding
    # interface with embed_query()/embed_documents(), so we intentionally use the
    # legacy factory mode here instead of the modern provider wrapper.
    return embedding_factory(
        settings.embedding_model,
        run_config=RunConfig(timeout=settings.openai_timeout_seconds),
        interface="legacy",
    )


def build_metrics(metric_names: list[str], llm, embeddings):
    factories = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_recall": lambda: ContextRecall(llm=llm),
        "context_precision_with_reference": lambda: LLMContextPrecisionWithReference(
            llm=llm,
            name="context_precision_with_reference",
        ),
        "answer_correctness": lambda: AnswerCorrectness(llm=llm, embeddings=embeddings),
    }

    unknown = sorted(set(metric_names) - set(factories))
    if unknown:
        raise ValueError(
            f"Unsupported metrics requested: {', '.join(unknown)}. "
            f"Supported metrics: {', '.join(sorted(factories))}."
        )

    return [factories[name]() for name in metric_names]


def prepare_evaluation_records(service: RagService, dataset_path: Path):
    samples = load_eval_dataset(dataset_path)
    indexed_documents: dict[tuple[str, str | None], str] = {}
    records: list[dict[str, object]] = []

    for sample in samples:
        resolved_document_path = resolve_document_path(dataset_path, sample.document_path)
        document_key = (str(resolved_document_path), sample.chunking_strategy)

        if document_key not in indexed_documents:
            upload = service.ingest_document(
                filename=resolved_document_path.name,
                content_type=infer_content_type(resolved_document_path),
                content=resolved_document_path.read_bytes(),
                chunking_strategy_key=sample.chunking_strategy,
            )
            indexed_documents[document_key] = upload.document_id

        query_result = service.answer_with_context(
            ChatRequest(
                question=sample.question,
                document_id=indexed_documents[document_key],
            )
        )

        retrieved_contexts = [doc.page_content for doc in query_result.retrieved_documents]
        records.append(
            {
                "sample_id": sample.sample_id,
                "document_path": str(resolved_document_path),
                "chunking_strategy": sample.chunking_strategy or "research_paper",
                "question": sample.question,
                "reference": sample.reference,
                "response": query_result.answer,
                "retrieved_contexts": retrieved_contexts,
                "retrieved_chunk_count": len(retrieved_contexts),
                "tags": sample.tags,
            }
        )

    return records


def compute_aggregate_scores(report_df, selected_metrics: list[str]) -> dict[str, float]:
    aggregate_scores: dict[str, float] = {}
    for metric in selected_metrics:
        if metric in report_df.columns:
            aggregate_scores[metric] = float(report_df[metric].mean())
    return aggregate_scores


def build_baseline_comparison(
    *,
    baseline_path: Path | None,
    selected_metrics: list[str],
    aggregate_scores: dict[str, float],
) -> dict[str, object] | None:
    if baseline_path is None:
        return None

    if not baseline_path.exists():
        LOGGER.info("No baseline snapshot found at %s; skipping comparison.", baseline_path)
        return None

    baseline_payload = load_baseline_snapshot(baseline_path)
    baseline_scores = baseline_payload.get("aggregate_scores", {})
    if not isinstance(baseline_scores, dict):
        raise ValueError(
            f"Baseline snapshot '{baseline_path}' is missing a valid aggregate_scores object."
        )

    metric_deltas = compare_aggregate_scores(
        current_scores=aggregate_scores,
        baseline_scores=baseline_scores,
        metrics=selected_metrics,
    )
    matched_metrics = [item for item in metric_deltas if item["baseline"] is not None]
    missing_metrics = [item["metric"] for item in metric_deltas if item["baseline"] is None]

    LOGGER.info("Baseline comparison against %s", baseline_path)
    for item in matched_metrics:
        LOGGER.info(
            "Metric %s: current=%.4f baseline=%.4f delta=%+.4f",
            item["metric"],
            item["current"],
            item["baseline"],
            item["delta"],
        )
    if missing_metrics:
        LOGGER.info(
            "Baseline snapshot is missing aggregate scores for: %s",
            ", ".join(missing_metrics),
        )

    return {
        "baseline_path": str(baseline_path),
        "baseline_generated_at": baseline_payload.get("generated_at"),
        "baseline_judge_provider": baseline_payload.get("judge_provider"),
        "baseline_judge_model": baseline_payload.get("judge_model"),
        "metrics": metric_deltas,
    }


def save_reports(
    *,
    report_dir: Path,
    dataset_path: Path,
    report_df,
    aggregate_scores: dict[str, float],
    selected_metrics: list[str],
    judge_provider: str,
    judge_model: str,
    baseline_comparison: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report_stem = f"{dataset_path.stem}-{timestamp}"
    csv_path = report_dir / f"{report_stem}.csv"
    json_path = report_dir / f"{report_stem}.json"

    report_df.to_csv(csv_path, index=False)

    payload = {
        "dataset_path": str(dataset_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "metrics": selected_metrics,
        "aggregate_scores": aggregate_scores,
        "samples": report_df.to_dict(orient="records"),
    }
    if baseline_comparison is not None:
        payload["baseline_comparison"] = baseline_comparison
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return csv_path, json_path


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    dataset_path = Path(args.dataset).resolve()
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = (BASE_DIR / report_dir).resolve()
    baseline_path = resolve_baseline_path(
        base_dir=BASE_DIR,
        dataset_path=dataset_path,
        baseline_arg=args.baseline,
        create_default=args.save_baseline,
    )
    if args.baseline and baseline_path is not None and not baseline_path.exists() and not args.save_baseline:
        raise FileNotFoundError(f"Baseline snapshot '{baseline_path}' does not exist.")

    selected_metrics = [
        metric_name.strip()
        for metric_name in args.metrics.split(",")
        if metric_name.strip()
    ]
    if not selected_metrics:
        raise ValueError("At least one metric must be provided.")

    judge_model = args.judge_model or settings.chat_model
    LOGGER.info(
        "Running RAGAS evaluation for dataset=%s with metrics=%s",
        dataset_path,
        selected_metrics,
    )

    with TemporaryDirectory(prefix="docutalk-ragas-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        service = create_eval_service(settings, temp_dir)
        records = prepare_evaluation_records(service, dataset_path)

    evaluation_dataset = EvaluationDataset.from_list(
        [
            SingleTurnSample(
                user_input=record["question"],
                retrieved_contexts=record["retrieved_contexts"],
                response=record["response"],
                reference=record["reference"],
            ).model_dump()
            for record in records
        ]
    )

    llm = create_ragas_llm(settings, args.judge_provider, args.judge_model)
    embeddings = create_ragas_embeddings(settings)
    metrics = build_metrics(selected_metrics, llm, embeddings)
    run_config = RunConfig(timeout=args.timeout_seconds, max_workers=args.max_workers)

    result = evaluate(
        evaluation_dataset,
        metrics=metrics,
        run_config=run_config,
        batch_size=args.batch_size,
        raise_exceptions=True,
        show_progress=True,
    )

    report_df = result.to_pandas()
    report_df.insert(0, "sample_id", [record["sample_id"] for record in records])
    report_df.insert(1, "document_path", [record["document_path"] for record in records])
    report_df.insert(2, "chunking_strategy", [record["chunking_strategy"] for record in records])
    report_df.insert(3, "retrieved_chunk_count", [record["retrieved_chunk_count"] for record in records])
    aggregate_scores = compute_aggregate_scores(report_df, selected_metrics)
    baseline_comparison = build_baseline_comparison(
        baseline_path=baseline_path,
        selected_metrics=selected_metrics,
        aggregate_scores=aggregate_scores,
    )

    csv_path, json_path = save_reports(
        report_dir=report_dir,
        dataset_path=dataset_path,
        report_df=report_df,
        aggregate_scores=aggregate_scores,
        selected_metrics=selected_metrics,
        judge_provider=args.judge_provider,
        judge_model=judge_model,
        baseline_comparison=baseline_comparison,
    )
    if args.save_baseline and baseline_path is not None:
        baseline_snapshot = build_baseline_snapshot(
            dataset_path=dataset_path,
            judge_provider=args.judge_provider,
            judge_model=judge_model,
            metrics=selected_metrics,
            aggregate_scores=aggregate_scores,
            sample_count=len(records),
        )
        save_baseline_snapshot(baseline_path, baseline_snapshot)
        LOGGER.info("Saved baseline snapshot to %s", baseline_path)

    LOGGER.info("Saved RAGAS CSV report to %s", csv_path)
    LOGGER.info("Saved RAGAS JSON report to %s", json_path)


if __name__ == "__main__":
    main()
