# DocuTalk RAGAS Evaluation

This directory contains an offline evaluation flow for DocuTalk's RAG pipeline.

It is intended for developer benchmarking, not live end-user uploads.

## Dataset Format

Create a `.jsonl` or `.json` dataset where each sample contains:

- `sample_id`
- `document_path`
- `question`
- `reference`
- `chunking_strategy` (optional)
- `tags` (optional)

Example JSONL row:

```json
{"sample_id":"paper-1-q1","document_path":"./docs/paper.pdf","question":"What is the main contribution?","reference":"The paper introduces ...","chunking_strategy":"research_paper"}
```

Relative `document_path` values are resolved relative to the dataset file location.

## Shipped Benchmark

The first curated regression dataset is checked into the repo at:

- `evals/datasets/docutalk_benchmark_v1.jsonl`

It includes:

- 3 synthetic but DocuTalk-representative text fixtures under `evals/datasets/docs/`
- 15 benchmark samples spanning research-style writing, long-form reports, and meeting notes
- coverage for all three chunking presets: `research_paper`, `general_article`, and `notes_transcript`
- repeated document/question pairs under alternate chunking strategies so retrieval changes are easier to compare over time

The fixtures are plain UTF-8 text on purpose. That keeps extraction deterministic while still exercising the real ingest, chunking, embedding, retrieval, and answer generation pipeline.

## How To Run

From the `backend/` directory:

```bash
python -m evals.run_ragas_eval --dataset evals/datasets/docutalk_benchmark_v1.jsonl
```

Optional flags:

- `--judge-provider anthropic|openai`
- `--judge-model <model-name>`
- `--metrics faithfulness,answer_relevancy,context_recall,context_precision_with_reference,answer_correctness`
- `--report-dir evals/reports`
- `--baseline evals/baselines/docutalk_benchmark_v1.json`
- `--save-baseline`

## Baselines

The evaluator can compare a fresh run against a checked-in aggregate-score snapshot.

- If `evals/baselines/<dataset_stem>.json` exists, it is loaded automatically.
- Use `--baseline <path>` to compare against an explicit snapshot file.
- Use `--save-baseline` to write the current run's aggregate scores back to the default or explicit baseline path.

Example flow:

```bash
python -m evals.run_ragas_eval \
  --dataset evals/datasets/docutalk_benchmark_v1.jsonl \
  --save-baseline
```

After that, later runs against the same dataset will log per-metric deltas versus the saved baseline and include the comparison in the JSON report.

## Output

Each run writes:

- a CSV report with per-sample scores
- a JSON report with aggregate scores and row-level details

Reports are written to `backend/evals/reports/`.

If you want to author your own dataset instead of using the curated benchmark, start from `evals/datasets/template.jsonl`.
