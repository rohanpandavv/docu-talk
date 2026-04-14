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

## How To Run

From the `backend/` directory:

```bash
python -m evals.run_ragas_eval --dataset evals/datasets/template.jsonl
```

Optional flags:

- `--judge-provider anthropic|openai`
- `--judge-model <model-name>`
- `--metrics faithfulness,answer_relevancy,context_recall,context_precision_with_reference,answer_correctness`
- `--report-dir evals/reports`

## Output

Each run writes:

- a CSV report with per-sample scores
- a JSON report with aggregate scores and row-level details

Reports are written to `backend/evals/reports/`.
