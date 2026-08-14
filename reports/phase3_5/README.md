# Phase 3.5A automatic citation analysis

This directory contains the automatic evaluation produced from frozen release commit `b3679ade8b233b71e5d3812dd44184aabee05f8f`.

The analysis verifies all eight frozen generation artifacts before reading them. It does not regenerate answers, change retrieval, change prompts, repair claims, run an LLM verifier, or enter Phase 4.

Key outputs:

- `phase3_5A_report_zh.md`: Chinese automatic-evaluation report and limits.
- `phase3_5_checkpoint_report.md`: checkpoint for review before Phase 3.5B.
- `citation_records.jsonl`: one normalized record per frozen answer, including unresolved citations.
- `citation_metrics.csv` / `citation_metrics.json`: model × method automatic metrics.
- `retrieval_citation_matrix.csv`: A/B/C/D retrieval-to-citation diagnostic matrix.
- `citation_metric_bootstrap.csv`: paired query-level bootstrap, seed 42, 10,000 samples.
- `answer_quality.csv` / `per_answer_quality.jsonl`: deterministic lexical similarity proxies. BERTScore is recorded as unavailable because the package was not installed.
- `manual_audit.xlsx`: 60 unique query IDs, 240 Direct/Dense primary rows, and BM25/Hybrid comparison context.
- `annotation_guideline.md`: semantic-review labels and boundaries.

The `[法条ID n]` strings that some RAG answers copied from the prompt are recorded separately as `evidence_marker_ids`; they are not counted as conventional law/article citations. This prevents internal prompt markers from inflating citation-presence and citation-quality metrics.

The project is intentionally stopped at Phase 3.5A. Human semantic review must be completed before any Phase 3.5B or Phase 4 design.
