# Phase 3.5 automatic citation-audit workspace

This directory contains the automatic evaluation produced from frozen release commit `b3679ade8b233b71e5d3812dd44184aabee05f8f`.

The analysis verifies all eight frozen generation artifacts before reading them. It does not regenerate answers, change retrieval, change prompts, repair claims, run an LLM verifier, or enter Phase 4.

Key outputs:

- `phase3_5A_report_zh.md`: Chinese automatic-evaluation report and limits.
- `phase3_5_checkpoint_report.md`: checkpoint for review before Phase 3.5B.
- `citation_records.jsonl`: one normalized record per frozen answer, including unresolved citations.
- `citation_metrics.csv` / `citation_metrics.json`: model × method automatic metrics, with occurrence-level gold matching and explicit conditional/joint estimands.
- `retrieval_citation_matrix.csv`: A/B/C/D retrieval-to-citation diagnostic matrix.
- `citation_metric_bootstrap.csv`: paired query-level bootstrap, seed 42, 10,000 samples.
- `answer_quality.csv` / `per_answer_quality.jsonl`: deterministic lexical similarity proxies. BERTScore is recorded as unavailable because the package was not installed.
- `manual_audit.xlsx`: a stratified diagnostic sample of 60 unique query IDs; `primary_answers` contains only 120 Dense rows (2 models × 60 queries), while Direct/BM25/Hybrid remain comparison context.
- `sampling_manifest.json`: source stratum, population/sample sizes, selection probabilities, and sampling weights. The sample is diagnostic and unweighted proportions are not 309-query population prevalence.
- `parser_qa.xlsx` / `parser_qa_data.json`: fixed-seed 100-unit parser QA sample (40 resolved citations, 40 unresolved/ambiguous/malformed citations, and 20 no-citation answers), with blank manual fields.
- `parser_qa_summary.json` / `parser_qa_summary_zh.md`: legacy human-review instrument; it remains `WAITING_FOR_HUMAN_ANNOTATION` because the 100 manual rows are blank and it is not modified by this route.
- `model_assisted_parser_qa/`: a separately evaluated ChatGPT model-assisted overlay. Its status remains `MODEL_ASSISTED_ONLY`; it is reference-only and must not be reported as human validation.
- `parser_v2/`: Parser v2, independent deterministic audit, v1/v2 comparison, model-assisted reference comparison, and the new automated gate.
- `parser_qa_independent_precheck.json` / `.md`: standalone raw-text precheck for reviewer convenience only; it is explicitly not Parser QA acceptance or human annotation.
- `manual_annotation_progress.json`: immutable-field and annotation-schema validation for the 120 Dense rows; currently `WAITING_FOR_HUMAN_ANNOTATION`.
- `phase3_5B1_checkpoint_report.md`: B-1 order, acceptance rule, and stop condition.
- `annotation_guideline.md`: semantic-review labels and boundaries.

The `[法条ID n]` strings that some RAG answers copied from the prompt are recorded separately as `evidence_marker_ids`; they are not counted as conventional law/article citations. This prevents internal prompt markers from inflating citation-presence and citation-quality metrics.

The formal human-QA gate has been removed for this protocol. The active route is **model-assisted reference analysis plus deterministic automated audit**. The blank original `parser_qa.xlsx` and the semantic `manual_audit.xlsx` remain untouched; no human labels are inferred or fabricated. Parser v2 is accepted only when its automated gate passes, and this does not constitute human validation.

After the Parser v2 automated gate, stop. Do not begin human annotation, Dense semantic repair, answer regeneration, retrieval reruns, or Phase 4.
