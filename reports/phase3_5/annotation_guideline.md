# Phase 3.5B semantic-review guideline

This workbook is a human-review instrument. Phase 3.5A automatically checks citation parsing, corpus existence, set-based gold membership, visibility in the model's evidence, and whether gold evidence entered context. It does not decide legal semantic correctness.

## First-round scope

Review only `primary_answers`: Dense CLS outputs for 60 queries × 2 models = 120 answers. Direct, BM25, and Hybrid remain in `additional_context` for comparison only.

## Multi-label annotation fields

- `overall_answer_quality`: `correct` / `partially_correct` / `incorrect` / `uncertain`
- `fabricated_citation_present`: `yes` / `no` / `uncertain`
- `wrong_existing_citation_present`: `yes` / `no` / `uncertain`
- `unsupported_by_cited_evidence_present`: `yes` / `no` / `uncertain`
- `missing_relevant_citation_present`: `yes` / `no` / `uncertain`
- `unsupported_extension_present`: `yes` / `no` / `uncertain`
- `evidence_misuse_present`: `yes` / `no` / `uncertain`
- `retrieval_failure_contributed`: `yes` / `no` / `not_applicable` / `uncertain`
- `primary_failure_type`: `none` / `fabricated_citation` / `wrong_existing_citation` / `unsupported_by_cited_evidence` / `missing_relevant_citation` / `unsupported_extension` / `evidence_misuse` / `retrieval_failure` / `mixed` / `uncertain`
- `gold_evidence_semantically_sufficient`: `yes` / `partial` / `no` / `uncertain`
- `failure_origin`: `retrieval` / `generation` / `both` / `neither` / `uncertain`
- `reviewer_id` and `annotation_round`: leave blank in this round; reserved for inter-annotator agreement.

Use the multi-label columns to record every observed failure. Use `primary_failure_type` only for the main source; do not hide multiple failures inside `mixed`.

## Boundary

Citation existence, citation-gold membership, and citation visibility do not replace semantic-support judgment. A citation outside gold is not automatically legally wrong, and a citation inside gold is not automatically sufficient. When statutory conflicts or specialist interpretation is unclear, use `uncertain` and `reviewer_confidence=low`; do not guess.

The selected queries retain `original_stratum`, `stratum_population_N`, `stratum_sample_n`, `selection_probability`, and `sampling_weight`. This is a stratified diagnostic sample, not a simple random sample. Unweighted annotation proportions must not be reported as 309-query prevalence; use the stored weights for any post-stratified estimate.
