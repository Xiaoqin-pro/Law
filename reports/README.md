# Scientific audit and corrected rerun

This directory records the validation work completed before any Phase 3.5 citation evaluator or Phase 4 training/ablation work.

## Scope

- Dense retrieval now uses the normalized BGE-M3 CLS representation. The previous mean-pooling implementation remains available only through the explicitly named legacy configuration.
- Retrieval metrics report both Hit@K and gold-statute Recall@K; they are no longer treated as the same measure.
- Dense and Hybrid retrieval were rerun on the 309-query LeCoQA test split.
- Generation was rerun for the corrected Dense and Hybrid inputs and the Phase 3.2 BM25 context-safe inputs with Qwen3-4B and Qwen2.5-7B.
- Prompt audits record token counts, truncation, and fully/partially visible statute blocks.
- Phase 3.2.1 corrected the BM25 evidence marker to `[法条ID <number>]` and reran both BM25 models under a new experiment ID.

## Phase 3.2 final baseline freeze

- Gold reconciliation is non-positional and formatting-only: all 309 queries have deterministic resolved sets; 267 are raw-clean, 42 are order-only conflicts, and 0 are unresolved or true-set-conflicted. Strict Level A/B high-confidence coverage is 11 queries; Level C matches are reported separately.
- BM25 v2 contains 618/618 successful records across both models, with 0 truncation and 0 partially visible statute blocks.
- Dense CLS is the primary retriever candidate on the raw and raw-clean views; Hybrid remains the comparison baseline.
- Phase 3.5 citation analysis and Phase 4 repair/training remain deferred.

## Acceptance results

| Check | Result |
|---|---:|
| Corrected retrieval rows | 309 per method |
| Corrected generation rows | 309 per model/method; 1,236 total |
| Corrected generation status | 1,236/1,236 `ok` |
| Corrected generation truncation | 0 |
| Corrected generation partial statute blocks | 0 |
| Unit tests | 20/20 passed |

The non-positional LeCoQA gold reconciliation classified 267 test records as raw-clean and 42 as order-only conflicts. All records are retained and reported; no record is silently discarded or promoted from Level C to strict high-confidence gold.

The legacy generation context audit found 4 truncated BM25 records and no truncation in the legacy Dense or Hybrid records. The corrected rerun has no truncation in either Dense or Hybrid method.

The pre-encoding-fix Phase 3.2 BM25 artifacts remain preserved as legacy outputs. The Phase 3.2.1 BM25 outputs are the canonical BM25 baseline: 618/618 successful records, 0 truncation, 0 partially visible statute blocks, and matching included/fully-visible evidence totals.

## Key corrected retrieval metrics

| Method | Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (CLS) | 0.414239 | 0.650485 | 0.715210 | 0.317907 | 0.533279 | 0.596074 | 0.512704 |
| Hybrid (CLS) | 0.375405 | 0.637540 | 0.705502 | 0.285578 | 0.511036 | 0.584499 | 0.482610 |

## Files

- `scientific_audit.json`: combined machine-readable audit report.
- `phase3_1_summary.json`: corrected generation summary and artifact hashes.
- `data/lecoqa_gold_audit.json`: full gold-alignment audit.
- `generation/context_audit.json`: audit of the earlier generation outputs.
- `final_baseline_manifest.json`: hashes, metrics, generation summaries, and final acceptance status.
- `final_baseline_report.md`: human-readable Phase 3.2 freeze report.
- `data/lecoqa_gold_reconciliation.json`: non-positional reconciliation output.
- `phase3_2/retrieval_final.csv`: raw, raw-clean, and high-confidence retrieval metrics.
- `configs/generation_phase3_2_1_bm25_encoding_fixed.json`: BM25 encoding-fixed experiment configuration.

The large model outputs, embedding cache, and FAISS index remain local and ignored by Git. Their metadata and hashes are recorded in the experiment reports.

Phase 3.5 and Phase 4 remain intentionally deferred until these corrected baselines are reviewed.
