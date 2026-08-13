# Scientific audit and corrected rerun

This directory records the validation work completed before any Phase 3.5 citation evaluator or Phase 4 training/ablation work.

## Scope

- Dense retrieval now uses the normalized BGE-M3 CLS representation. The previous mean-pooling implementation remains available only through the explicitly named legacy configuration.
- Retrieval metrics report both Hit@K and gold-statute Recall@K; they are no longer treated as the same measure.
- Dense and Hybrid retrieval were rerun on the 309-query LeCoQA test split.
- Generation was rerun only for the corrected Dense and Hybrid inputs with Qwen3-4B and Qwen2.5-7B.
- Prompt audits record token counts, truncation, and fully/partially visible statute blocks.

## Acceptance results

| Check | Result |
|---|---:|
| Corrected retrieval rows | 309 per method |
| Corrected generation rows | 309 per model/method; 1,236 total |
| Corrected generation status | 1,236/1,236 `ok` |
| Corrected generation truncation | 0 |
| Corrected generation partial statute blocks | 0 |
| Unit tests | 15/15 passed |

The LeCoQA gold audit classified 267 test records as clean and 42 as `id_evidence_conflict`. These records are retained and reported; they are not silently discarded.

The legacy generation context audit found 4 truncated BM25 records and no truncation in the legacy Dense or Hybrid records. The corrected rerun has no truncation in either Dense or Hybrid method.

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

The large model outputs, embedding cache, and FAISS index remain local and ignored by Git. Their metadata and hashes are recorded in the experiment reports.

Phase 3.5 and Phase 4 remain intentionally deferred until these corrected baselines are reviewed.
