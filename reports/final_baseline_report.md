# Phase 3.2 Final Baseline Freeze

- Dataset: LeCoQA official test split (309 queries)
- Manifest source commit: `68f00313f80606731c34de8b2928b5162401cf8f`
- Phase 3.5 citation analysis: deferred
- Phase 4: deferred

## Gold reconciliation

Raw-clean: 267; order-only conflicts: 42; true-set conflicts: 0; ambiguous: 0; unresolved: 0.
All 309 queries have a deterministic resolved set. Strict Level A/B high-confidence queries: 11; the remaining 298 queries are resolved only through lower-confidence Level C name matching and are reported separately, not silently promoted. Actual unresolved, ambiguous, or true-set-conflicted queries: 0.

## Retrieval metrics

The raw and raw-clean tables are the primary baseline views; the strict high-confidence subset is small and is reported as a sensitivity analysis.

| Gold definition | Retriever | N | Hit@10 | Recall@10 | MRR |
|---|---|---:|---:|---:|---:|
| raw_all | bm25 | 309 | 0.576052 | 0.472342 | 0.398776 |
| raw_all | dense_cls | 309 | 0.71521 | 0.596074 | 0.512704 |
| raw_all | hybrid_cls | 309 | 0.705502 | 0.584499 | 0.48261 |
| raw_clean | bm25 | 267 | 0.580524 | 0.491823 | 0.399551 |
| raw_clean | dense_cls | 267 | 0.719101 | 0.624719 | 0.515915 |
| raw_clean | hybrid_cls | 267 | 0.700375 | 0.608989 | 0.484521 |
| high_confidence_reconciled | bm25 | 11 | 0.272727 | 0.227273 | 0.190909 |
| high_confidence_reconciled | dense_cls | 11 | 0.272727 | 0.227273 | 0.145455 |
| high_confidence_reconciled | hybrid_cls | 11 | 0.545455 | 0.409091 | 0.142424 |

Dense CLS is the current primary retriever candidate because it is numerically ahead of Hybrid on the raw and raw-clean views. The paired-bootstrap file records uncertainty; this report does not call the difference statistically significant without a pre-specified decision rule.

## Frozen four baselines

| Baseline | Models | Retrieval input | Status |
|---|---|---|---|
| B1 Direct | Qwen2.5-7B, Qwen3-4B | none | legacy formal output retained |
| B2 BM25 v2 | Qwen2.5-7B, Qwen3-4B | legacy BM25 retrieval + complete-block packing | frozen |
| B3 Dense CLS | Qwen2.5-7B, Qwen3-4B | corrected CLS retrieval | frozen |
| B4 Hybrid CLS | Qwen2.5-7B, Qwen3-4B | corrected CLS RRF retrieval | frozen comparison baseline |

BM25 v2 acceptance: passed: 618/618 records are successful, with no truncation or partially visible statute blocks.

## Context audit

The earlier legacy BM25 output had 4 truncated records. The corrected Dense/Hybrid outputs had none. BM25 v2 is isolated under a new experiment ID and is not allowed to overwrite the legacy artifact.

## Next stage

The next allowed stage is Phase 3.5 baseline citation/failure analysis. No citation verifier, claim decomposition, re-retrieval, revision, or repair is implemented in Phase 3.2.
