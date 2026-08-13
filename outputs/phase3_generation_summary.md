# Phase 3 Generation Summary

## 1. Completion status

The Phase 3 formal generation experiment is complete.

- Models: Qwen3-4B-Instruct and Qwen2.5-7B-Instruct
- Methods: Direct, BM25 RAG, Dense RAG, Hybrid RAG
- Test queries per method: 309
- Formal output records: 2 models × 4 methods × 309 = 2472
- Successful records: 2472/2472
- Runtime errors: 0
- Every output file has 309 unique query IDs.

The raw JSONL files preserve the question, prompt, answer, status, latency, gold statute IDs, and retrieved statute IDs where applicable.

## 2. Retrieval facts already established

The retrieval input is the same for both generation models. On the 309-query test split:

| Retriever | Gold statute hit in top 10 |
|---|---:|
| BM25 | 0.576052 |
| BGE-M3 Dense | 0.682848 |
| Hybrid | 0.695793 |

Hybrid has the best top-10 coverage, but this alone does not prove that its generated legal answers are the most correct.

## 3. Generation runtime facts

| Model | Method | Average latency/query | Average answer length |
|---|---|---:|---:|
| Qwen3-4B | Direct | 8740.9 ms | 210.2 chars |
| Qwen3-4B | BM25 | 20499.3 ms | 252.8 chars |
| Qwen3-4B | Dense | 13602.0 ms | 252.9 chars |
| Qwen3-4B | Hybrid | 15449.9 ms | 259.4 chars |
| Qwen2.5-7B | Direct | 4874.7 ms | 148.4 chars |
| Qwen2.5-7B | BM25 | 9326.3 ms | 178.4 chars |
| Qwen2.5-7B | Dense | 10026.2 ms | 191.7 chars |
| Qwen2.5-7B | Hybrid | 21724.0 ms | 189.7 chars |

These are engineering measurements, not quality scores.

## 4. What is and is not concluded

Confirmed:

- Both local models can run with the fixed 4-bit configuration on the RTX 4060 Laptop GPU.
- All eight formal conditions completed without runtime failure.
- Retrieval coverage is highest for Hybrid, followed by Dense and BM25.

Not yet confirmed:

- Whether a cited statute is legally correct for the claim.
- Whether the answer contains unsupported legal claims.
- Whether RAG reduces citation errors compared with Direct.
- Whether Hybrid improves answer correctness rather than only retrieval coverage.

The last four points require human or rule-assisted legal verification. They must not be inferred from answer length, latency, or the presence of a statute name.

## 5. Recommended next decision

Do not begin large-scale Phase 4 repair yet. First conduct a stratified manual review of the completed answers:

1. Compare Direct with Hybrid for both models.
2. Prioritize queries where Dense succeeds but BM25 fails, where BM25 succeeds but Dense fails, where Hybrid rescues both failures, and where Hybrid regresses despite a successful single retriever.
3. Label each answer for citation correctness, evidence support, legal conclusion, and unsupported claims.
4. Compute citation-failure rates by model and method.

Decision rule:

- If Hybrid has lower citation-failure and unsupported-claim rates, use Hybrid as the evidence entry for Phase 4.
- If Dense and BM25 have different strengths, retain both as controlled baselines and analyze their error categories separately.
- If RAG does not improve over Direct, revise evidence selection or prompting before implementing targeted repair.

Phase 4 should begin only after this review produces a reproducible quality table.

