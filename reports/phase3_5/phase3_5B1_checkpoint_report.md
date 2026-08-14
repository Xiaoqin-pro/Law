# Phase 3.5B-1 checkpoint: Human Validation infrastructure

This checkpoint starts Phase 3.5B-1 without changing the frozen baseline or
regenerating any answer. The order is intentionally fixed:

1. Complete the 100-row `parser_qa.xlsx` review.
2. Run `scripts/evaluate_parser_qa.py` and accept Parser QA only if resolved
   precision is at least 0.95 and no repeated systematic parser failure is
   flagged.
3. Complete the 120-row Dense-only `primary_answers` review.
4. Run `scripts/validate_manual_audit.py` to produce diagnostic counts and
   post-stratified estimates only after all required human fields are filled.

## Current status

| Workstream | Status | Current result |
|---|---|---|
| Parser QA | WAITING_FOR_HUMAN_ANNOTATION | 0/100 rows complete |
| Dense semantic review | WAITING_FOR_HUMAN_ANNOTATION | 0/120 rows complete |
| Frozen baseline | FROZEN | commit `b3679ade8b233b71e5d3812dd44184aabee05f8f` |
| Phase 4 | BLOCKED BY STOP RULE | not started |

The validation scripts never fill manual fields. They distinguish blank,
uncertain, invalid, and completed values, and they report incomplete status
instead of treating blank workbooks as a pass.

An independent raw-text precheck is also available in
`parser_qa_independent_precheck.json`. It does not call the production parser,
but it is only a convenience check and cannot be reported as human Parser QA.

## Parser QA boundary

Parser QA checks extraction, law-name normalization, article-number
normalization, statute mapping, and whether originally unresolved citations are
actually resolvable. It does not judge whether a legal conclusion is correct.
No-citation controls are kept separate from citation precision denominators.

## Semantic annotation boundary

The `primary_answers` sheet contains 60 unique queries × 2 Dense models = 120
answers. `additional_context` is comparison context only. Automatic gold
membership and visibility flags are evidence for review, not semantic labels.
Citation outside the annotated gold set is not automatically a wrong citation;
citation inside the gold set is not automatically supported.

## Statistical stop rule

Until both workbooks are manually completed, do not generate final failure
prevalence, model-comparison conclusions, or Phase 4 recommendations. After
completion, raw diagnostic counts and query-level post-stratified weighted
estimates will be reported separately. The 60-query design is not a simple
random sample of 309 queries.
