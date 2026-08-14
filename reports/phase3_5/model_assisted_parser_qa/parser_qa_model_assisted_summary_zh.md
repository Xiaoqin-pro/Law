# Parser QA — Model-Assisted Overlay Evaluation

> This is a model-assisted annotation report, not an independent human QA result.

- Label source: `ChatGPT model-assisted review`
- Overlay status: **MODEL_ASSISTED_ONLY**
- Human acceptance preview: **FAIL_SYSTEMATIC_FAILURES**
- Input: `E:\xiazai\google\parser_qa_model_assisted_filled.xlsx`
- Reference sample: `D:\111\Desktop\Law\reports\phase3_5\parser_qa_data.json`

## Structural checks

- Numeric annotation rows: `100`
- Extra/blank rows ignored: `5`
- Exact audit-unit coverage 1–100: `True`
- Duplicate IDs: `[]`
- Missing expected IDs: `[]`
- Model-assisted disclaimer rows: `100`

## Recomputed metrics

- Resolved-citation precision: `1.0` (n=40)
- Unresolved rows judged actually resolvable: `0.9743589743589743` (excluding uncertain; n=39)
- No-citation control extraction accuracy: `1.0` (n=20)

## Repeated parser failure signals

- `contextual_law_name_not_propagated`: 28 rows
- `citation_boundary_bracket_error`: 6 rows

## Decision

The overlay is complete enough to prioritize human checking, but it cannot be reported as human Parser QA. The repeated failure types mean the model-assisted evidence should be independently validated before Phase 3.5B-1 is accepted. The original blank human-review workbook remains unchanged.
