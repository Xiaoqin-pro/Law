# Parser v2 automated protocol

Parser v2 reanalyses the 2472 answers from the frozen baseline commit
`b3679ade8b233b71e5d3812dd44184aabee05f8f`. It does not regenerate answers,
rerun retrieval, modify prompts, or overwrite Parser v1 artifacts.

The main automatic metrics count only `resolved_unique` citations with
`confidence=high`. Medium/low confidence results are reported separately.
Ambiguous and context-free citations remain unresolved.

`parser_v2_independent_audit.json` is produced by a separate deterministic
checker that reconstructs law/article keys and checks provenance. It does not
call Parser v2 to validate itself. The model-assisted workbook is retained as
`MODEL_ASSISTED_REFERENCE_ONLY`; it is not human validation and is not a gold
standard.

The gate is `PARSER_V2_AUTOMATED_AUDIT_PASS` only when the full regression
suite, 40-case mapping check, 20 no-citation controls, independent checker,
repeated-failure check, and adversarial ambiguity checks all pass. After this
gate, the route stops before human annotation, semantic repair, and Phase 4.
