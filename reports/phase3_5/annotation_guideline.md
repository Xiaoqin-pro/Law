# Phase 3.5B 人工审核指南（仅供人工填写）

本文件对应 `manual_audit.xlsx`。Phase 3.5A 的自动结果只判断：引用是否被解析、法条是否存在、是否命中 gold 集合、是否位于模型实际可见 evidence，以及 gold evidence 是否进入上下文；它不判断法律结论是否正确。

## 填写范围

第一轮只填写 `primary_answers` 工作表中的 Direct 与 Dense 四类答案。`additional_context` 仅用于对照，不作为第一轮主要标注对象。

## 字段定义

- `overall_answer_quality`：`correct` / `partially_correct` / `incorrect` / `uncertain`
- `citation_failure_type`：`none` / `fabricated_citation` / `wrong_existing_citation` / `unsupported_by_cited_evidence` / `missing_relevant_citation` / `mixed` / `uncertain`
- `unsupported_extension`：`yes` / `no` / `uncertain`
- `retrieval_failure_contributed`：`yes` / `no` / `not_applicable` / `uncertain`
- `evidence_misuse`：`yes` / `no` / `uncertain`
- `reviewer_confidence`：`high` / `medium` / `low`

## 重要边界

“法条存在”“citation 命中 gold”“citation 在可见 evidence 中”都不能替代语义支持判断。只有在阅读问题、参考答案、法条正文和模型答案后，才填写 `unsupported_by_cited_evidence`、`evidence_misuse` 或法律结论质量。

- Fabricated Citation：corpus 中不存在该法条。
- Wrong Existing Citation：法条真实存在，但不是当前问题/结论对应的正确引用。
- Unsupported by Cited Evidence：法条存在，但正文不能支持答案中的相关法律主张。
- Missing Relevant Citation：答案提出了需要法律依据的结论，却没有给出应有的显式引用。
- Unsupported Extension：在法条或题目事实之外擅自增加期限、金额、条件、责任形式、例外或程序要求等具体结论。
- Evidence Misuse：模型看到了相关 evidence，但仍然错误解释、扩大或适用它。

不要把语言风格、答案长短或措辞不漂亮本身标为 citation failure。
