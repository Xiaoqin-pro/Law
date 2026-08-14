# Phase 3.5 Parser v2 自动化审计报告

- Parser v2: `citation_parser_v2`
- 冻结基线 commit: `b3679ade8b233b71e5d3812dd44184aabee05f8f`
- 范围：2472 条冻结答案；只重新解析 citation，不重新生成答案、不重新检索。
- 审计路线：模型辅助参考 + 独立确定性审计；**不是人工验证**。

## 1. v1 冻结与 v2 变化

v1 保留为不可覆盖的冻结版本：共 1829 个 conventional citation，其中 1465 个已解析，364 个 unresolved/uncertain。
v2 共发现 1829 个 citation；高置信度解析 1749 个，仍 unresolved/uncertain 80 个。按 occurrence 顺序配对后，v1 未解析而被 v2 高置信度解析的数量为 284。
v2 的新增能力是：唯一局部前件下的法名上下文传播、边界/括号归一化、长标题嵌套括号处理，以及轻度 malformed citation 修复；没有把多法名歧义强行猜成某一部法律。
因此，v1/v2 会改变 parser-dependent 的 citation existence、gold-match 和 citation consistency 数值；论文中的 citation 表应采用 v2 版本。它不改变冻结的 retrieval 结果，也不提供法律语义正确性结论。当前不存在会推翻 Dense CLS 作为后续分析主检索候选的自动证据，但后续仍不得把 parser 指标写成显著的法律质量结论。

## 2. 解析方法与主指标规则

主自动指标只计入 `confidence=high` 的 `resolved_unique`；medium/low 单独报告，不混入主结果。citation 的存在、gold 命中和 evidence 可见性仍然不等于法律语义正确。

| Model | Method | Citation count | High resolved | Unresolved/uncertain | Medium | Low | High-confidence gold match |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen25_7b | Direct | 16 | 13 | 3 | 0 | 0 | 0.38461538461538464 |
| qwen25_7b | BM25 encoding-fixed | 234 | 233 | 1 | 0 | 0 | 0.3905579399141631 |
| qwen25_7b | Dense CLS | 269 | 264 | 5 | 0 | 0 | 0.38636363636363635 |
| qwen25_7b | Hybrid CLS | 276 | 271 | 5 | 0 | 0 | 0.39114391143911437 |
| qwen3_4b | Direct | 295 | 293 | 2 | 0 | 0 | 0.20136518771331058 |
| qwen3_4b | BM25 encoding-fixed | 242 | 225 | 17 | 0 | 0 | 0.36 |
| qwen3_4b | Dense CLS | 232 | 210 | 22 | 0 | 0 | 0.34285714285714286 |
| qwen3_4b | Hybrid CLS | 265 | 240 | 25 | 0 | 0 | 0.375 |

详细指标见 `citation_metrics_v2.csv`；检索-引用诊断和 paired bootstrap 也以 v2 解析结果重新计算，但没有改变冻结的 retrieval 结果。

## 3. 模型辅助参考对照

40 条 resolved sample 的 statute mapping accuracy：40/40。20 条 no-citation control 的 false positive：0/20。
40 条 priority review case 与模型辅助标签的 agreement：36/39；这只是 `MODEL_ASSISTED_REFERENCE_ONLY` 对照，不是人工金标准。

## 4. 独立确定性审计

独立 checker 检查了 1749 个高置信度解析，冲突数为 0，状态为 **PASS**。checker 独立重建 corpus law/article key，并检查原始条号未被 repair 改变；没有调用 Parser v2 进行自证。
假阳性控制：20 条 no-citation control 均未产生 conventional citation；剩余 80 条被明确保留为 unresolved/uncertain，不能自动计为 fabrication。

## 5. 自动化 gate

最终状态：**PARSER_V2_AUTOMATED_AUDIT_PASS**。

| 条件 | 结果 |
|---|---|
| 全部 unit/regression tests | True |
| 40 条 resolved sample mapping accuracy = 100% | True |
| 20 条 no-citation controls 无 false positive | True |
| 独立 checker 无高置信度冲突 | True |
| 无重复高置信度 parser failure | True |
| 歧义案例保持 unresolved | True |

本 gate 是 `PARSER_V2_AUTOMATED_AUDIT_PASS/FAIL`，明确不是 human validation。完成本阶段后停止：不开始人工标注、不做 Dense semantic repair、不进入 Phase 4、不重新生成答案或运行 retrieval。
