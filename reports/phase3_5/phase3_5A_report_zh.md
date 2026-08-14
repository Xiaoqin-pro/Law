# Phase 3.5A 自动评估与引用失败诊断

- Frozen baseline release commit: `b3679ade8b233b71e5d3812dd44184aabee05f8f`
- Scope: 2 models × 4 frozen methods × 309 queries = 2472 answers
- Phase 3.5A.1 / Phase 3.5B-0 status: **complete; stop before Phase 3.5B-1 and Phase 4**
- No answer regeneration, retrieval change, prompt change, claim repair, verifier, or stress test was performed.

## 1. 冻结 artifact 完整性

8 个正式 generation artifact 均通过 SHA256、309 条记录、query_id 唯一、status=ok 检查；总记录数为 `2472`。
manifest 生成 commit 为 `0d186257ef96f2abb512bf8499c67a6a8ac1e50c`，本阶段按发布节点 `b3679ade8b233b71e5d3812dd44184aabee05f8f` 记录。

## 2. 确定性 citation index

corpus 共 `55348` 条 statute，结构化 canonical key `55348` 条，重复 key `0`，无法结构化 `0`。
法律名称只允许 corpus 内唯一 alias 映射；缺少法律名称、未知法律、非唯一映射均保留为 unresolved/ambiguous。

## 3. Parser QA 与 marker attribution

本轮生成固定 seed=42 的 100-unit parser QA：40 个 resolved_unique、40 个 unresolved/uncertain、20 个无 conventional citation 的 answer。人工字段仍为空，当前不能据此估计 parser precision/false-negative rate。
答案内部 evidence marker 共单独记录；marker metrics 已写入 `citation_metrics.csv`，不与 conventional citation 合并。

## 4. 自动 citation / answer 指标

下表的 `citation gold match` 按 citation occurrence 计算，不等于法律语义正确；BLEU/METEOR/ROUGE-L 是字符/词元层面的 reference similarity proxy。BERTScore 在本环境未安装，因此不作虚构数值。答案中的内部 `[法条ID n]` evidence marker 单独记录，不当作传统法律 citation 计数。

| Model | Method | Conventional citation presence | Any attribution presence | Citation existence | All exist given citation | All exist joint | Gold citation match | Marker presence | Gold evidence visible | ROUGE-L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen25_7b | Direct | 5.18% | 5.18% | 56.25% | 56.25% | 2.91% | 22.22% | 0.00% | N/A | 0.2496 |
| qwen25_7b | BM25 encoding-fixed | 50.49% | 55.66% | 90.17% | 87.82% | 44.34% | 42.18% | 11.00% | 57.61% | 0.3088 |
| qwen25_7b | Dense CLS | 50.16% | 57.93% | 91.45% | 89.03% | 44.66% | 39.84% | 15.53% | 71.52% | 0.3176 |
| qwen25_7b | Hybrid CLS | 56.63% | 62.14% | 91.30% | 89.71% | 50.81% | 40.08% | 14.56% | 70.55% | 0.3166 |
| qwen3_4b | Direct | 62.78% | 62.78% | 60.00% | 51.55% | 32.36% | 22.60% | 0.00% | N/A | 0.2505 |
| qwen3_4b | BM25 encoding-fixed | 45.31% | 55.02% | 80.58% | 77.14% | 34.95% | 37.95% | 22.01% | 57.61% | 0.2953 |
| qwen3_4b | Dense CLS | 40.45% | 53.07% | 76.29% | 71.20% | 28.80% | 37.29% | 17.80% | 71.52% | 0.3111 |
| qwen3_4b | Hybrid CLS | 44.01% | 54.69% | 74.72% | 72.06% | 31.72% | 42.42% | 19.74% | 70.55% | 0.3010 |

解释：`citation_existence_rate` 是 citation-level；`all_citations_exist_given_citation` 是有 conventional citation 答案中的 conditional rate；`answer_all_citations_exist_joint` 是全部 309 个答案中的 joint answer-level rate。RAG 的 visible 指标也同时报告 conditional 与 joint 版本。Direct 的 evidence consistency / gold evidence visible 为 N/A，因为 Direct 没有检索 evidence。unresolved/ambiguous citation 不会被自动标成 fabricated，而是保留为 parse-uncertain 并进入人工审核。

## 5. Dense-centered A/B/C/D diagnostic matrix

A = gold evidence visible 且答案显式命中 gold；B = gold evidence visible 但没有显式命中 gold；C = gold evidence 不可见但答案命中 gold（只能标记为 requires manual review）；D = 两者均未发生。

| Model | Retriever | A | B | C | D |
|---|---|---:|---:|---:|---:|
| qwen25_7b | BM25 encoding-fixed | 80 (25.89%) | 98 (31.72%) | 1 (0.32%) | 130 (42.07%) |
| qwen25_7b | Dense CLS | 85 (27.51%) | 136 (44.01%) | 1 (0.32%) | 87 (28.16%) |
| qwen25_7b | Hybrid CLS | 97 (31.39%) | 121 (39.16%) | 0 (0.00%) | 91 (29.45%) |
| qwen3_4b | BM25 encoding-fixed | 65 (21.04%) | 113 (36.57%) | 4 (1.29%) | 127 (41.10%) |
| qwen3_4b | Dense CLS | 61 (19.74%) | 160 (51.78%) | 0 (0.00%) | 88 (28.48%) |
| qwen3_4b | Hybrid CLS | 72 (23.30%) | 146 (47.25%) | 1 (0.32%) | 90 (29.13%) |

C 类没有被自动解释为参数记忆、数据泄漏或模型错误；这些只能进入人工审核。

## 6. Paired bootstrap

所有 paired bootstrap 使用同一 query_id 配对、seed=42、10000 次重采样；`difference = right_method - left_method`。CI 跨 0 时不写 statistically significant。完整结果见 `citation_metric_bootstrap.csv`。

主要比较（每个模型分别计算）：
- qwen25_7b Direct → Dense CLS：
  - citation_presence: difference=0.4498, 95% CI=[0.3916, 0.5081]（CI不跨0）
  - answer_all_citations_exist_joint: difference=0.4175, 95% CI=[0.3592, 0.4757]（CI不跨0）
  - any_gold_citation: difference=0.2718, 95% CI=[0.2233, 0.3236]（CI不跨0）
- qwen3_4b Direct → Dense CLS：
  - citation_presence: difference=-0.2233, 95% CI=[-0.2977, -0.1456]（CI不跨0）
  - answer_all_citations_exist_joint: difference=-0.0356, 95% CI=[-0.1068, 0.0356]（CI跨0，不称显著）
  - any_gold_citation: difference=0.0744, 95% CI=[0.0162, 0.1327]（CI不跨0）
- qwen25_7b BM25 encoding-fixed → Dense CLS：
  - citation_presence: difference=-0.0032, 95% CI=[-0.0647, 0.0583]（CI跨0，不称显著）
  - answer_all_citations_exist_joint: difference=0.0032, 95% CI=[-0.0615, 0.0647]（CI跨0，不称显著）
  - any_gold_citation: difference=0.0162, 95% CI=[-0.0324, 0.0647]（CI跨0，不称显著）
  - answer_all_citations_visible_joint: difference=-0.0032, 95% CI=[-0.0680, 0.0615]（CI跨0，不称显著）
- qwen3_4b BM25 encoding-fixed → Dense CLS：
  - citation_presence: difference=-0.0485, 95% CI=[-0.1100, 0.0162]（CI跨0，不称显著）
  - answer_all_citations_exist_joint: difference=-0.0615, 95% CI=[-0.1294, 0.0032]（CI跨0，不称显著）
  - any_gold_citation: difference=-0.0259, 95% CI=[-0.0744, 0.0227]（CI跨0，不称显著）
  - answer_all_citations_visible_joint: difference=-0.0583, 95% CI=[-0.1230, 0.0097]（CI跨0，不称显著）
- qwen25_7b Dense CLS → Hybrid CLS：
  - citation_presence: difference=0.0647, 95% CI=[0.0096, 0.1230]（CI不跨0）
  - answer_all_citations_exist_joint: difference=0.0615, 95% CI=[0.0032, 0.1197]（CI不跨0）
  - any_gold_citation: difference=0.0356, 95% CI=[-0.0162, 0.0874]（CI跨0，不称显著）
  - answer_all_citations_visible_joint: difference=0.0777, 95% CI=[0.0194, 0.1359]（CI不跨0）
- qwen3_4b Dense CLS → Hybrid CLS：
  - citation_presence: difference=0.0356, 95% CI=[-0.0227, 0.0939]（CI跨0，不称显著）
  - answer_all_citations_exist_joint: difference=0.0291, 95% CI=[-0.0291, 0.0874]（CI跨0，不称显著）
  - any_gold_citation: difference=0.0388, 95% CI=[-0.0065, 0.0841]（CI跨0，不称显著）
  - answer_all_citations_visible_joint: difference=0.0421, 95% CI=[-0.0162, 0.1036]（CI跨0，不称显著）

## 7. 60-query manual audit sample

选取 `60` 个唯一 query_id；按两套 Dense CLS 的 pooled gold_visible / gold_citation_match 分层，seed=42。实际分层可用数：`{'S1_gold_visible_and_gold_cited': 114, 'S2_gold_visible_but_not_gold_cited': 107, 'S3_gold_not_visible_but_gold_cited': 1, 'S4_gold_not_visible_and_not_gold_cited': 87}`；选入数：`{'S4_gold_not_visible_and_not_gold_cited': 29, 'S1_gold_visible_and_gold_cited': 15, 'S2_gold_visible_but_not_gold_cited': 15, 'S3_gold_not_visible_but_gold_cited': 1}`。每条样本保留 original_stratum、selection_probability 和 sampling_weight；该样本用于 failure diagnosis，不能把未加权的人工比例直接解释为 309-query population prevalence。

`manual_audit.xlsx` 的 primary_answers 现在只包含 Dense CLS 的 60 queries × 2 models = 120 条首轮语义审核对象；Direct/BM25/Hybrid 保留在 additional_context 作为比较背景，所有人工字段保持空白。

## 8. 当前可以支持的结论

1. 可以报告不同 baseline 的显式 citation presence、解析成功率、corpus existence、set-based gold match，以及 RAG 的实际 visible-evidence consistency。
2. 可以报告 Dense 的 retrieval gold visibility 是否传递到显式 citation 行为，并用 A/B/C/D 量化 retrieval 与 generation/citation 的关系。
3. 可以报告 Direct、BM25、Dense、Hybrid 的 answer/reference lexical similarity proxy，但不得把这些 proxy 直接写成 legal correctness。

## 9. 当前不能支持的结论

1. 自动指标不能证明 citation 对答案主张具有法律语义支持，也不能自动判断 wrong legal conclusion、unsupported extension 或 evidence misuse。
2. citation 在 gold 中不等于法律上正确；citation 不在 gold 中也不能自动等同于法律错误。
3. C 类不能自动归因于参数记忆；Dense 的点估计优势不能据此写成显著优于 Hybrid。

## 10. 阶段停止点

Phase 3.5A.1 / Phase 3.5B-0 已完成：指标口径 hotfix、parser QA、sampling manifest、Dense-only manual_audit.xlsx、annotation_guideline.md、checkpoint report 均已生成。parser QA 与语义人工审核仍待填写；本次不进入 Phase 3.5B-1，不进入 Phase 4。
