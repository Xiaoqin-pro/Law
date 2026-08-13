# Research Checkpoint Report

日期：2026-08-13  
项目：Chinese Legal Citation Failure Analysis + Controlled Citation Error Stress Test + Claim-Level Targeted Evidence Repair

## 1. 当前 Git 版本

- Phase 0–1 commit：`2e735f86821ea9d14a0c746be4f6299042c4359a`
- Phase 2 commit：`0ce2fce9a8538826e26008bd2372608f08ac6e53`
- Retrieval 正式实验使用 commit：`0ce2fce9a8538826e26008bd2372608f08ac6e53`

## 2. 已完成实验

### Phase 0–1 数据准备

- LeCoQA QA：1543
- Train：1234
- Test：309
- Statute corpus：55348
- 原始数据已保存在本地 `data/raw/LeCoQA/`，未提交 Git。
- 统一 JSONL、统计、样本检查和基础测试已完成。
- 上游 `match_id` / `match_name` 存在 593 个错位 pair，已保留并写入质量统计，没有静默修正原始数据。

### Phase 2 Retrieval Baselines

正式实验 ID：`phase2_full_20260813_010257`  
数据：LeCoQA 官方 test split，309 queries  
模型：`BAAI/bge-m3`，revision `5617a9f61b028005a4858fdac845db406aefb181`  
配置：BM25 `k1=1.5, b=0.75`；Dense top-10；Hybrid BM25/Dense top-50 + RRF `k=60`；seed=42。

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | Avg latency/query |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.313916 | 0.521036 | 0.576052 | 0.398776 | 134.1589 ms |
| Dense | 0.368932 | 0.588997 | 0.682848 | 0.466202 | 13.5037 ms |
| Hybrid | 0.365696 | 0.631068 | 0.695793 | 0.466651 | 185.5525 ms |

解释：Hybrid 在 Recall@5、Recall@10 和 MRR 上优于单一检索，但 Recall@1 略低于 Dense。这是正式实验的真实结果，未进行结果筛选或人为调参。

## 3. Retrieval error analysis

自动分类按 Recall@10 判断：

- BM25 成功、Dense 失败：21
- Dense 成功、BM25 失败：50
- BM25 与 Dense 均失败：50
- Hybrid rescue（两者均失败但 Hybrid 成功）：7
- Hybrid regression（至少一个单一检索成功但 Hybrid 失败）：24

典型现象：Dense 对语义改写和概念相近问题更有帮助；BM25 对法条名称、固定法律术语和局部关键词有补充作用；RRF 能救回部分双失败 query，但也可能把单一检索中的正确法条挤出 top-10。

## 4. Phase 3 状态

已完成：

- Phase 3 配置、prompt version、B1 Direct / B2 BM25 RAG / B3 Dense RAG / B4 Hybrid RAG 框架；
- 原始 prompt、回答、检索 statute ID、异常、latency 的增量 JSONL 设计；
- `bitsandbytes 0.46.1` 安装；
- RTX 4060 + CUDA 11.8 + 4-bit 依赖导入验证。

未完成：

- Qwen2.5-7B-Instruct 权重未完整下载；
- Qwen3-4B-Instruct-2507 权重未完整下载；
- B1–B4 smoke 和正式生成结果均不存在。

## 5. 当前阻塞与成本

阻塞原因是 Hugging Face 大文件权重下载通道：元数据和小文件可访问，但 4 GB 级 safetensors 分片在 CLI/Range 并发下载中出现 0 字节等待或长连接重置。通过代理直接下载成功取得了 Qwen3 第 1 片和第 3 片，但第 2 片未完整取得；所有下载进程已停止，未完成文件保留为断点材料。

这不是显存 OOM：BGE-M3 已在 RTX 4060 上完成 corpus embedding，显存约 3.8 GB；Qwen 生成模型尚未进入加载阶段。

当前未产生付费 API 或云 GPU 费用，也没有生成任何虚假 Phase 3 结果。

## 6. 数据与实验异常

- LeCoQA 上游 `match_id` / `match_name` 错位 593 pair，见 `data/processed/data_stats.json`。
- Hybrid Recall@1 略低于 Dense，属于应保留的真实结果。
- Phase 3 受模型权重下载阻塞，不能据此判断生成 baseline 或 citation failure 分布。

## 7. 当前对研究假设的支持程度

Phase 2 支持继续研究 Hybrid 作为候选证据入口：它在 top-5/top-10 覆盖上优于 BM25 和 Dense。

Phase 2 尚不能支持以下结论：

- 哪类 citation failure 最常见；
- RAG 是否降低错误法条引用；
- whole-answer revision 是否产生 collateral errors；
- claim-level targeted repair 是否优于整篇重写。

这些结论必须等待真实 B1–B4 原始回答，以及后续人工/规则化 verification 结果。

## 8. 下一步

先解决模型权重下载或转移到可稳定访问 Hugging Face 大文件的环境，再按既定顺序运行：

1. Qwen3-4B 10 条 smoke；
2. Qwen3-4B 四种方法 309 条正式实验；
3. Qwen2.5-7B 10 条 smoke；
4. Qwen2.5-7B 四种方法 309 条正式实验；
5. 汇总 B1–B4 原始结果和 citation error 初步分布。

在 B1–B4 完成前，不进入 Phase 4 claim-level repair 的大规模实现。

## 9. Phase 3 最终完成状态

Phase 3 已经完成，不再处于模型下载阻塞状态：

- Qwen3-4B-Instruct：Direct、BM25 RAG、Dense RAG、Hybrid RAG 各 309 条；309/309 成功。
- Qwen2.5-7B-Instruct：Direct、BM25 RAG、Dense RAG、Hybrid RAG 各 309 条；309/309 成功。
- 正式生成记录总数：2472 条；运行错误：0。
- 两个模型均使用本地权重和固定 4-bit 配置，GPU 显存运行稳定。

自动结果只能确认实验完成性、运行耗时、回答长度和检索覆盖率；不能自动证明法律引用正确。下一步应先进行分层人工/规则复核，比较 Direct、BM25、Dense 和 Hybrid 的 citation failure、unsupported claim 和 collateral error，再决定是否进入 Phase 4。

完整结果汇总见 `outputs/phase3_generation_summary.md`。
