# Citelaw

## Current status (2026-08-14)

The reproducibility work has completed the corrected Phase 2.1 retrieval rerun and Phase 3.1 Dense/Hybrid generation rerun. Dense retrieval uses normalized BGE-M3 CLS pooling, and the evaluation now separates Hit@K from gold-statute Recall@K. The corrected generation set contains 1,236 successful records (309 test queries × 2 models × 2 retrieval methods), with no prompt truncation or partially visible statute block.

The machine-readable audit and the exact acceptance results are in [`reports/README.md`](reports/README.md) and [`reports/scientific_audit.json`](reports/scientific_audit.json). Phase 3.5 citation evaluation and Phase 4 training/ablation are intentionally deferred pending review of these baselines.

## Phase 3.2 status (2026-08-14)

The final baseline freeze is complete pending commit publication. Gold reconciliation is non-positional and formatting-only: 309/309 queries have deterministic resolved sets, with 267 raw-clean records, 42 order-only conflicts, and 0 unresolved or true-set-conflicted records. The strict Level A/B high-confidence subset contains 11 queries; Level C matches remain separately reported.

BM25 v2 was rerun for both Qwen2.5-7B and Qwen3-4B with deterministic complete-statute-block packing. All 618 records are successful, with no prompt truncation or partially visible statute blocks. Dense CLS is the primary retriever candidate; Hybrid remains the comparison baseline. See [`reports/final_baseline_report.md`](reports/final_baseline_report.md) and [`reports/final_baseline_manifest.json`](reports/final_baseline_manifest.json). Phase 3.5 citation analysis and Phase 4 repair/training remain deferred.

面向中文法律咨询的可复现实验项目。当前只实现研究方案中的 Phase 0–1：项目骨架、LeCoQA 原始数据准备、统一 JSONL 转换、统计、人工抽样检查和基础测试。

当前阶段暂不训练模型，也不实现 BM25、Dense、Hybrid、生成或验证模块。这样可以先把数据入口和可重复性固定下来，再进入检索实验。

## 目录

```text
citelaw/
├── configs/                 # 数据源和实验配置
├── data/
│   ├── raw/                 # 下载的上游原始数据（本地保留，不提交）
│   └── processed/           # 统一 JSONL、统计和抽样检查结果
├── scripts/
│   ├── download_lecoqa.py   # 下载 LeCoQA
│   ├── prepare_data.py       # 转换、校验、统计
│   └── inspect_data.py       # 打印统计和人工检查样本
├── src/citelaw/data.py       # 数据处理核心逻辑
├── tests/                   # Phase 0–1 基础测试
├── requirements.txt
└── README.md
```

## 环境

需要 Python 3.9 或更高版本。Phase 0–1 的数据处理只使用 Python 标准库；`pytest` 仅用于方便运行测试。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 下载原始数据

数据来自官方 [oneal2000/LeCoQA](https://github.com/oneal2000/LeCoQA) 仓库。脚本会保留上游文件内容，不覆盖原始结构：

```powershell
python scripts/download_lecoqa.py
```

文件会放在 `data/raw/LeCoQA/`，包括：

- `data/queries.json`
- `data/corpus.jsonl`
- `data/example/train.json`
- `data/example/test.json`
- `README.md` 和 `License`

如需重新下载，可加 `--force`。

## 转换和检查

```powershell
python scripts/prepare_data.py
```

脚本会完成以下检查：

- `query_id` 不重复；
- corpus 的 `statute_id` 不重复；
- 每个 gold statute ID 都存在于 corpus；
- 测试集问题非空；
- reference answer 非空；
- gold ID、名称、正文三组字段长度一致；
- QA 引用的法条名称与 corpus 名称一致。

上游数据审计发现，部分记录的 `match_id` 与 `match_name` 存在错位（这两列不能互相覆盖）。转换时保留上游的 `gold_statute_names`，并根据 `match_id` 从 corpus 回填 `gold_statute_texts`，同时在每条记录中保存 `gold_corpus_names` 和 `gold_name_matches_corpus`，统计报告也会记录不一致数量。这样既不篡改原始数据，也不会把错位误当成缺失法条。

生成文件位于 `data/processed/`：

- `queries.jsonl`：全部 1,543 条 QA；
- `train.jsonl`：1,234 条训练数据；
- `test.jsonl`：309 条测试数据；
- `corpus.jsonl`：55,348 条法条；
- `data_stats.json`：数据规模和平均长度；
- `sample_check.json`：固定随机种子抽取的 10 条人工检查样本。

也可以只查看统计和样本，不生成 processed 文件：

```powershell
python scripts/inspect_data.py
```

## 测试

不依赖第三方包的运行方式：

```powershell
python -m unittest discover -s tests -v
```

如果已安装 pytest，也可以运行：

```powershell
python -m pytest
```

## 数据许可与研究记录

LeCoQA 上游仓库包含其 `License` 文件。本项目不把下载的数据文件提交到 Git；运行下载和转换脚本即可重建本地数据。后续论文实验需要在记录中固定数据源仓库、下载日期、文件哈希和脚本版本。

## 下一阶段

Phase 1 通过后，再实现：

1. BM25 检索基线；
2. Dense 检索和 FAISS 索引；
3. RRF Hybrid；
4. 检索指标 Recall@1/5/10 和 MRR；
5. 在明确实验设计后接入生成、citation verification 和 claim-level repair。

## Phase 2：检索基线

Phase 2 使用 LeCoQA 官方 test split，包含 BM25、`BAAI/bge-m3` Dense 和 BM25/Dense 的 RRF Hybrid。默认配置见 `configs/retrieval_phase2.json`。

先运行测试：

```powershell
python -m unittest discover -s tests -v
```

运行 10 条固定随机种子 smoke test：

```powershell
python scripts/run_retrieval.py --mode smoke
```

确认 smoke 输出后运行完整 309 条测试：

```powershell
python scripts/run_retrieval.py --mode full
```

结果会写入 `outputs/retrieval/`，包括每个 query 的完整 JSONL 检索结果、指标 CSV、可读 smoke 样本、错误分析和实验 metadata。`outputs/` 不提交到 Git；每个正式实验的 metadata 会记录配置、seed、数据 fingerprint、模型 revision、平台和指标。

Dense 模型第一次运行会下载 `BAAI/bge-m3`。corpus embedding 使用可恢复的 NumPy memmap 分批写入，FAISS 使用精确 inner-product index；中断后重新运行同一配置会复用已经完成的 embedding/index。显存不足时会自动将当前 batch 减半，最低降到 1。

## Phase 3：生成基线

Phase 3 框架已加入 `configs/generation_phase3.json`、`scripts/run_generation.py` 和 `src/citelaw/generation.py`。它会保存完整 prompt、原始回答、检索到的 statute ID、状态、异常和延迟，并按 `query_id` 增量写盘，支持中断恢复。四个方法是 `direct`、`bm25`、`dense`、`hybrid`。

运行前需要安装支持 4-bit 推理的 `bitsandbytes`，并首次下载 Qwen 模型。当前项目不会在缺少该依赖时静默改成 FP16 或 CPU 设置；这样可避免正式实验的模型量化条件发生变化。
