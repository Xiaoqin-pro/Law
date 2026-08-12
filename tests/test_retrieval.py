from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.retrieval import BM25Index, chinese_tokens, reciprocal_rank_fusion, retrieval_metrics  # noqa: E402


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = [
            {"statute_id": 1, "statute_name": "劳动合同法第一条", "statute_text": "劳动合同保护劳动者权益。"},
            {"statute_id": 2, "statute_name": "民法典第二条", "statute_text": "民事主体依法享有权利。"},
            {"statute_id": 3, "statute_name": "公司法第三条", "statute_text": "公司具有独立法人财产。"},
        ]

    def test_chinese_tokenizer_is_nonempty_and_deterministic(self) -> None:
        self.assertEqual(chinese_tokens("劳动合同？"), chinese_tokens("劳动合同？"))
        self.assertTrue(chinese_tokens("劳动合同？"))

    def test_bm25_returns_relevant_statute(self) -> None:
        index = BM25Index().build(self.corpus)
        results = index.search("劳动者合同权益", top_k=2)
        self.assertEqual(results[0]["statute_id"], 1)
        self.assertEqual([item["rank"] for item in results], [1, 2])

    def test_rrf_deduplicates_and_ranks(self) -> None:
        first = [{"rank": 1, "score": 3, "statute_id": 1, "statute_name": "一"}]
        second = [{"rank": 1, "score": 2, "statute_id": 1, "statute_name": "一"}, {"rank": 2, "score": 1, "statute_id": 2, "statute_name": "二"}]
        fused = reciprocal_rank_fusion([first, second], top_k=2)
        self.assertEqual([row["statute_id"] for row in fused], [1, 2])
        self.assertEqual([row["rank"] for row in fused], [1, 2])

    def test_metrics_use_any_gold_statute(self) -> None:
        rows = [{"gold_statute_ids": [2, 3], "results": [{"statute_id": 3}, {"statute_id": 9}]}]
        metrics = retrieval_metrics(rows)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
