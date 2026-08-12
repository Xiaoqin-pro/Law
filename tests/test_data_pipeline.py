from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.data import (  # noqa: E402
    normalize_corpus,
    normalize_qa,
    validate_normalized,
)


class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_corpus = [
            {"id": 1, "name": "法条一", "content": "正文一"},
            {"id": 2, "name": "法条二", "content": "正文二"},
        ]
        self.corpus = normalize_corpus(self.raw_corpus)
        self.corpus_by_id = {row["statute_id"]: row for row in self.corpus}
        self.raw_qa = [
            {
                "问题": "问题一？",
                "答案文本": "答案一",
                "match_id": [1],
                "match_name": ["法条一"],
                "query_id": 10,
            }
        ]

    def test_normalize_qa_keeps_required_fields(self) -> None:
        rows = normalize_qa(self.raw_qa, self.corpus_by_id)
        self.assertEqual(rows[0]["query_id"], 10)
        self.assertEqual(rows[0]["gold_statute_texts"], ["正文一"])
        validate_normalized(rows, self.corpus)

    def test_name_mismatch_is_preserved_and_reported(self) -> None:
        rows = normalize_qa(
            [dict(self.raw_qa[0], match_name=["上游错位名称"])], self.corpus_by_id
        )
        self.assertEqual(rows[0]["gold_statute_names"], ["上游错位名称"])
        self.assertEqual(rows[0]["gold_corpus_names"], ["法条一"])
        self.assertEqual(rows[0]["gold_name_matches_corpus"], [False])

    def test_duplicate_query_id_is_rejected(self) -> None:
        duplicate = self.raw_qa + [dict(self.raw_qa[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate query id"):
            normalize_qa(duplicate, self.corpus_by_id)

    def test_duplicate_statute_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate corpus statute id"):
            normalize_corpus(self.raw_corpus + [self.raw_corpus[0]])

    def test_missing_gold_statute_is_rejected(self) -> None:
        bad = [dict(self.raw_qa[0], match_id=[999], match_name=["不存在"])]
        with self.assertRaisesRegex(ValueError, "missing corpus statute"):
            normalize_qa(bad, self.corpus_by_id)

    def test_empty_question_is_rejected(self) -> None:
        rows = [dict(normalize_qa(self.raw_qa, self.corpus_by_id)[0], question=" ")]
        with self.assertRaisesRegex(ValueError, "empty question"):
            validate_normalized(rows, self.corpus, split_name="test")

    def test_output_is_utf8_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "row.json"
            output.write_text(json.dumps({"问题": "中文"}, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["问题"], "中文")


if __name__ == "__main__":
    unittest.main()
