from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_citation_analysis import occurrence_gold_match_rate  # noqa: E402

from citelaw.citation import (  # noqa: E402
    build_citation_index,
    extract_citations,
    extract_statute_id_markers,
    normalize_article_number,
    normalize_law_name,
    split_statute_name,
)


class CitationParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = build_citation_index(
            [
                {
                    "statute_id": 1,
                    "statute_name": "中华人民共和国民法典第五百七十七条",
                    "statute_text": "违约责任。",
                },
                {
                    "statute_id": 2,
                    "statute_name": "中华人民共和国刑法第二百六十三条",
                    "statute_text": "抢劫罪。",
                },
                {
                    "statute_id": 3,
                    "statute_name": "中华人民共和国刑法第二百六十三条之一",
                    "statute_text": "特别规定。",
                },
            ]
        )

    def test_chinese_and_arabic_article_normalization(self) -> None:
        self.assertEqual(normalize_article_number("第五百七十七"), "577")
        self.assertEqual(normalize_article_number("577"), "577")
        self.assertEqual(normalize_article_number("二百六十三之一"), "263之1")
        self.assertEqual(normalize_law_name("《民法典》"), "民法典")

    def test_corpus_name_split(self) -> None:
        self.assertEqual(
            split_statute_name("中华人民共和国刑法第二百六十三条之一"),
            ("中华人民共和国刑法", "263之1"),
        )

    def test_named_citations_and_aliases(self) -> None:
        citations = extract_citations(
            "根据《民法典》第577条及中华人民共和国刑法第二百六十三条之一。",
            self.index,
        )
        self.assertEqual([item["canonical_statute_id"] for item in citations], [1, 3])
        self.assertTrue(all(item["parse_status"] == "resolved_unique" for item in citations))

    def test_article_only_is_retained_as_unresolved(self) -> None:
        citations = extract_citations("仅写第577条，不能从文本推断法律名称。", self.index)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["parse_status"], "unresolved")
        self.assertEqual(citations[0]["resolution_method"], "law_name_missing")

    def test_missing_law_and_multiple_citations(self) -> None:
        citations = extract_citations(
            "依据民法典第577条、刑法第263条和第999条。", self.index
        )
        self.assertEqual(len(citations), 3)
        self.assertEqual([item["canonical_statute_id"] for item in citations], [1, 2, None])

    def test_subarticle_is_not_confused_with_parent_article(self) -> None:
        citations = extract_citations("《刑法》第263条之一。", self.index)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["canonical_statute_id"], 3)
        self.assertEqual(citations[0]["article_number"], "263之1")

    def test_unknown_law_is_unresolved(self) -> None:
        citations = extract_citations("依据《不存在的法》第1条。", self.index)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["parse_status"], "unresolved")
        self.assertEqual(citations[0]["resolution_method"], "law_name_not_in_corpus")

    def test_internal_evidence_marker_is_a_separate_audit_field(self) -> None:
        text = "根据[法条ID 1]《民法典》第577条。"
        self.assertEqual(len(extract_citations(text, self.index)), 1)
        self.assertEqual(extract_statute_id_markers(text), [1])

    def test_gold_citation_match_rate_is_occurrence_level(self) -> None:
        duplicate_gold = extract_citations(
            "《民法典》第577条、《民法典》第577条。", self.index
        )
        mixed_gold_and_wrong = extract_citations(
            "《民法典》第577条、《刑法》第263条。", self.index
        )
        self.assertEqual(occurrence_gold_match_rate(duplicate_gold, {1}), 1.0)
        self.assertEqual(occurrence_gold_match_rate(mixed_gold_and_wrong, {1}), 0.5)


if __name__ == "__main__":
    unittest.main()
