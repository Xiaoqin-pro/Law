from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.citation import build_citation_index  # noqa: E402
from citelaw.citation_v2 import extract_citations_v2  # noqa: E402


class CitationParserV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = build_citation_index(
            [
                {"statute_id": 1, "statute_name": "民法典第五百条", "statute_text": "x"},
                {"statute_id": 2, "statute_name": "民法典第三十九条", "statute_text": "x"},
                {"statute_id": 3, "statute_name": "刑法第二百条", "statute_text": "x"},
                {"statute_id": 4, "statute_name": "行政许可法第十二条", "statute_text": "x"},
                {"statute_id": 5, "statute_name": "劳动合同法第十四条", "statute_text": "x"},
                {
                    "statute_id": 6,
                    "statute_name": "最高人民法院关于适用《保险法》若干问题的解释(二)第二条",
                    "statute_text": "x",
                },
            ]
        )

    def test_unique_context_propagation(self) -> None:
        rows = extract_citations_v2("根据《民法典》第五百条，本法第三十九条也适用。", self.index)
        self.assertEqual([row["canonical_statute_id"] for row in rows], [1, 2])
        self.assertEqual(rows[1]["resolution_method"], "context_propagation")
        self.assertEqual(rows[1]["confidence"], "high")

    def test_two_law_adversarial_case_stays_unresolved(self) -> None:
        rows = extract_citations_v2("根据《民法典》第五百条和《刑法》第二百条，第一百条规定。", self.index)
        self.assertEqual([row["canonical_statute_id"] for row in rows], [1, 3, None])
        self.assertEqual(rows[-1]["parse_status"], "ambiguous")
        self.assertEqual(rows[-1]["resolution_method"], "unresolved")

    def test_article_only_without_antecedent_stays_unresolved(self) -> None:
        rows = extract_citations_v2("根据第十条规定。", self.index)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["canonical_statute_id"])
        self.assertEqual(rows[0]["parse_status"], "unresolved")

    def test_boundary_and_mild_malformed_forms_resolve_with_provenance(self) -> None:
        rows = extract_citations_v2("行政许可法》第12条；《劳动合同法第14条。", self.index)
        self.assertEqual([row["canonical_statute_id"] for row in rows], [4, 5])
        self.assertEqual(rows[0]["resolution_method"], "boundary_repair")
        self.assertTrue(rows[0]["repair_applied"])
        self.assertEqual(rows[1]["resolution_method"], "malformed_citation_repair")
        self.assertTrue(rows[1]["repair_applied"])
        self.assertEqual(rows[1]["article_number"], "14")

    def test_nested_title_brackets_normalize_without_changing_article(self) -> None:
        rows = extract_citations_v2(
            "最高人民法院关于适用〈保险法〉若干问题的解释（二）》第二条。", self.index
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["canonical_statute_id"], 6)
        self.assertEqual(rows[0]["resolution_method"], "long_title_normalization")
        self.assertEqual(rows[0]["article_number"], "2")


if __name__ == "__main__":
    unittest.main()
