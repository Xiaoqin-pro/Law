from __future__ import annotations

import csv
import json
import re
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.citation import build_citation_index, normalize_article_number  # noqa: E402
from citelaw.citation_v2 import PARSER_VERSION, extract_citations_v2  # noqa: E402


ARTICLE_IN_RAW = re.compile(r"第\s*([0-9零一二三四五六七八九十百千万亿〇两]+)\s*条")


class CitationParserV2PriorityRegressionTests(unittest.TestCase):
    """Keep every model-assisted priority case in the deterministic suite.

    The model-assisted workbook is reference-only.  The expected unresolved
    cases below encode the v2 safety rule: an ambiguous or context-free bare
    article must stay unresolved even if a model-assisted reviewer suggested a
    possible interpretation.
    """

    @classmethod
    def setUpClass(cls) -> None:
        corpus = [
            json.loads(line)
            for line in (PROJECT_ROOT / "data/processed/corpus.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.index = build_citation_index(corpus)
        sample = json.loads((PROJECT_ROOT / "reports/phase3_5/parser_qa_data.json").read_text(encoding="utf-8"))["rows"]
        cls.sample_by_id = {int(row["audit_unit_id"]): row for row in sample}
        with (PROJECT_ROOT / "reports/phase3_5/model_assisted_parser_qa/parser_qa_model_assisted_priority_review.csv").open(encoding="utf-8-sig", newline="") as handle:
            cls.priority = list(csv.DictReader(handle))

    def test_all_40_priority_cases_have_explicit_v2_expectations(self) -> None:
        self.assertEqual([int(row["audit_unit_id"]) for row in self.priority], list(range(41, 81)))
        conservative_unresolved = {51, 57, 60, 78, 80}
        for item in self.priority:
            audit_id = int(item["audit_unit_id"])
            with self.subTest(audit_unit_id=audit_id):
                sample = self.sample_by_id[audit_id]
                self.assertEqual(item["query_id"], str(sample["query_id"]))
                answer = sample["raw_answer"]
                citations = extract_citations_v2(answer, self.index)
                article_match = ARTICLE_IN_RAW.search(item["raw_citation"])
                self.assertIsNotNone(article_match)
                article_number = normalize_article_number(article_match.group(1))
                target = [citation for citation in citations if citation.get("article_number") == article_number]
                self.assertTrue(target, f"priority case {audit_id} was not extracted")
                self.assertTrue(all(citation.get("parser_version") == PARSER_VERSION for citation in target))
                has_high = any(citation.get("parse_status") == "resolved_unique" and citation.get("confidence") == "high" for citation in target)
                if audit_id in conservative_unresolved:
                    self.assertFalse(has_high, f"priority case {audit_id} must remain unresolved under the safety rule")
                else:
                    self.assertTrue(has_high, f"priority case {audit_id} lost its deterministic v2 resolution")


if __name__ == "__main__":
    unittest.main()
