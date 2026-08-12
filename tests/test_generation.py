from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.generation import build_evidence, render_prompt  # noqa: E402


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query = {"query_id": 1, "question": "谁可以成为个体工商户？"}
        self.corpus = {1: {"statute_id": 1, "statute_name": "法条一", "statute_text": "正文一"}}

    def test_evidence_uses_retrieved_ids_and_top_k(self) -> None:
        row = {"results": [{"statute_id": 1}, {"statute_id": 99}]}
        evidence = build_evidence(row, self.corpus, top_k=2)
        self.assertIn("法条一", evidence)
        self.assertNotIn("99", evidence)

    def test_direct_prompt_contains_question(self) -> None:
        prompt = render_prompt("direct", self.query, prompt_templates={"direct": "Q={question}", "rag": "Q={question} E={evidence}"})
        self.assertEqual(prompt, "Q=谁可以成为个体工商户？")

    def test_rag_prompt_contains_evidence(self) -> None:
        prompt = render_prompt("hybrid", self.query, prompt_templates={"direct": "", "rag": "Q={question} E={evidence}"}, evidence="法条正文")
        self.assertIn("法条正文", prompt)


if __name__ == "__main__":
    unittest.main()
