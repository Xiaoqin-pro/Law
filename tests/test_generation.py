from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.generation import build_evidence, pack_evidence_to_budget, render_prompt  # noqa: E402


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def __call__(self, text, add_special_tokens=False, truncation=False, return_offsets_mapping=False):
        result = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


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


    def test_budget_packing_keeps_complete_ranked_blocks(self) -> None:
        row = {"results": [{"statute_id": 1}, {"statute_id": 2}]}
        corpus = {
            1: {"statute_id": 1, "statute_name": "A", "statute_text": "one"},
            2: {"statute_id": 2, "statute_name": "B", "statute_text": "two"},
        }
        packed = pack_evidence_to_budget(
            self.query,
            row,
            corpus,
            tokenizer=CharacterTokenizer(),
            prompt_templates={"direct": "", "rag": "Q={question} E={evidence}"},
            top_k=2,
            max_input_tokens=45,
        )
        self.assertEqual(packed["included_statute_ids"], [1])
        self.assertEqual(packed["evidence_packing_version"], "phase3_2_ranked_complete_blocks_v1")


if __name__ == "__main__":
    unittest.main()
