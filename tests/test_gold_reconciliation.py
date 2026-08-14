import unittest

from scripts.reconcile_gold import normalize_match_text, reconcile_dataset


class GoldReconciliationTests(unittest.TestCase):
    def test_normalization_only_changes_formatting(self):
        self.assertEqual(normalize_match_text("  a\r\n\u3000 b  "), "a b")
        self.assertNotEqual(normalize_match_text("第１条"), normalize_match_text("第一条"))

    def test_order_only_conflict_uses_evidence_set(self):
        corpus = [
            {"statute_id": 1, "statute_name": "法甲", "statute_text": "甲文"},
            {"statute_id": 2, "statute_name": "法乙", "statute_text": "乙文"},
        ]
        raw = [{
            "query_id": 1,
            "问题": "问",
            "match_id": [1, 2],
            "match_name": ["法甲", "法乙"],
            "相关法规": {"法乙": "乙文\n", "法甲": "甲文\n"},
        }]
        row = reconcile_dataset(raw, corpus)["rows"][0]
        self.assertEqual(row["reconciliation_status"], "order_only_conflict")
        self.assertEqual(row["reconciled_gold_ids"], [1, 2])
        self.assertEqual(row["canonical_gold_ids"], [1, 2])

    def test_true_set_conflict_is_not_silently_canonicalized(self):
        corpus = [
            {"statute_id": 1, "statute_name": "法甲", "statute_text": "甲文"},
            {"statute_id": 2, "statute_name": "法乙", "statute_text": "乙文"},
        ]
        raw = [{
            "query_id": 2,
            "问题": "问",
            "match_id": [1],
            "match_name": ["法甲"],
            "相关法规": {"法乙": "乙文"},
        }]
        row = reconcile_dataset(raw, corpus)["rows"][0]
        self.assertEqual(row["reconciliation_status"], "true_set_conflict")
        self.assertEqual(row["canonical_gold_ids"], [])
        self.assertEqual(row["reconciled_gold_ids"], [2])


if __name__ == "__main__":
    unittest.main()
