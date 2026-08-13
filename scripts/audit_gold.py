"""Audit LeCoQA gold IDs, names, related-statute names, and corpus evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.data import load_jsonl  # noqa: E402


def load_raw_queries(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_row(row: Mapping[str, Any], corpus_by_id: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    ids = [int(value) for value in row["match_id"]]
    names = [str(value) for value in row["match_name"]]
    related = row.get("相关法规", {})
    related_names = [str(value) for value in related.keys()]
    pairs: List[Dict[str, Any]] = []
    id_evidence_conflict = False
    name_mismatch = False
    unresolved = False
    for index, statute_id in enumerate(ids):
        expected_name = names[index] if index < len(names) else ""
        related_name = expected_name if index >= len(related_names) else related_names[index]
        corpus_row = corpus_by_id.get(statute_id)
        corpus_name = str(corpus_row["statute_name"]) if corpus_row else None
        corpus_text = str(corpus_row["statute_text"]) if corpus_row else None
        id_exists = corpus_row is not None
        name_matches = expected_name == corpus_name if id_exists else False
        related_matches = related_name == corpus_name if id_exists else False
        name_mismatch = name_mismatch or not name_matches
        id_evidence_conflict = id_evidence_conflict or (id_exists and not related_matches)
        unresolved = unresolved or not id_exists
        pairs.append({
            "position": index,
            "statute_id": statute_id,
            "match_name": expected_name,
            "related_statute_name": related_name,
            "corpus_statute_name": corpus_name,
            "corpus_statute_text": corpus_text,
            "id_exists_in_corpus": id_exists,
            "match_name_equals_corpus": name_matches,
            "related_name_equals_corpus": related_matches,
        })
    if unresolved:
        classification = "unresolved"
    elif id_evidence_conflict:
        classification = "id_evidence_conflict"
    elif name_mismatch:
        classification = "name_mismatch_only"
    else:
        classification = "clean"
    return {
        "query_id": int(row["query_id"]),
        "question": str(row["问题"]),
        "classification": classification,
        "gold_statute_ids": ids,
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-query-file", type=Path, default=PROJECT_ROOT / "data/raw/LeCoQA/data/example/test.json")
    parser.add_argument("--corpus-file", type=Path, default=PROJECT_ROOT / "data/processed/corpus.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/data/lecoqa_gold_audit.json")
    parser.add_argument("--sample-output", type=Path, default=PROJECT_ROOT / "reports/data/lecoqa_gold_audit_sample.jsonl")
    args = parser.parse_args()
    corpus = {int(row["statute_id"]): row for row in load_jsonl(args.corpus_file)}
    audited = [audit_row(row, corpus) for row in load_raw_queries(args.raw_query_file)]
    counts = Counter(row["classification"] for row in audited)
    payload = {
        "audit_version": "v1_id_name_related_corpus",
        "dataset": "LeCoQA",
        "query_count": len(audited),
        "classification_counts": dict(sorted(counts.items())),
        "rows": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_rows = [row for row in audited if row["classification"] != "clean"][:20]
    with args.sample_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"query_count": len(audited), "classification_counts": dict(sorted(counts.items()))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
