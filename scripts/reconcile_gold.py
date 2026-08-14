"""Reconcile LeCoQA gold evidence without relying on field position."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION_FIELD = "\u95ee\u9898"
RELATED_FIELD = "\u76f8\u5173\u6cd5\u89c4"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_match_text(value: Any) -> str:
    """Normalize only deterministic formatting differences.

    No semantic normalization, fuzzy matching, or model judgment is used.
    """

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_index(corpus_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[Any, List[int]]]:
    by_pair: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    by_text: Dict[str, List[int]] = defaultdict(list)
    by_name: Dict[str, List[int]] = defaultdict(list)
    for row in corpus_rows:
        statute_id = int(row["statute_id"])
        name = normalize_match_text(row["statute_name"])
        text = normalize_match_text(row["statute_text"])
        by_pair[(name, text)].append(statute_id)
        by_text[text].append(statute_id)
        by_name[name].append(statute_id)
    return {"pair": dict(by_pair), "text": dict(by_text), "name": dict(by_name)}


def resolve_related_statute(
    related_name: str,
    related_text: str,
    index: Mapping[str, Mapping[Any, List[int]]],
) -> Dict[str, Any]:
    normalized_name = normalize_match_text(related_name)
    normalized_text = normalize_match_text(related_text)
    pair_candidates = list(index["pair"].get((normalized_name, normalized_text), []))
    text_candidates = list(index["text"].get(normalized_text, []))
    name_candidates = list(index["name"].get(normalized_name, []))

    if len(pair_candidates) == 1:
        level = "A_exact_name_and_text"
        candidates = pair_candidates
    elif len(text_candidates) == 1:
        level = "B_exact_text_unique"
        candidates = text_candidates
    elif len(name_candidates) == 1:
        level = "C_exact_name_unique"
        candidates = name_candidates
    elif pair_candidates or text_candidates or name_candidates:
        level = "D_ambiguous"
        candidates = sorted(set(pair_candidates + text_candidates + name_candidates))
    else:
        level = "D_unresolved"
        candidates = []

    return {
        "related_statute_name": str(related_name),
        "related_statute_text": str(related_text),
        "normalized_name": normalized_name,
        "normalized_text": normalized_text,
        "resolution_level": level,
        "resolved_statute_id": candidates[0] if len(candidates) == 1 and level.startswith(("A_", "B_", "C_")) else None,
        "candidate_ids": candidates,
        "evidence": {
            "pair_candidate_ids": pair_candidates,
            "text_candidate_ids": text_candidates,
            "name_candidate_ids": name_candidates,
        },
    }


def _raw_position_is_clean(
    raw_ids: Sequence[int],
    raw_names: Sequence[str],
    related_names: Sequence[str],
    corpus_by_id: Mapping[int, Mapping[str, Any]],
) -> bool:
    if len(raw_ids) != len(raw_names) or len(raw_ids) != len(related_names):
        return False
    for index, statute_id in enumerate(raw_ids):
        corpus_row = corpus_by_id.get(statute_id)
        if corpus_row is None:
            return False
        corpus_name = str(corpus_row["statute_name"])
        if raw_names[index] != corpus_name or related_names[index] != corpus_name:
            return False
    return True


def reconcile_row(
    row: Mapping[str, Any],
    corpus_by_id: Mapping[int, Mapping[str, Any]],
    index: Mapping[str, Mapping[Any, List[int]]],
) -> Dict[str, Any]:
    raw_ids = [int(value) for value in row.get("match_id", [])]
    raw_names = [str(value) for value in row.get("match_name", [])]
    related = row.get(RELATED_FIELD, {}) or {}
    related_items = [
        resolve_related_statute(name, text, index)
        for name, text in related.items()
    ]
    resolved_ids = [
        int(item["resolved_statute_id"])
        for item in related_items
        if item["resolved_statute_id"] is not None
    ]
    reconciled_ids = sorted(set(resolved_ids))
    raw_set = set(raw_ids)
    reconciled_set = set(reconciled_ids)
    all_high_confidence = bool(related_items) and all(
        item["resolution_level"].startswith(("A_", "B_")) for item in related_items
    )
    all_resolved = bool(related_items) and all(
        item["resolved_statute_id"] is not None for item in related_items
    )
    related_names = [str(name) for name in related.keys()]
    raw_clean = _raw_position_is_clean(raw_ids, raw_names, related_names, corpus_by_id)

    if not related_items:
        status = "unresolved"
    elif all_resolved and raw_set == reconciled_set:
        status = "clean" if raw_clean and raw_ids == resolved_ids else "order_only_conflict"
    elif all_resolved:
        status = "true_set_conflict"
    elif resolved_ids:
        status = "partially_resolved"
    elif any(item["resolution_level"] == "D_ambiguous" for item in related_items):
        status = "ambiguous"
    else:
        status = "unresolved"

    canonical_ids = reconciled_ids if status in {"clean", "order_only_conflict"} else []
    return {
        "query_id": int(row["query_id"]),
        "question": str(row.get(QUESTION_FIELD, "")),
        "raw_match_ids": raw_ids,
        "raw_match_names": raw_names,
        "raw_related_statute_names": related_names,
        "related_statute_resolutions": related_items,
        "reconciled_gold_ids": reconciled_ids,
        "canonical_gold_ids": canonical_ids,
        "reconciliation_status": status,
        "raw_position_audit": {
            "raw_clean": raw_clean,
            "raw_id_set": sorted(raw_set),
            "reconciled_id_set": sorted(reconciled_set),
            "raw_set_equals_reconciled_set": raw_set == reconciled_set,
            "all_related_high_confidence": all_high_confidence,
            "all_related_resolved": all_resolved,
        },
    }


def reconcile_dataset(
    raw_rows: Sequence[Mapping[str, Any]],
    corpus_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    corpus_by_id = {int(row["statute_id"]): row for row in corpus_rows}
    index = build_index(corpus_rows)
    rows = [reconcile_row(row, corpus_by_id, index) for row in raw_rows]
    statuses = Counter(row["reconciliation_status"] for row in rows)
    levels = Counter(
        item["resolution_level"]
        for row in rows
        for item in row["related_statute_resolutions"]
    )
    raw_clean_count = sum(bool(row["raw_position_audit"]["raw_clean"]) for row in rows)
    high_confidence_count = sum(
        bool(row["raw_position_audit"]["all_related_high_confidence"]) for row in rows
    )
    all_resolved_count = sum(
        bool(row["raw_position_audit"]["all_related_resolved"]) for row in rows
    )
    high_confidence_agreement_count = sum(
        row["reconciliation_status"] in {"clean", "order_only_conflict"}
        and row["raw_position_audit"]["all_related_high_confidence"]
        for row in rows
    )
    actual_unresolved_count = sum(
        row["reconciliation_status"] in {"ambiguous", "unresolved", "true_set_conflict"}
        for row in rows
    )
    return {
        "reconciliation_version": "phase3_2_v1_nonpositional",
        "dataset": "LeCoQA",
        "query_count": len(rows),
        "normalization": {
            "operations": ["CRLF/LF normalization", "full-width space normalization", "whitespace collapse", "trim"],
            "semantic_matching": False,
            "llm_matching": False,
        },
        "resolution_levels": {
            "A_exact_name_and_text": "high_confidence",
            "B_exact_text_unique": "high_confidence",
            "C_exact_name_unique": "reported_separately_not_high_confidence",
            "D_ambiguous": "not_resolved",
            "D_unresolved": "not_resolved",
        },
        "counts": {
            "reconciliation_status": dict(sorted(statuses.items())),
            "resolution_level": dict(sorted(levels.items())),
            "raw_position_clean": raw_clean_count,
            "all_related_resolved": all_resolved_count,
            "high_confidence_all_related": high_confidence_count,
            "high_confidence_agreement": high_confidence_agreement_count,
            "not_strict_high_confidence": len(rows) - high_confidence_count,
            "actual_unresolved_or_conflicted": actual_unresolved_count,
        },
        "rows": rows,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-query-file", type=Path, default=PROJECT_ROOT / "data/raw/LeCoQA/data/example/test.json")
    parser.add_argument("--corpus-file", type=Path, default=PROJECT_ROOT / "data/processed/corpus.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/data/lecoqa_gold_reconciliation.json")
    parser.add_argument("--cases-output", type=Path, default=PROJECT_ROOT / "reports/phase3_2/gold_reconciliation_cases.jsonl")
    args = parser.parse_args()
    raw_rows = json.loads(args.raw_query_file.read_text(encoding="utf-8"))
    corpus_rows = load_jsonl(args.corpus_file)
    payload = reconcile_dataset(raw_rows, corpus_rows)
    write_json(args.output, payload)
    args.cases_output.parent.mkdir(parents=True, exist_ok=True)
    with args.cases_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in payload["rows"]:
            if row["reconciliation_status"] != "clean":
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"query_count": payload["query_count"], "counts": payload["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
