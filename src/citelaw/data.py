"""LeCoQA loading, normalization, validation, and statistics utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REQUIRED_QA_FIELDS = {
    "query_id",
    "question",
    "reference_answer",
    "gold_statute_ids",
    "gold_statute_names",
    "gold_statute_texts",
}
REQUIRED_CORPUS_FIELDS = {"statute_id", "statute_name", "statute_text"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _as_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc


def normalize_corpus(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, row in enumerate(rows):
        missing = {"id", "name", "content"} - set(row)
        if missing:
            raise ValueError(f"Corpus row {index} is missing fields: {sorted(missing)}")
        statute_id = _as_int(row["id"], "corpus.id")
        if statute_id in seen_ids:
            raise ValueError(f"Duplicate corpus statute id: {statute_id}")
        seen_ids.add(statute_id)
        normalized.append(
            {
                "statute_id": statute_id,
                "statute_name": str(row["name"]),
                "statute_text": str(row["content"]),
            }
        )
    return normalized


def normalize_qa(
    rows: Sequence[Mapping[str, Any]],
    corpus_by_id: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, row in enumerate(rows):
        required_source_fields = {"问题", "答案文本", "match_id", "match_name", "query_id"}
        missing = required_source_fields - set(row)
        if missing:
            raise ValueError(f"QA row {index} is missing fields: {sorted(missing)}")

        query_id = _as_int(row["query_id"], "query_id")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query id: {query_id}")
        seen_ids.add(query_id)

        statute_ids = [_as_int(value, "match_id") for value in row["match_id"]]
        statute_names = [str(value) for value in row["match_name"]]
        if len(statute_ids) != len(statute_names):
            raise ValueError(f"QA row {query_id} has mismatched match_id/match_name lengths")

        gold_texts: List[str] = []
        gold_corpus_names: List[str] = []
        gold_name_matches_corpus: List[bool] = []
        for statute_id, expected_name in zip(statute_ids, statute_names):
            if statute_id not in corpus_by_id:
                raise ValueError(
                    f"QA row {query_id} references missing corpus statute id {statute_id}"
                )
            statute = corpus_by_id[statute_id]
            gold_texts.append(statute["statute_text"])
            gold_corpus_names.append(statute["statute_name"])
            gold_name_matches_corpus.append(expected_name == statute["statute_name"])

        normalized.append(
            {
                "query_id": query_id,
                "question": str(row["问题"]),
                "reference_answer": str(row["答案文本"]),
                "gold_statute_ids": statute_ids,
                "gold_statute_names": statute_names,
                "gold_statute_texts": gold_texts,
                "gold_corpus_names": gold_corpus_names,
                "gold_name_matches_corpus": gold_name_matches_corpus,
            }
        )
    return normalized


def validate_normalized(
    qa_rows: Sequence[Mapping[str, Any]],
    corpus_rows: Sequence[Mapping[str, Any]],
    *,
    split_name: Optional[str] = None,
) -> None:
    prefix = f"{split_name}: " if split_name else ""
    corpus_ids = [row["statute_id"] for row in corpus_rows]
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError(f"{prefix}corpus statute_id values are not unique")

    query_ids = [row["query_id"] for row in qa_rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"{prefix}query_id values are not unique")

    corpus_by_id = {row["statute_id"]: row for row in corpus_rows}
    for row in qa_rows:
        missing = REQUIRED_QA_FIELDS - set(row)
        if missing:
            raise ValueError(f"{prefix}query {row.get('query_id')} missing {sorted(missing)}")
        if not str(row["question"]).strip():
            raise ValueError(f"{prefix}query {row['query_id']} has an empty question")
        if not str(row["reference_answer"]).strip():
            raise ValueError(f"{prefix}query {row['query_id']} has an empty answer")
        ids = row["gold_statute_ids"]
        names = row["gold_statute_names"]
        texts = row["gold_statute_texts"]
        if not (len(ids) == len(names) == len(texts)):
            raise ValueError(f"{prefix}query {row['query_id']} has misaligned gold fields")
        for statute_id in ids:
            if statute_id not in corpus_by_id:
                raise ValueError(
                    f"{prefix}query {row['query_id']} has missing gold statute {statute_id}"
                )

    for row in corpus_rows:
        missing = REQUIRED_CORPUS_FIELDS - set(row)
        if missing:
            raise ValueError(f"{prefix}corpus row missing {sorted(missing)}")
        if not str(row["statute_name"]).strip() or not str(row["statute_text"]).strip():
            raise ValueError(f"{prefix}corpus statute {row['statute_id']} has empty text")


def _mean(values: Sequence[int]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def make_statistics(
    all_qa: Sequence[Mapping[str, Any]],
    train_qa: Sequence[Mapping[str, Any]],
    test_qa: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    mismatches = [
        {
            "query_id": row["query_id"],
            "statute_ids": [
                statute_id
                for statute_id, matches in zip(
                    row["gold_statute_ids"], row["gold_name_matches_corpus"]
                )
                if not matches
            ],
        }
        for row in all_qa
        if not all(row["gold_name_matches_corpus"])
    ]
    mismatch_pairs = sum(len(item["statute_ids"]) for item in mismatches)
    return {
        "dataset": "LeCoQA",
        "qa_total": len(all_qa),
        "train_count": len(train_qa),
        "test_count": len(test_qa),
        "corpus_count": len(corpus),
        "average_gold_statutes_per_question": _mean(
            [len(row["gold_statute_ids"]) for row in all_qa]
        ),
        "average_question_length": _mean([len(row["question"]) for row in all_qa]),
        "average_answer_length": _mean([len(row["reference_answer"]) for row in all_qa]),
        "quality": {
            "gold_name_mismatch_qa_count": len(mismatches),
            "gold_name_mismatch_pair_count": mismatch_pairs,
            "gold_name_mismatch_examples": mismatches[:20],
            "note": (
                "gold_statute_names are preserved from LeCoQA. "
                "gold_statute_texts and gold_corpus_names are resolved by statute ID."
            ),
        },
        "query_id_ranges": {
            "all": [min(row["query_id"] for row in all_qa), max(row["query_id"] for row in all_qa)],
            "train": [min(row["query_id"] for row in train_qa), max(row["query_id"] for row in train_qa)],
            "test": [min(row["query_id"] for row in test_qa), max(row["query_id"] for row in test_qa)],
        },
    }


def choose_samples(rows: Sequence[Mapping[str, Any]], count: int = 10, seed: int = 20260813) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    selected = list(rows) if len(rows) <= count else rng.sample(list(rows), count)
    return [dict(row) for row in selected]


def load_raw_dataset(raw_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    data_root = raw_root / "data"
    corpus = normalize_corpus(load_jsonl(data_root / "corpus.jsonl"))
    corpus_by_id = {row["statute_id"]: row for row in corpus}
    all_qa = normalize_qa(load_json(data_root / "queries.json"), corpus_by_id)
    train_qa = normalize_qa(load_json(data_root / "example" / "train.json"), corpus_by_id)
    test_qa = normalize_qa(load_json(data_root / "example" / "test.json"), corpus_by_id)
    return all_qa, train_qa, test_qa, corpus
