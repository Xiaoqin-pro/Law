"""Independent deterministic checks for Parser v2 JSONL records.

This checker does not call ``extract_citations`` or ``extract_citations_v2``.
It validates the emitted provenance against a separately constructed
law/article lookup and simple raw-text article checks.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


ROOT = Path(__file__).resolve().parents[1]
ARTICLE_RE = re.compile(
    r"第\s*(?P<article>[0-9０-９一二三四五六七八九十百千万亿零〇两]+)\s*条"
    r"(?:\s*之\s*(?P<suffix>[0-9０-９一二三四五六七八九十百千万亿零〇两]+))?"
)
DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}


def normalize_number(value: str) -> str | None:
    value = unicodedata.normalize("NFKC", value).replace(" ", "")
    if value.isdigit():
        return str(int(value))
    if not value or any(char not in DIGITS and char not in UNITS for char in value):
        return None
    if not any(char in UNITS for char in value):
        return "".join(str(DIGITS[char]) for char in value)
    total = 0
    section = 0
    number = 0
    for char in value:
        if char in DIGITS:
            number = DIGITS[char]
        else:
            unit = UNITS[char]
            if unit in (10000, 100000000):
                section += (number or 1) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
    return str(total + section + number)


def name_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = value.translate(str.maketrans({"〈": "《", "〉": "》", "(" : "（", ")": "）"}))
    value = re.sub(r"\s+", "", value)
    return value.strip("《》〈〉（）()[]{}")


def statute_key(statute_name: str) -> Tuple[str, str] | None:
    match = re.search(
        r"第(?P<article>[0-9０-９一二三四五六七八九十百千万亿零〇两]+)条"
        r"(?:之(?P<suffix>[0-9０-９一二三四五六七八九十百千万亿零〇两]+))?$",
        str(statute_name),
    )
    if not match:
        return None
    article = normalize_number(match.group("article"))
    if match.group("suffix"):
        suffix = normalize_number(match.group("suffix"))
        article = f"{article}之{suffix}" if article and suffix else None
    return (name_key(str(statute_name)[: match.start()]), article) if article else None


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_lookup(corpus_path: Path) -> tuple[Dict[Tuple[str, str], list[int]], Dict[int, Tuple[str, str]], Dict[str, set[str]]]:
    key_to_ids: Dict[Tuple[str, str], list[int]] = {}
    id_to_key: Dict[int, Tuple[str, str]] = {}
    aliases_by_law: Dict[str, set[str]] = {}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = statute_key(row["statute_name"])
        if key is None:
            continue
        statute_id = int(row["statute_id"])
        key_to_ids.setdefault(key, []).append(statute_id)
        id_to_key[statute_id] = key
        law = key[0]
        aliases_by_law.setdefault(law, {law})
        prefix = "\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd"
        if law.startswith(prefix) and law[len(prefix):]:
            aliases_by_law[law].add(law[len(prefix):])
    return key_to_ids, id_to_key, aliases_by_law


def select_nonoverlapping_laws(context: str, canonical_laws: Iterable[str], aliases_by_law: Mapping[str, Iterable[str]]) -> set[str]:
    folded = name_key(context)
    matches = []
    for law in canonical_laws:
        for alias in aliases_by_law.get(law, (law,)):
            law_key = name_key(alias)
            position = folded.find(law_key)
            if law_key and position >= 0:
                matches.append((position, position + len(law_key), law))
    selected: list[tuple[int, int, str]] = []
    for item in sorted(matches, key=lambda value: (value[0], -(value[1] - value[0]))):
        if any(left < item[1] and item[0] < right for left, right, _ in selected):
            continue
        selected.append(item)
    return {item[2] for item in selected}


def audit_records(records: Iterable[Mapping[str, Any]], corpus_path: Path) -> Dict[str, Any]:
    key_to_ids, id_to_key, aliases_by_law = build_lookup(corpus_path)
    canonical_laws = {key[0] for key in key_to_ids}
    conflicts: list[Dict[str, Any]] = []
    method_counts = Counter()
    checked_high = 0
    checked_non_high = 0
    for row in records:
        for citation in row.get("citations", []):
            method = citation.get("resolution_method")
            method_counts[method or "<blank>"] += 1
            if citation.get("parse_status") != "resolved_unique":
                continue
            if citation.get("confidence") == "high":
                checked_high += 1
            else:
                checked_non_high += 1
                continue
            citation_id = citation.get("canonical_statute_id")
            law = citation.get("normalized_law_name")
            article = citation.get("article_number")
            key = (name_key(str(law or "")), str(article or ""))
            ids = key_to_ids.get(key, [])
            if citation_id is None or len(ids) != 1 or int(citation_id) != ids[0]:
                conflicts.append({
                    "query_id": row.get("query_id"),
                    "model": row.get("model"),
                    "method": row.get("method"),
                    "raw_citation": citation.get("raw_citation"),
                    "resolution_method": method,
                    "reason": "resolved_key_not_unique_or_id_mismatch",
                    "expected_ids": ids,
                    "observed_id": citation_id,
                })
                continue
            raw_match = ARTICLE_RE.search(str(citation.get("raw_citation") or ""))
            if raw_match:
                raw_article = normalize_number(raw_match.group("article"))
                if raw_match.group("suffix"):
                    raw_suffix = normalize_number(raw_match.group("suffix"))
                    raw_article = f"{raw_article}之{raw_suffix}" if raw_article and raw_suffix else None
                if raw_article != str(article):
                    conflicts.append({
                        "query_id": row.get("query_id"),
                        "model": row.get("model"),
                        "method": row.get("method"),
                        "raw_citation": citation.get("raw_citation"),
                        "resolution_method": method,
                        "reason": "article_number_changed_by_repair",
                        "raw_article": raw_article,
                        "observed_article": article,
                    })
            if method == "context_propagation":
                context = str(citation.get("context_source_text") or "")
                target = name_key(str(law or ""))
                found = select_nonoverlapping_laws(context, canonical_laws, aliases_by_law)
                if target not in found or found != {target}:
                    conflicts.append({
                        "query_id": row.get("query_id"),
                        "model": row.get("model"),
                        "method": row.get("method"),
                        "raw_citation": citation.get("raw_citation"),
                        "resolution_method": method,
                        "reason": "context_antecedent_not_unique_in_independent_check",
                        "target_law": law,
                        "found_laws": sorted(found),
                    })
    return {
        "audit_version": "parser_v2_independent_deterministic_audit_v1",
        "method": "independent corpus key reconstruction, raw article-number check, and context antecedent check; parser extraction was not called",
        "corpus_statute_count": len(id_to_key),
        "checked_high_confidence_resolutions": checked_high,
        "checked_non_high_confidence_resolutions": checked_non_high,
        "method_counts": dict(method_counts),
        "high_confidence_conflict_count": len(conflicts),
        "high_confidence_conflicts": conflicts,
        "status": "PASS" if not conflicts else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/corpus.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_records(read_jsonl(args.input), args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["status", "checked_high_confidence_resolutions", "high_confidence_conflict_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
