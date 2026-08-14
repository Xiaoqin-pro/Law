"""Deterministic legal-citation extraction and corpus resolution.

This module deliberately does not make a semantic legal-support judgment.  It
only extracts explicit law/article mentions, normalizes them, and resolves
unique corpus keys to statute IDs.  Unresolved and ambiguous mentions remain
visible in the output so downstream analysis can audit them.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


_ARTICLE_NUMBER = r"[0-9０-９一二三四五六七八九十百千万亿零〇两]+"
ARTICLE_RE = re.compile(
    rf"第\s*(?P<article>{_ARTICLE_NUMBER})\s*条"
    rf"(?:\s*之\s*(?P<suffix>{_ARTICLE_NUMBER}))?"
)
STATUTE_NAME_RE = re.compile(
    rf"^(?P<law>.+?)第(?P<article>{_ARTICLE_NUMBER})条"
    rf"(?:之(?P<suffix>{_ARTICLE_NUMBER}))?$"
)
STATUTE_ID_MARKER_RE = re.compile(r"\[法条ID\s+(?P<statute_id>\d+)\]")

_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}


def normalize_law_name(value: str) -> str:
    """Normalize a corpus law name or a cited law alias."""

    value = unicodedata.normalize("NFKC", str(value)).strip()
    value = re.sub(r"\s+", "", value)
    if value.startswith("《") and value.endswith("》"):
        value = value[1:-1]
    return value


def _chinese_number(value: str) -> Optional[int]:
    value = unicodedata.normalize("NFKC", value).replace(" ", "")
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if any(char not in _DIGITS and char not in _UNITS for char in value):
        return None
    # A sequence without a unit is a digit string, e.g. 二〇二四.
    if not any(char in _UNITS for char in value):
        try:
            return int("".join(str(_DIGITS[char]) for char in value))
        except ValueError:
            return None
    total = 0
    section = 0
    number = 0
    for char in value:
        if char in _DIGITS:
            number = _DIGITS[char]
            continue
        unit = _UNITS[char]
        if unit in (10000, 100000000):
            section += (number or 1) * unit
            total += section
            section = 0
        else:
            section += (number or 1) * unit
        number = 0
    return total + section + number


def normalize_article_number(value: str) -> Optional[str]:
    """Return an article key such as ``577`` or ``120之2``."""

    value = unicodedata.normalize("NFKC", str(value)).replace(" ", "")
    value = value.removeprefix("第").removesuffix("条")
    parts = re.split(r"之", value, maxsplit=1)
    main = _chinese_number(parts[0])
    if main is None:
        return None
    if len(parts) == 1:
        return str(main)
    suffix = _chinese_number(parts[1])
    if suffix is None:
        return None
    return f"{main}之{suffix}"


def split_statute_name(statute_name: str) -> Optional[Tuple[str, str]]:
    """Split a corpus name into its law name and normalized article key."""

    value = unicodedata.normalize("NFKC", str(statute_name)).strip()
    match = STATUTE_NAME_RE.match(value)
    if not match:
        return None
    article_raw = match.group("article")
    if match.group("suffix"):
        article_raw += "之" + match.group("suffix")
    article = normalize_article_number(article_raw)
    if article is None:
        return None
    return normalize_law_name(match.group("law")), article


@dataclass(frozen=True)
class CitationIndex:
    """Canonical lookup tables built from ``corpus.jsonl``."""

    statute_by_id: Mapping[int, Mapping[str, Any]]
    canonical_laws: Tuple[str, ...]
    law_article_to_ids: Mapping[Tuple[str, str], Tuple[int, ...]]
    alias_to_laws: Mapping[str, Tuple[str, ...]]
    alias_article_to_ids: Mapping[Tuple[str, str], Tuple[int, ...]]
    malformed_statute_names: Tuple[str, ...]

    @classmethod
    def from_corpus(cls, rows: Iterable[Mapping[str, Any]]) -> "CitationIndex":
        statute_by_id: Dict[int, Mapping[str, Any]] = {}
        law_article: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        law_to_ids: Dict[str, Set[int]] = defaultdict(set)
        malformed: List[str] = []
        for row in rows:
            statute_id = int(row["statute_id"])
            statute_by_id[statute_id] = row
            split = split_statute_name(str(row["statute_name"]))
            if split is None:
                malformed.append(str(row["statute_name"]))
                continue
            law, article = split
            law_article[(law, article)].add(statute_id)
            law_to_ids[law].add(statute_id)

        # Full names are always valid aliases.  A legal-name suffix such as
        # "民法典" is allowed only when it identifies one corpus law.
        alias_to_laws: Dict[str, Set[str]] = defaultdict(set)
        for law in law_to_ids:
            alias_to_laws[law].add(law)
            if law.startswith("中华人民共和国"):
                alias_to_laws[law[len("中华人民共和国") :]].add(law)

        alias_article: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        for (law, article), ids in law_article.items():
            for alias, laws in alias_to_laws.items():
                if law in laws:
                    alias_article[(alias, article)].update(ids)

        return cls(
            statute_by_id=statute_by_id,
            canonical_laws=tuple(sorted(law_to_ids)),
            law_article_to_ids={key: tuple(sorted(ids)) for key, ids in law_article.items()},
            alias_to_laws={key: tuple(sorted(values)) for key, values in alias_to_laws.items()},
            alias_article_to_ids={key: tuple(sorted(ids)) for key, ids in alias_article.items()},
            malformed_statute_names=tuple(sorted(malformed)),
        )

    @property
    def aliases(self) -> Tuple[str, ...]:
        return tuple(sorted(self.alias_to_laws, key=lambda value: (-len(value), value)))

    def audit(self) -> Dict[str, Any]:
        ambiguous_keys = [
            {"law_name": law, "article_number": article, "candidate_statute_ids": list(ids)}
            for (law, article), ids in sorted(self.law_article_to_ids.items())
            if len(ids) > 1
        ]
        return {
            "statute_count": len(self.statute_by_id),
            "structured_count": len(self.law_article_to_ids),
            "unstructured_count": len(self.malformed_statute_names),
            "duplicate_canonical_key_count": sum(
                1 for ids in self.law_article_to_ids.values() if len(ids) > 1
            ),
            "ambiguous_key_examples": ambiguous_keys[:20],
            "law_count": len(self.canonical_laws),
            "alias_count": len(self.alias_to_laws),
        }


def _citation_record(
    *,
    raw_citation: str,
    law_name_raw: Optional[str],
    article_raw: Optional[str],
    article_number: Optional[str],
    normalized_law_name: Optional[str],
    canonical_statute_id: Optional[int],
    parse_status: str,
    resolution_method: str,
    candidate_statute_ids: Sequence[int],
    span: Tuple[int, int],
) -> Dict[str, Any]:
    return {
        "raw_citation": raw_citation,
        "law_name_raw": law_name_raw,
        "article_raw": article_raw,
        "article_number": article_number,
        "normalized_law_name": normalized_law_name,
        "canonical_statute_id": canonical_statute_id,
        "matched_statute_id": canonical_statute_id,
        "parse_status": parse_status,
        "resolution_method": resolution_method,
        "candidate_statute_ids": list(candidate_statute_ids),
        "span": [span[0], span[1]],
    }


def _resolve_named_citation(
    *,
    raw_citation: str,
    law_name_raw: str,
    article_raw: str,
    span: Tuple[int, int],
    index: CitationIndex,
) -> Dict[str, Any]:
    law = normalize_law_name(law_name_raw)
    article = normalize_article_number(article_raw)
    laws = index.alias_to_laws.get(law, ())
    candidates = index.alias_article_to_ids.get((law, article or ""), ())
    if article is None:
        return _citation_record(
            raw_citation=raw_citation,
            law_name_raw=law_name_raw,
            article_raw=article_raw,
            article_number=None,
            normalized_law_name=law or None,
            canonical_statute_id=None,
            parse_status="malformed",
            resolution_method="invalid_article_number",
            candidate_statute_ids=(),
            span=span,
        )
    if not laws:
        status = "unresolved"
        method = "law_name_not_in_corpus"
    elif len(laws) > 1 or len(candidates) > 1:
        status = "ambiguous"
        method = "non_unique_corpus_mapping"
    elif len(candidates) == 1:
        status = "resolved_unique"
        method = "canonical_law_article_key"
    else:
        status = "unresolved"
        method = "law_article_not_in_corpus"
    return _citation_record(
        raw_citation=raw_citation,
        law_name_raw=law_name_raw,
        article_raw=article_raw,
        article_number=article,
        normalized_law_name=law or None,
        canonical_statute_id=candidates[0] if status == "resolved_unique" else None,
        parse_status=status,
        resolution_method=method,
        candidate_statute_ids=candidates,
        span=span,
    )


def _resolve_article_only(
    *, raw_citation: str, article_raw: str, span: Tuple[int, int], index: CitationIndex
) -> Dict[str, Any]:
    article = normalize_article_number(article_raw)
    return _citation_record(
        raw_citation=raw_citation,
        law_name_raw=None,
        article_raw=article_raw,
        article_number=article,
        normalized_law_name=None,
        canonical_statute_id=None,
        parse_status="unresolved" if article is not None else "malformed",
        resolution_method="law_name_missing" if article is not None else "invalid_article_number",
        candidate_statute_ids=(),
        span=span,
    )


def extract_citations(text: str, index: CitationIndex) -> List[Dict[str, Any]]:
    """Extract explicit citations in source order.

    Named law/article citations are matched against corpus law aliases.  A
    standalone ``第...条`` is retained as unresolved rather than inferred from
    the question, reference answer, or nearby evidence.
    """

    source = unicodedata.normalize("NFKC", str(text or ""))
    named: List[Dict[str, Any]] = []
    for alias in index.aliases:
        cursor = 0
        while True:
            position = source.find(alias, cursor)
            if position < 0:
                break
            law_start = position
            article_cursor = position + len(alias)
            if law_start > 0 and source[law_start - 1] == "《":
                law_start -= 1
            if article_cursor < len(source) and source[article_cursor] == "》":
                article_cursor += 1
            tail = source[article_cursor : article_cursor + 80]
            article_match = ARTICLE_RE.search(tail)
            if article_match is not None and article_match.start() <= 20:
                article_start = article_cursor + article_match.start()
                article_end = article_cursor + article_match.end()
                span = (law_start, article_end)
                raw = source[law_start:article_end]
                raw_law = source[law_start:article_start].strip()
                named.append(
                    {
                        "article_span": (article_start, article_end),
                        "law_len": len(alias),
                        "record": _resolve_named_citation(
                            raw_citation=raw,
                            law_name_raw=raw_law,
                            article_raw=(
                                article_match.group("article")
                                + (
                                    "之" + article_match.group("suffix")
                                    if article_match.group("suffix")
                                    else ""
                                )
                            ),
                            span=span,
                            index=index,
                        ),
                    }
                )
            cursor = position + max(1, len(alias))

    # Multiple aliases can match the same citation (e.g. full name and its
    # suffix alias).  Keep the longest alias for each article span.
    selected: List[Dict[str, Any]] = []
    by_article_span: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for item in named:
        key = item["article_span"]
        current = by_article_span.get(key)
        if current is None or item["law_len"] > current["law_len"]:
            by_article_span[key] = item
    selected.extend(by_article_span.values())
    selected.sort(key=lambda item: item["record"]["span"][0])

    records: List[Dict[str, Any]] = [item["record"] for item in selected]
    occupied = [tuple(record["span"]) for record in records]

    # Preserve a bracketed law name even when it is not present in the
    # corpus.  This is still unresolved; the parser must not silently reduce
    # it to an anonymous article-only citation.
    unknown_bracket = re.compile(
        rf"《(?P<law>[^》]+)》\s*第\s*(?P<article>{_ARTICLE_NUMBER})\s*条"
        rf"(?:\s*之\s*(?P<suffix>{_ARTICLE_NUMBER}))?"
    )
    for match in unknown_bracket.finditer(source):
        span = (match.start(), match.end())
        if any(start <= span[0] < end or start < span[1] <= end for start, end in occupied):
            continue
        article_raw = match.group("article")
        if match.group("suffix"):
            article_raw += "之" + match.group("suffix")
        records.append(
            _resolve_named_citation(
                raw_citation=source[match.start() : match.end()],
                law_name_raw="《" + match.group("law") + "》",
                article_raw=article_raw,
                span=span,
                index=index,
            )
        )
        occupied.append(span)

    for match in ARTICLE_RE.finditer(source):
        span = (match.start(), match.end())
        if any(start <= span[0] < end or start < span[1] <= end for start, end in occupied):
            continue
        records.append(
            _resolve_article_only(
                raw_citation=source[match.start() : match.end()],
                article_raw=(
                    match.group("article")
                    + ("之" + match.group("suffix") if match.group("suffix") else "")
                ),
                span=span,
                index=index,
            )
        )

    records.sort(key=lambda record: (record["span"][0], record["span"][1]))
    for record in records:
        record.pop("span", None)
    return records


def extract_statute_id_markers(text: str) -> List[int]:
    """Return internal ``[法条ID n]`` markers separately from legal citations.

    These markers identify prompt evidence blocks but are not conventional
    law/article citations.  Phase 3.5A records them as an audit field without
    counting them in citation-presence or citation-quality metrics.
    """

    source = unicodedata.normalize("NFKC", str(text or ""))
    return [int(match.group("statute_id")) for match in STATUTE_ID_MARKER_RE.finditer(source)]


def build_citation_index(rows: Iterable[Mapping[str, Any]]) -> CitationIndex:
    return CitationIndex.from_corpus(rows)
