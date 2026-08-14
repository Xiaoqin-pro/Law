"""Citation parser v2 with conservative context repair and provenance.

Parser v1 remains in :mod:`citelaw.citation` and is intentionally untouched.
This module adds a separate parser for frozen-answer re-analysis.  It only
resolves a bare article number when the local context contains one unique law
antecedent; otherwise it keeps the citation unresolved.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .citation import ARTICLE_RE, CitationIndex, normalize_article_number


PARSER_VERSION = "citation_parser_v2"
RESOLUTION_METHODS = {
    "exact",
    "alias",
    "context_propagation",
    "boundary_repair",
    "long_title_normalization",
    "malformed_citation_repair",
    "unresolved",
}

_BRACKET_FOLD = str.maketrans(
    {
        "〈": "《",
        "〉": "》",
        "「": "《",
        "」": "》",
        "『": "《",
        "』": "》",
        "﹁": "《",
        "﹂": "》",
        "(": "（",
        ")": "）",
        "〔": "（",
        "〕": "）",
    }
)
_OPENING_BRACKETS = "《〈「『﹁（([{"
_CLOSING_BRACKETS = "》〉」』﹂）》)]}"
_BOUNDARY_GAP = set(" \t\r\n》〉」』﹂）)]}，,、。；;：:！!？?\"")
_SENTENCE_BREAKS = set("。！？!?；;\n")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def _fold_brackets(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).translate(_BRACKET_FOLD)


def _clean_name(value: str) -> str:
    value = _fold_brackets(value).strip()
    value = re.sub(r"\s+", "", value)
    value = value.strip("《》()（）[]{}〈〉「」『』〔〕")
    return value


def _fold_name(value: str) -> str:
    """Fold wrapper variants for matching, while retaining name characters."""

    return _clean_name(value)


@dataclass(frozen=True)
class LawMention:
    start: int
    end: int
    alias: str
    canonical_laws: Tuple[str, ...]


def _all_law_mentions(source: str, index: CitationIndex) -> List[LawMention]:
    folded_source = _fold_brackets(source)
    candidates: List[LawMention] = []
    for alias in index.aliases:
        folded_alias = _fold_brackets(alias)
        if not folded_alias:
            continue
        cursor = 0
        while True:
            position = folded_source.find(folded_alias, cursor)
            if position < 0:
                break
            laws = tuple(sorted(index.alias_to_laws.get(alias, ())))
            if laws:
                candidates.append(
                    LawMention(
                        start=position,
                        end=position + len(folded_alias),
                        alias=alias,
                        canonical_laws=laws,
                    )
                )
            cursor = position + max(1, len(folded_alias))

    # Corpus aliases overlap heavily (e.g. a full law name contains its short
    # alias). Keep the longest mention for each overlapping region.
    selected: List[LawMention] = []
    for mention in sorted(candidates, key=lambda item: (item.start, -(item.end - item.start), item.alias)):
        overlaps = [item for item in selected if item.start < mention.end and mention.start < item.end]
        if not overlaps:
            selected.append(mention)
            continue
        if all((mention.end - mention.start) > (item.end - item.start) for item in overlaps):
            selected = [item for item in selected if item not in overlaps]
            selected.append(mention)
    return sorted(selected, key=lambda item: (item.start, item.end))


def _article_raw(match: re.Match[str]) -> str:
    article = match.group("article")
    suffix = match.group("suffix")
    return article + ("之" + suffix if suffix else "")


def _gap_is_boundary(gap: str) -> bool:
    return len(gap) <= 8 and all(char in _BOUNDARY_GAP for char in gap)


def _paragraph_start(source: str, position: int) -> int:
    matches = list(_PARAGRAPH_RE.finditer(source, 0, position))
    return matches[-1].end() if matches else 0


def _sentence_start(source: str, position: int) -> int:
    start = 0
    for index in range(position - 1, -1, -1):
        if source[index] in _SENTENCE_BREAKS:
            start = index + 1
            break
    return start


def _direct_mentions(
    source: str, article_start: int, mentions: Sequence[LawMention]
) -> List[LawMention]:
    direct = [
        mention
        for mention in mentions
        if mention.end <= article_start
        and article_start - mention.end <= 8
        and _gap_is_boundary(source[mention.end:article_start])
    ]
    return sorted(direct, key=lambda item: (item.end, item.end - item.start), reverse=True)


def _context_mentions(
    source: str, article_start: int, mentions: Sequence[LawMention]
) -> List[LawMention]:
    paragraph_start = _paragraph_start(source, article_start)
    sentence_start = _sentence_start(source, article_start)
    lower_bound = max(paragraph_start, article_start - 240)
    candidates = [
        mention
        for mention in mentions
        if lower_bound <= mention.start
        and mention.end <= article_start
        and mention.end > paragraph_start
    ]
    # Prefer the current sentence; if it has no explicit law, a unique recent
    # paragraph antecedent is still allowed by the audit protocol.
    current_sentence = [mention for mention in candidates if mention.start >= sentence_start]
    return current_sentence or candidates


def _unique_laws(mentions: Sequence[LawMention]) -> Tuple[str, ...]:
    return tuple(sorted({law for mention in mentions for law in mention.canonical_laws}))


def _ids_for_laws(index: CitationIndex, laws: Iterable[str], article: str) -> Tuple[int, ...]:
    ids: Set[int] = set()
    for law in laws:
        ids.update(index.law_article_to_ids.get((law, article), ()))
    return tuple(sorted(ids))


def _mention_law_name(mention: LawMention, source: str) -> str:
    return source[mention.start:mention.end]


def _repair_kind(
    source: str,
    mention: LawMention,
    citation_start: int,
    article_start: int,
    canonical_law: str,
) -> Tuple[bool, Optional[str]]:
    law_raw = _mention_law_name(mention, source)
    citation_raw = source[citation_start:mention.end]
    gap = source[mention.end:article_start]
    folded_raw = _fold_name(law_raw)
    folded_canonical = _fold_name(canonical_law)
    has_variant_bracket = any(
        char in law_raw for char in "〈〉「」『』﹁﹂()〔〕"
    ) and _fold_brackets(law_raw) != law_raw
    has_nested_title = any(char in law_raw for char in "〈〉()（）") and any(
        char in canonical_law for char in "《》()（）"
    )
    has_opening_wrapper = bool(citation_raw) and citation_raw[0] in _OPENING_BRACKETS
    missing_or_extra_wrapper = (
        (has_opening_wrapper and "》" not in gap)
        or (not has_opening_wrapper and "》" in gap)
        or citation_raw.endswith(_CLOSING_BRACKETS)
    )
    if has_nested_title and (has_variant_bracket or folded_raw != folded_canonical):
        return True, "long_title_normalization"
    if missing_or_extra_wrapper:
        return True, "malformed_citation_repair" if not gap else "boundary_repair"
    if folded_raw != folded_canonical:
        return True, "boundary_repair"
    return False, None


def _normalized_citation(law: str, article_raw: str) -> str:
    return f"《{law}》第{article_raw}条"


def _record(
    *,
    raw_citation: str,
    normalized_citation: str,
    law_name_raw: Optional[str],
    article_raw: str,
    article_number: Optional[str],
    normalized_law_name: Optional[str],
    canonical_statute_id: Optional[int],
    parse_status: str,
    resolution_method: str,
    candidate_statute_ids: Sequence[int],
    span: Tuple[int, int],
    repair_applied: bool,
    repair_type: Optional[str],
    context_source_text: Optional[str],
    confidence: str,
) -> Dict[str, Any]:
    return {
        "raw_citation": raw_citation,
        "normalized_citation": normalized_citation,
        "law_name_raw": law_name_raw,
        "article_raw": article_raw,
        "article_number": article_number,
        "normalized_law_name": normalized_law_name,
        "canonical_statute_id": canonical_statute_id,
        "matched_statute_id": canonical_statute_id,
        "parse_status": parse_status,
        "resolution_method": resolution_method,
        "candidate_statute_ids": list(candidate_statute_ids),
        "parser_version": PARSER_VERSION,
        "repair_applied": repair_applied,
        "repair_type": repair_type,
        "context_source_text": context_source_text,
        "confidence": confidence,
        "span": [span[0], span[1]],
    }


def _resolve_named(
    source: str,
    match: re.Match[str],
    mention: LawMention,
    index: CitationIndex,
) -> Dict[str, Any]:
    article_raw = _article_raw(match)
    article_number = normalize_article_number(article_raw)
    laws = mention.canonical_laws
    candidates = _ids_for_laws(index, laws, article_number or "")
    canonical_law = laws[0] if len(laws) == 1 else None
    citation_start = mention.start - 1 if mention.start and source[mention.start - 1] in _OPENING_BRACKETS else mention.start
    repair_applied, repair_type = (
        _repair_kind(source, mention, citation_start, match.start(), canonical_law)
        if canonical_law
        else (False, None)
    )
    if article_number is None or len(laws) != 1 or len(candidates) != 1:
        status = "malformed" if article_number is None else "ambiguous" if len(candidates) > 1 or len(laws) > 1 else "unresolved"
        return _record(
            raw_citation=source[citation_start:match.end()],
            normalized_citation=source[citation_start:match.end()],
            law_name_raw=_mention_law_name(mention, source),
            article_raw=article_raw,
            article_number=article_number,
            normalized_law_name=canonical_law,
            canonical_statute_id=None,
            parse_status=status,
            resolution_method="unresolved",
            candidate_statute_ids=candidates,
            span=(mention.start, match.end),
            repair_applied=False,
            repair_type=None,
            context_source_text=None,
            confidence="low",
        )
    raw_citation = source[citation_start:match.end()]
    method = "exact" if mention.alias == canonical_law else "alias"
    if repair_type == "long_title_normalization":
        method = "long_title_normalization"
    elif repair_type == "boundary_repair":
        method = "boundary_repair"
    elif repair_type == "malformed_citation_repair":
        method = "malformed_citation_repair"
    return _record(
        raw_citation=raw_citation,
        normalized_citation=_normalized_citation(canonical_law, article_raw),
        law_name_raw=source[citation_start:mention.end],
        article_raw=article_raw,
        article_number=article_number,
        normalized_law_name=canonical_law,
        canonical_statute_id=candidates[0],
        parse_status="resolved_unique",
        resolution_method=method,
        candidate_statute_ids=candidates,
        span=(citation_start, match.end()),
        repair_applied=repair_applied,
        repair_type=repair_type,
        context_source_text=None,
        confidence="high",
    )


def _resolve_article_only(
    source: str,
    match: re.Match[str],
    mentions: Sequence[LawMention],
    index: CitationIndex,
) -> Dict[str, Any]:
    article_raw = _article_raw(match)
    article_number = normalize_article_number(article_raw)
    raw = source[match.start():match.end()]
    context = _context_mentions(source, match.start(), mentions)
    laws = _unique_laws(context)
    candidates = _ids_for_laws(index, laws, article_number or "") if len(laws) == 1 else ()
    if article_number is None or len(laws) != 1 or len(candidates) != 1:
        return _record(
            raw_citation=raw,
            normalized_citation=raw,
            law_name_raw=None,
            article_raw=article_raw,
            article_number=article_number,
            normalized_law_name=None,
            canonical_statute_id=None,
            parse_status="malformed" if article_number is None else "ambiguous" if len(laws) > 1 else "unresolved",
            resolution_method="unresolved",
            candidate_statute_ids=candidates,
            span=(match.start(), match.end()),
            repair_applied=False,
            repair_type=None,
            context_source_text=None,
            confidence="low",
        )
    law = laws[0]
    source_text = source[context[-1].start:match.start()] if context else None
    return _record(
        raw_citation=raw,
        normalized_citation=_normalized_citation(law, article_raw),
        law_name_raw=None,
        article_raw=article_raw,
        article_number=article_number,
        normalized_law_name=law,
        canonical_statute_id=candidates[0],
        parse_status="resolved_unique",
        resolution_method="context_propagation",
        candidate_statute_ids=candidates,
        span=(match.start(), match.end()),
        repair_applied=True,
        repair_type="context_propagation",
        context_source_text=source_text,
        confidence="high",
    )


def extract_citations_v2(text: str, index: CitationIndex) -> List[Dict[str, Any]]:
    """Extract citations with conservative local context recovery."""

    source = unicodedata.normalize("NFKC", str(text or ""))
    mentions = _all_law_mentions(source, index)
    records: List[Dict[str, Any]] = []
    for match in ARTICLE_RE.finditer(source):
        direct = _direct_mentions(source, match.start(), mentions)
        if direct:
            record = _resolve_named(source, match, direct[0], index)
        else:
            record = _resolve_article_only(source, match, mentions, index)
        records.append(record)
    records.sort(key=lambda record: (record["span"][0], record["span"][1]))
    for record in records:
        record.pop("span", None)
    return records


def build_citation_index_v2(rows: Iterable[Mapping[str, Any]]) -> CitationIndex:
    """Build the same immutable corpus index used by v1."""

    from .citation import build_citation_index

    return build_citation_index(rows)
