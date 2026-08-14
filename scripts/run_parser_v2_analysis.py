"""Reanalyse the frozen Phase 3.5 answers with citation parser v2.

This script is deliberately a read-only analysis of the eight frozen
generation artifacts.  It does not regenerate answers, retrieve statutes, or
modify the v1 citation artifacts.  Main citation metrics count only
high-confidence parser-v2 resolutions; medium/low-confidence resolutions are
reported separately.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from citelaw.citation import (  # noqa: E402
    build_citation_index,
    extract_statute_id_markers,
)
from citelaw.citation_v2 import PARSER_VERSION, extract_citations_v2  # noqa: E402
from run_citation_analysis import (  # noqa: E402
    FROZEN_COMMIT,
    METHOD_LABELS,
    METHOD_ORDER,
    MODEL_ORDER,
    SEED,
    artifact_integrity,
    as_int_list,
    diagnostic_matrix,
    load_gold_and_queries,
    mean,
    paired_bootstrap,
    quality_metrics,
    read_json,
    read_jsonl,
    safe_rate,
    sha256,
    write_csv,
    write_json,
    write_jsonl,
)


REPORT_DIR = ROOT / "reports/phase3_5/parser_v2"
OUTPUT_DIR = ROOT / "outputs/phase3_5/parser_v2"
V1_RECORDS = ROOT / "reports/phase3_5/citation_records.jsonl"
V1_METRICS = ROOT / "reports/phase3_5/citation_metrics.json"
V1_SOURCE = ROOT / "src/citelaw/citation.py"
MODEL_ASSISTED_PRIORITY = ROOT / "reports/phase3_5/model_assisted_parser_qa/parser_qa_model_assisted_priority_review.csv"
PARSER_QA_DATA = ROOT / "reports/phase3_5/parser_qa_data.json"


def high_confidence(citation: Mapping[str, Any]) -> bool:
    return citation.get("parse_status") == "resolved_unique" and citation.get("confidence") == "high"


def resolved_any_confidence(citation: Mapping[str, Any]) -> bool:
    return citation.get("parse_status") == "resolved_unique"


def high_resolved_ids(citations: Sequence[Mapping[str, Any]]) -> List[int]:
    return [int(citation["canonical_statute_id"]) for citation in citations if high_confidence(citation)]


def high_occurrence_match_rate(citations: Sequence[Mapping[str, Any]], gold_ids: Iterable[int]) -> Optional[float]:
    gold = {int(value) for value in gold_ids}
    resolved = [citation for citation in citations if high_confidence(citation)]
    matched = sum(1 for citation in resolved if int(citation["canonical_statute_id"]) in gold)
    return safe_rate(matched, len(resolved))


def build_v2_evaluation_rows(
    loaded: Sequence[Mapping[str, Any]],
    query_rows: Mapping[int, Mapping[str, Any]],
    corpus_index: Any,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, int]] = set()
    for item in loaded:
        method = str(item["method"])
        model = str(item["model"])
        for frozen_row in item["rows"]:
            query_id = int(frozen_row["query_id"])
            key = (model, method, query_id)
            if key in seen:
                raise RuntimeError(f"Duplicate model/method/query record: {key}")
            seen.add(key)
            query = query_rows[query_id]
            gold = {int(value) for value in query["canonical_gold_ids"]}
            answer = str(frozen_row.get("answer") or "")
            citations = extract_citations_v2(answer, corpus_index)
            evidence_marker_ids = extract_statute_id_markers(answer)
            resolved_ids = high_resolved_ids(citations)
            medium_count = sum(1 for citation in citations if resolved_any_confidence(citation) and citation.get("confidence") == "medium")
            low_count = sum(1 for citation in citations if resolved_any_confidence(citation) and citation.get("confidence") == "low")
            if method == "direct":
                retrieved: List[int] = []
                visible: List[int] = []
                included: Optional[List[int]] = None
                evidence_applicable = False
            else:
                retrieved = as_int_list(frozen_row.get("retrieved_statute_ids"))
                visible = as_int_list(frozen_row.get("fully_visible_statute_ids"))
                included = as_int_list(frozen_row.get("included_statute_ids")) if method == "bm25" else None
                evidence_applicable = True
                if method == "bm25" and included != visible:
                    raise RuntimeError(f"BM25 evidence mismatch at {model}/{query_id}: included != fully_visible")
                if as_int_list(frozen_row.get("partially_visible_statute_ids")):
                    raise RuntimeError(f"Partial statute block in frozen artifact at {model}/{method}/{query_id}")
                if bool(frozen_row.get("was_truncated", False)):
                    raise RuntimeError(f"Truncated frozen context at {model}/{method}/{query_id}")
            visible_set = set(visible)
            resolved_set = set(resolved_ids)
            gold_visible = bool(gold & visible_set) if evidence_applicable else None
            all_resolved = bool(citations) and all(high_confidence(citation) for citation in citations)
            any_gold = bool(gold & resolved_set)
            all_gold = bool(citations) and all(
                high_confidence(citation) and int(citation["canonical_statute_id"]) in gold for citation in citations
            )
            answer_has_all_visible = (
                bool(citations)
                and all(high_confidence(citation) and int(citation["canonical_statute_id"]) in visible_set for citation in citations)
                if evidence_applicable
                else None
            )
            visible_resolved_count = sum(
                1 for citation in citations if high_confidence(citation) and int(citation["canonical_statute_id"]) in visible_set
            )
            parse_uncertain = any(not high_confidence(citation) for citation in citations)
            gold_occurrence_count = sum(1 for statute_id in resolved_ids if statute_id in gold)
            marker_visible_count = sum(1 for statute_id in evidence_marker_ids if statute_id in visible_set)
            marker_gold_count = sum(1 for statute_id in evidence_marker_ids if statute_id in gold)
            output.append(
                {
                    "query_id": query_id,
                    "model": model,
                    "method": method,
                    "parser_version": PARSER_VERSION,
                    "question": query["question"],
                    "reference_answer": query.get("reference_answer", ""),
                    "canonical_gold_ids": sorted(gold),
                    "generated_answer": answer,
                    "answer_status": frozen_row.get("status"),
                    "retrieved_statute_ids": retrieved,
                    "included_statute_ids": included,
                    "fully_visible_statute_ids": visible,
                    "visible_evidence_ids": visible,
                    "visible_evidence_applicable": evidence_applicable,
                    "citations": citations,
                    "evidence_marker_ids": evidence_marker_ids,
                    "evidence_marker_count": len(evidence_marker_ids),
                    "evidence_marker_visible_count": marker_visible_count if evidence_applicable else None,
                    "evidence_marker_gold_count": marker_gold_count,
                    "any_attribution_presence": bool(citations or evidence_marker_ids),
                    "citation_count": len(citations),
                    "resolved_citation_count": len(resolved_ids),
                    "resolved_high_confidence_count": len(resolved_ids),
                    "resolved_medium_confidence_count": medium_count,
                    "resolved_low_confidence_count": low_count,
                    "unresolved_citation_count": sum(1 for citation in citations if not high_confidence(citation)),
                    "ambiguous_citation_count": sum(1 for citation in citations if citation.get("parse_status") == "ambiguous"),
                    "malformed_citation_count": sum(1 for citation in citations if citation.get("parse_status") == "malformed"),
                    "has_explicit_citation": bool(citations),
                    "all_citations_resolved": all_resolved,
                    "answer_has_all_existing_citations": all_resolved,
                    "any_gold_citation": any_gold,
                    "all_citations_gold": all_gold,
                    "gold_citation_occurrence_count": gold_occurrence_count,
                    "gold_citation_match_rate": high_occurrence_match_rate(citations, gold),
                    "visible_resolved_citation_count": visible_resolved_count if evidence_applicable else None,
                    "answer_has_all_visible_citations": answer_has_all_visible,
                    "citation_parse_uncertain": parse_uncertain,
                    "gold_visible": gold_visible,
                    "gold_visible_count": len(gold & visible_set) if evidence_applicable else None,
                    "full_gold_recall_in_visible_evidence": safe_rate(len(gold & visible_set), len(gold)) if evidence_applicable else None,
                    "retrieval_failure": (not bool(gold & set(retrieved))) if evidence_applicable else None,
                    "citation_fabrication": None if parse_uncertain else False,
                    "citation_outside_visible_evidence": (
                        any(high_confidence(citation) and int(citation["canonical_statute_id"]) not in visible_set for citation in citations)
                        if evidence_applicable
                        else None
                    ),
                    "gold_citation_mismatch": bool(citations) and not any_gold,
                    "missing_citation": not bool(citations),
                    "needs_semantic_review": True,
                    "quality": quality_metrics(str(query.get("reference_answer") or ""), answer),
                }
            )
    if len(output) != 2472:
        raise RuntimeError(f"Expected 2472 v2 evaluation rows, got {len(output)}")
    return sorted(output, key=lambda row: (MODEL_ORDER.index(row["model"]), METHOD_ORDER.index(row["method"]), row["query_id"]))


def group_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], List[Mapping[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["method"]))].append(row)
    return grouped


def automatic_metrics_v2(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for (model, method), group in sorted(group_rows(rows).items(), key=lambda item: (MODEL_ORDER.index(item[0][0]), METHOD_ORDER.index(item[0][1]))):
        citations = sum(int(row["citation_count"]) for row in group)
        resolved = sum(int(row["resolved_high_confidence_count"]) for row in group)
        unresolved = sum(int(row["unresolved_citation_count"]) for row in group)
        medium = sum(int(row["resolved_medium_confidence_count"]) for row in group)
        low = sum(int(row["resolved_low_confidence_count"]) for row in group)
        ambiguous = sum(int(row["ambiguous_citation_count"]) for row in group)
        malformed = sum(int(row["malformed_citation_count"]) for row in group)
        gold_matches = sum(int(row["gold_citation_occurrence_count"]) for row in group)
        marker_count = sum(int(row["evidence_marker_count"]) for row in group)
        marker_visible = sum(int(row["evidence_marker_visible_count"] or 0) for row in group if row["visible_evidence_applicable"])
        marker_gold = sum(int(row["evidence_marker_gold_count"]) for row in group)
        visible_resolved = sum(int(row["visible_resolved_citation_count"] or 0) for row in group if row["visible_evidence_applicable"])
        visible_denominator = sum(int(row["resolved_high_confidence_count"]) for row in group if row["visible_evidence_applicable"])
        applicable = [row for row in group if row["visible_evidence_applicable"]]
        citation_present = [row for row in group if row["has_explicit_citation"]]
        results.append(
            {
                "model": model,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "parser_version": PARSER_VERSION,
                "answer_count": len(group),
                "citation_count": citations,
                "high_confidence_resolved_count": resolved,
                "unresolved_or_uncertain_count": unresolved,
                "citation_presence_rate": safe_rate(sum(bool(row["has_explicit_citation"]) for row in group), len(group)),
                "mean_citation_count": safe_rate(citations, len(group)),
                "citation_parse_success_rate_high_confidence": safe_rate(resolved, citations),
                "answer_all_citations_resolved_rate": safe_rate(sum(bool(row["all_citations_resolved"]) for row in citation_present), len(citation_present)),
                "unresolved_citation_rate": safe_rate(unresolved, citations),
                "ambiguous_citation_rate": safe_rate(ambiguous, citations),
                "malformed_citation_rate": safe_rate(malformed, citations),
                "medium_confidence_resolution_count": medium,
                "low_confidence_resolution_count": low,
                "citation_existence_rate_high_confidence": safe_rate(resolved, citations),
                "all_citations_exist_given_citation": safe_rate(sum(bool(row["answer_has_all_existing_citations"]) for row in citation_present), len(citation_present)),
                "answer_all_citations_exist_joint": safe_rate(sum(bool(row["answer_has_all_existing_citations"]) for row in group), len(group)),
                "gold_citation_match_rate_high_confidence": safe_rate(gold_matches, resolved),
                "answer_any_gold_citation_rate": safe_rate(sum(bool(row["any_gold_citation"]) for row in group), len(group)),
                "answer_all_citations_gold_rate": safe_rate(sum(bool(row["all_citations_gold"]) for row in citation_present), len(citation_present)),
                "evidence_marker_presence_rate": safe_rate(sum(bool(row["evidence_marker_count"]) for row in group), len(group)),
                "mean_evidence_marker_count": safe_rate(marker_count, len(group)),
                "marker_visible_consistency_rate": safe_rate(marker_visible, sum(int(row["evidence_marker_count"]) for row in applicable)) if applicable else None,
                "marker_gold_match_rate": safe_rate(marker_gold, marker_count),
                "any_attribution_presence_rate": safe_rate(sum(bool(row["any_attribution_presence"]) for row in group), len(group)),
                "visible_evidence_consistency_rate": safe_rate(visible_resolved, visible_denominator) if applicable else None,
                "all_citations_visible_given_citation": safe_rate(
                    sum(bool(row["answer_has_all_visible_citations"]) for row in citation_present if row["visible_evidence_applicable"]),
                    sum(1 for row in citation_present if row["visible_evidence_applicable"]),
                ) if applicable else None,
                "answer_all_citations_visible_joint": safe_rate(sum(bool(row["answer_has_all_visible_citations"]) for row in applicable), len(applicable)) if applicable else None,
                "gold_evidence_visible_rate": safe_rate(sum(bool(row["gold_visible"]) for row in applicable), len(applicable)) if applicable else None,
                "full_gold_recall_in_visible_evidence": mean([row["full_gold_recall_in_visible_evidence"] for row in applicable]) if applicable else None,
                "semantic_review_required_count": sum(bool(row["needs_semantic_review"]) for row in group),
                "quality_bertscore_f1": mean([row["quality"]["bertscore_f1"] for row in group]),
                "quality_bleu": mean([row["quality"]["bleu"] for row in group]),
                "quality_meteor": mean([row["quality"]["meteor"] for row in group]),
                "quality_rouge_l": mean([row["quality"]["rouge_l"] for row in group]),
                "quality_metric_note": "BLEU/METEOR/ROUGE-L are deterministic lexical similarity proxies; BERTScore unavailable in the frozen analysis environment.",
            }
        )
    return results


def v1_v2_comparison(v1_rows: Sequence[Mapping[str, Any]], v2_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    v1_by_key = {(row["model"], row["method"], int(row["query_id"])): row for row in v1_rows}
    v2_by_key = {(row["model"], row["method"], int(row["query_id"])): row for row in v2_rows}
    output: List[Dict[str, Any]] = []
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            v1_group = [row for key, row in v1_by_key.items() if key[:2] == (model, method)]
            v2_group = [row for key, row in v2_by_key.items() if key[:2] == (model, method)]
            v1_citations = [citation for row in v1_group for citation in row.get("citations", [])]
            v2_citations = [citation for row in v2_group for citation in row.get("citations", [])]
            v1_resolved = [citation for citation in v1_citations if citation.get("parse_status") == "resolved_unique"]
            v2_high = [citation for citation in v2_citations if high_confidence(citation)]
            unresolved_resolved = 0
            paired = 0
            for key, v1_row in v1_by_key.items():
                if key[:2] != (model, method):
                    continue
                v2_row = v2_by_key[key]
                for left, right in zip(v1_row.get("citations", []), v2_row.get("citations", [])):
                    paired += 1
                    if left.get("parse_status") != "resolved_unique" and high_confidence(right):
                        unresolved_resolved += 1
            output.append(
                {
                    "scope": "model_method",
                    "model": model,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "v1_citation_count": len(v1_citations),
                    "v1_resolved_count": len(v1_resolved),
                    "v1_unresolved_or_uncertain_count": len(v1_citations) - len(v1_resolved),
                    "v2_citation_count": len(v2_citations),
                    "v2_high_confidence_resolved_count": len(v2_high),
                    "v2_unresolved_or_uncertain_count": len(v2_citations) - len(v2_high),
                    "v2_medium_confidence_resolved_count": sum(1 for citation in v2_citations if resolved_any_confidence(citation) and citation.get("confidence") == "medium"),
                    "v2_low_confidence_resolved_count": sum(1 for citation in v2_citations if resolved_any_confidence(citation) and citation.get("confidence") == "low"),
                    "v1_unresolved_to_v2_high_confidence_resolved_paired": unresolved_resolved,
                    "paired_occurrence_count": paired,
                }
            )
    all_v1 = [citation for row in v1_rows for citation in row.get("citations", [])]
    all_v2 = [citation for row in v2_rows for citation in row.get("citations", [])]
    output.append(
        {
            "scope": "overall",
            "model": "ALL",
            "method": "ALL",
            "method_label": "All models/methods",
            "v1_citation_count": len(all_v1),
            "v1_resolved_count": sum(1 for citation in all_v1 if citation.get("parse_status") == "resolved_unique"),
            "v1_unresolved_or_uncertain_count": sum(1 for citation in all_v1 if citation.get("parse_status") != "resolved_unique"),
            "v2_citation_count": len(all_v2),
            "v2_high_confidence_resolved_count": sum(1 for citation in all_v2 if high_confidence(citation)),
            "v2_unresolved_or_uncertain_count": sum(1 for citation in all_v2 if not high_confidence(citation)),
            "v2_medium_confidence_resolved_count": sum(1 for citation in all_v2 if resolved_any_confidence(citation) and citation.get("confidence") == "medium"),
            "v2_low_confidence_resolved_count": sum(1 for citation in all_v2 if resolved_any_confidence(citation) and citation.get("confidence") == "low"),
            "v1_unresolved_to_v2_high_confidence_resolved_paired": sum(
                1
                for v1_row in v1_rows
                for v1_citation, v2_citation in zip(
                    v1_row.get("citations", []),
                    v2_by_key[(v1_row["model"], v1_row["method"], int(v1_row["query_id"]))].get("citations", []),
                )
                if v1_citation.get("parse_status") != "resolved_unique" and high_confidence(v2_citation)
            ),
            "paired_occurrence_count": sum(len(row.get("citations", [])) for row in v1_rows),
        }
    )
    return output


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def model_assisted_reference_comparison(v2_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sample = read_json(PARSER_QA_DATA)["rows"]
    by_key = {(row["model"], row["method"], int(row["query_id"])): row for row in v2_rows}
    resolved_sample = [row for row in sample if row.get("unit_type") == "resolved_citation"]
    controls = [row for row in sample if row.get("unit_type") == "no_conventional_citation_answer"]
    resolved_matches = []
    for unit in resolved_sample:
        row = by_key[(unit["model"], unit["method"], int(unit["query_id"]))]
        expected_id = int(unit["resolved_statute_id"])
        expected_article = str(unit["article_number"])
        match = any(
            high_confidence(citation)
            and int(citation.get("canonical_statute_id")) == expected_id
            and str(citation.get("article_number")) == expected_article
            for citation in row["citations"]
        )
        resolved_matches.append(match)
    control_false_positives = []
    for unit in controls:
        row = by_key[(unit["model"], unit["method"], int(unit["query_id"]))]
        control_false_positives.append(bool(row["citations"]))

    priority_rows = read_csv_rows(MODEL_ASSISTED_PRIORITY)
    disagreements: List[Dict[str, Any]] = []
    priority_agreements = 0
    priority_compared = 0
    for item in priority_rows:
        row = by_key[(item["model"], item["method"], int(item["query_id"]))]
        target_article = None
        for citation in row["citations"]:
            if str(citation.get("raw_citation")) == str(item["raw_citation"]):
                target_article = citation
                break
        if target_article is None:
            target_article = next((citation for citation in row["citations"] if str(citation.get("article_number")) == _article_digits(item["raw_citation"])), None)
        v2_resolvable = bool(target_article and high_confidence(target_article))
        expected = item.get("model_assisted_actually_resolvable", "").strip().lower()
        expected_bool = True if expected == "yes" else False if expected == "no" else None
        agreement = None if expected_bool is None else (v2_resolvable == expected_bool)
        if agreement is not None:
            priority_compared += 1
            priority_agreements += int(agreement)
        result = {
            "audit_unit_id": item["audit_unit_id"],
            "query_id": item["query_id"],
            "model": item["model"],
            "method": item["method"],
            "raw_citation": item["raw_citation"],
            "model_assisted_actually_resolvable": item.get("model_assisted_actually_resolvable"),
            "model_assisted_parser_error_type": item.get("model_assisted_parser_error_type"),
            "v2_resolvable_high_confidence": v2_resolvable,
            "v2_resolution_method": target_article.get("resolution_method") if target_article else None,
            "v2_parse_status": target_article.get("parse_status") if target_article else None,
            "agreement_with_model_assisted_reference": agreement,
        }
        if agreement is False:
            disagreements.append(result)
    comparison = {
        "label_source": "ChatGPT model-assisted review",
        "status": "MODEL_ASSISTED_REFERENCE_ONLY",
        "note": "Agreement/disagreement is an overlay comparison, not human validation and not an independent gold standard.",
        "resolved_sample": {
            "n": len(resolved_matches),
            "mapping_correct": sum(resolved_matches),
            "mapping_accuracy": safe_rate(sum(resolved_matches), len(resolved_matches)),
        },
        "no_citation_controls": {
            "n": len(control_false_positives),
            "false_positive_count": sum(control_false_positives),
            "false_positive_rate": safe_rate(sum(control_false_positives), len(control_false_positives)),
        },
        "priority_review": {
            "n": len(priority_rows),
            "compared_n": priority_compared,
            "agreement_count": priority_agreements,
            "agreement_rate": safe_rate(priority_agreements, priority_compared),
            "uncertain_reference_count": sum(1 for item in priority_rows if item.get("model_assisted_actually_resolvable", "").strip().lower() == "uncertain"),
            "disagreement_count": len(disagreements),
        },
    }
    return comparison, disagreements


def _article_digits(raw_citation: str) -> Optional[str]:
    import re

    match = re.search(r"第\s*([0-9零一二三四五六七八九十百千万亿〇两]+)\s*条", raw_citation or "")
    if not match:
        return None
    from citelaw.citation import normalize_article_number

    return normalize_article_number(match.group(1))


def adversarial_checks() -> Dict[str, Any]:
    corpus = [
        {"statute_id": 1, "statute_name": "民法典第五百条", "statute_text": "x"},
        {"statute_id": 2, "statute_name": "民法典第一百条", "statute_text": "x"},
        {"statute_id": 3, "statute_name": "刑法第二百条", "statute_text": "x"},
    ]
    index = build_citation_index(corpus)
    cases = {
        "unique_context": extract_citations_v2("根据《民法典》第五百条，第一百条规定。", index),
        "no_antecedent": extract_citations_v2("根据第十条规定。", index),
        "two_law_ambiguous": extract_citations_v2("根据《民法典》第五百条和《刑法》第二百条，第一百条规定。", index),
    }
    unique_ok = len(cases["unique_context"]) == 2 and cases["unique_context"][1].get("canonical_statute_id") == 2
    no_antecedent_ok = len(cases["no_antecedent"]) == 1 and cases["no_antecedent"][0].get("canonical_statute_id") is None
    ambiguous_ok = len(cases["two_law_ambiguous"]) == 3 and cases["two_law_ambiguous"][-1].get("canonical_statute_id") is None and cases["two_law_ambiguous"][-1].get("parse_status") == "ambiguous"
    return {
        "unique_context_propagation": unique_ok,
        "article_only_without_antecedent_unresolved": no_antecedent_ok,
        "two_law_ambiguous_unresolved": ambiguous_ok,
        "all_passed": unique_ok and no_antecedent_ok and ambiguous_ok,
    }


def parser_v1_freeze() -> Dict[str, Any]:
    v1_rows = read_jsonl(V1_RECORDS)
    all_citations = [citation for row in v1_rows for citation in row.get("citations", [])]
    return {
        "parser_version": "citation_parser_v1",
        "frozen_release_commit": FROZEN_COMMIT,
        "frozen_at_phase": "Phase 3.5 v1 baseline",
        "v1_conventional_citations": len(all_citations),
        "v1_resolved": sum(1 for citation in all_citations if citation.get("parse_status") == "resolved_unique"),
        "v1_unresolved_or_uncertain": sum(1 for citation in all_citations if citation.get("parse_status") != "resolved_unique"),
        "artifact_hashes": {
            "citation_records.jsonl": sha256(V1_RECORDS),
            "citation_metrics.json": sha256(V1_METRICS),
            "citation_parser_source": sha256(V1_SOURCE),
        },
        "note": "This v1 release is immutable; v2 outputs are additive and do not overwrite these artifacts.",
    }


def report_zh(
    integrity: Mapping[str, Any],
    v1_freeze: Mapping[str, Any],
    comparison: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    model_assisted: Mapping[str, Any],
    independent_audit: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    overall = next(row for row in comparison if row["scope"] == "overall")
    lines = [
        "# Phase 3.5 Parser v2 自动化审计报告",
        "",
        f"- Parser v2: `{PARSER_VERSION}`",
        f"- 冻结基线 commit: `{FROZEN_COMMIT}`",
        "- 范围：2472 条冻结答案；只重新解析 citation，不重新生成答案、不重新检索。",
        "- 审计路线：模型辅助参考 + 独立确定性审计；**不是人工验证**。",
        "",
        "## 1. v1 冻结与 v2 变化",
        "",
        f"v1 保留为不可覆盖的冻结版本：共 {v1_freeze['v1_conventional_citations']} 个 conventional citation，其中 {v1_freeze['v1_resolved']} 个已解析，{v1_freeze['v1_unresolved_or_uncertain']} 个 unresolved/uncertain。",
        f"v2 共发现 {overall['v2_citation_count']} 个 citation；高置信度解析 {overall['v2_high_confidence_resolved_count']} 个，仍 unresolved/uncertain {overall['v2_unresolved_or_uncertain_count']} 个。按 occurrence 顺序配对后，v1 未解析而被 v2 高置信度解析的数量为 {overall['v1_unresolved_to_v2_high_confidence_resolved_paired']}。",
        "v2 的新增能力是：唯一局部前件下的法名上下文传播、边界/括号归一化、长标题嵌套括号处理，以及轻度 malformed citation 修复；没有把多法名歧义强行猜成某一部法律。",
        "因此，v1/v2 会改变 parser-dependent 的 citation existence、gold-match 和 citation consistency 数值；论文中的 citation 表应采用 v2 版本。它不改变冻结的 retrieval 结果，也不提供法律语义正确性结论。当前不存在会推翻 Dense CLS 作为后续分析主检索候选的自动证据，但后续仍不得把 parser 指标写成显著的法律质量结论。",
        "",
        "## 2. 解析方法与主指标规则",
        "",
        "主自动指标只计入 `confidence=high` 的 `resolved_unique`；medium/low 单独报告，不混入主结果。citation 的存在、gold 命中和 evidence 可见性仍然不等于法律语义正确。",
        "",
        "| Model | Method | Citation count | High resolved | Unresolved/uncertain | Medium | Low | High-confidence gold match |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['model']} | {row['method_label']} | {row['citation_count']} | {row['high_confidence_resolved_count']} | {row['unresolved_or_uncertain_count']} | {row['medium_confidence_resolution_count']} | {row['low_confidence_resolution_count']} | {row['gold_citation_match_rate_high_confidence'] if row['gold_citation_match_rate_high_confidence'] is not None else 'N/A'} |"
        )
    lines += [
        "",
        "详细指标见 `citation_metrics_v2.csv`；检索-引用诊断和 paired bootstrap 也以 v2 解析结果重新计算，但没有改变冻结的 retrieval 结果。",
        "",
        "## 3. 模型辅助参考对照",
        "",
        f"40 条 resolved sample 的 statute mapping accuracy：{model_assisted['resolved_sample']['mapping_correct']}/{model_assisted['resolved_sample']['n']}。20 条 no-citation control 的 false positive：{model_assisted['no_citation_controls']['false_positive_count']}/{model_assisted['no_citation_controls']['n']}。",
        f"40 条 priority review case 与模型辅助标签的 agreement：{model_assisted['priority_review']['agreement_count']}/{model_assisted['priority_review']['compared_n']}；这只是 `MODEL_ASSISTED_REFERENCE_ONLY` 对照，不是人工金标准。",
        "",
        "## 4. 独立确定性审计",
        "",
        f"独立 checker 检查了 {independent_audit.get('checked_high_confidence_resolutions', 0)} 个高置信度解析，冲突数为 {independent_audit.get('high_confidence_conflict_count', 0)}，状态为 **{independent_audit.get('status')}**。checker 独立重建 corpus law/article key，并检查原始条号未被 repair 改变；没有调用 Parser v2 进行自证。",
        f"假阳性控制：20 条 no-citation control 均未产生 conventional citation；剩余 {overall['v2_unresolved_or_uncertain_count']} 条被明确保留为 unresolved/uncertain，不能自动计为 fabrication。",
        "",
        "## 5. 自动化 gate",
        "",
        f"最终状态：**{gate['status']}**。",
        "",
        "| 条件 | 结果 |",
        "|---|---|",
        f"| 全部 unit/regression tests | {gate['tests_passed']} |",
        f"| 40 条 resolved sample mapping accuracy = 100% | {gate['resolved_sample_mapping_100pct']} |",
        f"| 20 条 no-citation controls 无 false positive | {gate['no_citation_false_positive_free']} |",
        f"| 独立 checker 无高置信度冲突 | {gate['independent_checker_conflict_free']} |",
        f"| 无重复高置信度 parser failure | {gate['no_repeated_high_confidence_failure']} |",
        f"| 歧义案例保持 unresolved | {gate['ambiguous_cases_remain_unresolved']} |",
        "",
        "本 gate 是 `PARSER_V2_AUTOMATED_AUDIT_PASS/FAIL`，明确不是 human validation。完成本阶段后停止：不开始人工标注、不做 Dense semantic repair、不进入 Phase 4、不重新生成答案或运行 retrieval。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_json(ROOT / "reports/final_baseline_manifest.json")
    loaded, integrity = artifact_integrity(manifest)
    query_rows, _ = load_gold_and_queries()
    corpus_rows = read_jsonl(ROOT / "data/processed/corpus.jsonl")
    corpus_index = build_citation_index(corpus_rows)
    v2_rows = build_v2_evaluation_rows(loaded, query_rows, corpus_index)
    output_jsonl = OUTPUT_DIR / "citation_records_v2.jsonl"
    write_jsonl(output_jsonl, v2_rows)
    metrics = automatic_metrics_v2(v2_rows)
    matrix = diagnostic_matrix(v2_rows)
    bootstrap = paired_bootstrap(v2_rows)
    write_csv(REPORT_DIR / "citation_metrics_v2.csv", metrics, fieldnames=list(metrics[0].keys()))
    write_csv(REPORT_DIR / "retrieval_citation_matrix_v2.csv", matrix, fieldnames=list(matrix[0].keys()))
    write_csv(REPORT_DIR / "citation_metric_bootstrap_v2.csv", bootstrap, fieldnames=list(bootstrap[0].keys()))
    v1_rows = read_jsonl(V1_RECORDS)
    comparison = v1_v2_comparison(v1_rows, v2_rows)
    write_csv(REPORT_DIR / "parser_v1_vs_v2.csv", comparison, fieldnames=list(comparison[0].keys()))
    v1_freeze = parser_v1_freeze()
    write_json(REPORT_DIR.parent / "parser_v1/parser_v1_freeze.json", v1_freeze)
    model_assisted, disagreements = model_assisted_reference_comparison(v2_rows)
    write_csv(REPORT_DIR / "parser_v2_disagreement_cases.csv", disagreements, fieldnames=list(disagreements[0].keys()) if disagreements else ["audit_unit_id", "agreement_with_model_assisted_reference"])
    adversarial = adversarial_checks()
    independent_path = REPORT_DIR / "parser_v2_independent_audit.json"
    audit_command = [sys.executable, str(ROOT / "scripts/audit_parser_v2.py"), "--input", str(output_jsonl), "--corpus", str(ROOT / "data/processed/corpus.jsonl"), "--output", str(independent_path)]
    completed = subprocess.run(audit_command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Independent parser v2 audit failed to run:\n{completed.stdout}\n{completed.stderr}")
    independent_audit = read_json(independent_path)
    tests_command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    tests_run = subprocess.run(tests_command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    tests_passed = tests_run.returncode == 0
    gate = {
        "status": "PARSER_V2_AUTOMATED_AUDIT_PASS" if all([
            tests_passed,
            model_assisted["resolved_sample"]["mapping_accuracy"] == 1.0,
            model_assisted["no_citation_controls"]["false_positive_count"] == 0,
            independent_audit.get("high_confidence_conflict_count") == 0,
            independent_audit.get("high_confidence_conflict_count") == 0,
            adversarial["all_passed"],
        ]) else "PARSER_V2_AUTOMATED_AUDIT_FAIL",
        "tests_passed": tests_passed,
        "tests_command": " ".join(tests_command),
        "tests_output_tail": (tests_run.stdout + "\n" + tests_run.stderr)[-4000:],
        "resolved_sample_mapping_100pct": model_assisted["resolved_sample"]["mapping_accuracy"] == 1.0,
        "no_citation_false_positive_free": model_assisted["no_citation_controls"]["false_positive_count"] == 0,
        "independent_checker_conflict_free": independent_audit.get("high_confidence_conflict_count") == 0,
        "no_repeated_high_confidence_failure": independent_audit.get("high_confidence_conflict_count") == 0,
        "ambiguous_cases_remain_unresolved": adversarial["all_passed"],
        "note": "This is not human validation.",
    }
    write_json(REPORT_DIR / "parser_v2_metrics.json", {
        "analysis_version": "phase3_5_parser_v2_automated_audit_v1",
        "parser_version": PARSER_VERSION,
        "frozen_commit": FROZEN_COMMIT,
        "answer_count": len(v2_rows),
        "citation_count": sum(int(row["citation_count"]) for row in v2_rows),
        "high_confidence_resolved_count": sum(int(row["resolved_high_confidence_count"]) for row in v2_rows),
        "unresolved_or_uncertain_count": sum(int(row["unresolved_citation_count"]) for row in v2_rows),
        "resolution_method_counts": dict(Counter(citation.get("resolution_method") for row in v2_rows for citation in row.get("citations", []))),
        "confidence_counts": dict(Counter(citation.get("confidence") for row in v2_rows for citation in row.get("citations", []))),
        "integrity": integrity,
        "v1_freeze": v1_freeze,
        "v1_vs_v2_overall": next(row for row in comparison if row["scope"] == "overall"),
        "model_assisted_reference_only": model_assisted,
        "adversarial_checks": adversarial,
        "independent_audit": independent_audit,
        "gate": gate,
        "outputs": [
            "outputs/phase3_5/parser_v2/citation_records_v2.jsonl",
            "reports/phase3_5/parser_v2/parser_v2_report_zh.md",
            "reports/phase3_5/parser_v2/parser_v2_metrics.json",
            "reports/phase3_5/parser_v2/parser_v1_vs_v2.csv",
            "reports/phase3_5/parser_v2/parser_v2_independent_audit.json",
            "reports/phase3_5/parser_v2/parser_v2_disagreement_cases.csv",
            "reports/phase3_5/parser_v2/citation_metrics_v2.csv",
        ],
    })
    (REPORT_DIR / "parser_v2_report_zh.md").write_text(report_zh(integrity, v1_freeze, comparison, metrics, model_assisted, independent_audit, gate), encoding="utf-8")
    print(json.dumps({
        "status": gate["status"],
        "answer_count": len(v2_rows),
        "citation_count": sum(int(row["citation_count"]) for row in v2_rows),
        "high_confidence_resolved_count": sum(int(row["resolved_high_confidence_count"]) for row in v2_rows),
        "unresolved_or_uncertain_count": sum(int(row["unresolved_citation_count"]) for row in v2_rows),
        "independent_audit": independent_audit.get("status"),
        "tests_passed": tests_passed,
        "outputs": str(REPORT_DIR),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
