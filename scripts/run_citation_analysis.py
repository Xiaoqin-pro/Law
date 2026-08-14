"""Run Phase 3.5A automatic citation analysis on frozen generation artifacts.

The script is read-only with respect to the eight frozen generation inputs:
it verifies their manifest hashes and writes only new Phase 3.5 artifacts.
It does not regenerate answers, retrieve statutes, or perform semantic legal
judgment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - the analysis environment supplies numpy
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from citelaw.citation import (  # noqa: E402
    build_citation_index,
    extract_citations,
    extract_statute_id_markers,
    split_statute_name,
)


FROZEN_COMMIT = "b3679ade8b233b71e5d3812dd44184aabee05f8f"
SEED = 42
BOOTSTRAP_SAMPLES = 10_000
EXPECTED_QUERIES = 309
EXPECTED_ARTIFACTS = 8
METHOD_ORDER = ["direct", "bm25", "dense", "hybrid"]
MODEL_ORDER = ["qwen25_7b", "qwen3_4b"]
METHOD_LABELS = {
    "direct": "Direct",
    "bm25": "BM25 encoding-fixed",
    "dense": "Dense CLS",
    "hybrid": "Hybrid CLS",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_int_list(value: Any) -> List[int]:
    return [int(item) for item in (value or [])]


def safe_rate(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator == 0 else float(numerator) / float(denominator)


def occurrence_gold_match_rate(
    citations: Sequence[Mapping[str, Any]], gold_ids: Iterable[int]
) -> Optional[float]:
    """Match resolved citation occurrences to gold statute IDs.

    This deliberately counts occurrences, not distinct statute IDs.  Thus a
    duplicated gold citation contributes two matched occurrences, while a
    mixed ``[gold, wrong]`` answer contributes one match out of two.
    """

    gold = {int(value) for value in gold_ids}
    resolved = [
        citation
        for citation in citations
        if citation.get("parse_status") == "resolved_unique"
    ]
    matched = sum(
        1
        for citation in resolved
        if int(citation["canonical_statute_id"]) in gold
    )
    return safe_rate(matched, len(resolved))


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else sum(clean) / len(clean)


def artifact_integrity(manifest: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    generation = manifest.get("generation", [])
    if len(generation) != EXPECTED_ARTIFACTS:
        raise RuntimeError(f"Frozen manifest has {len(generation)} generation artifacts, expected {EXPECTED_ARTIFACTS}.")
    loaded: List[Dict[str, Any]] = []
    checks: List[Dict[str, Any]] = []
    for entry in generation:
        path = ROOT / Path(entry["file"])
        metadata_path = ROOT / Path(entry["metadata_file"])
        if not path.exists() or not metadata_path.exists():
            raise RuntimeError(f"Missing frozen artifact or metadata: {path}")
        actual_hash = sha256(path)
        actual_metadata_hash = sha256(metadata_path)
        rows = read_jsonl(path)
        query_ids = [int(row["query_id"]) for row in rows]
        valid = {
            "file_exists": path.exists(),
            "metadata_exists": metadata_path.exists(),
            "artifact_hash_matches": actual_hash == entry["artifact_sha256"],
            "metadata_hash_matches": actual_metadata_hash == entry["metadata_sha256"],
            "record_count_matches": len(rows) == int(entry["record_count"]) == EXPECTED_QUERIES,
            "unique_query_ids": len(set(query_ids)) == EXPECTED_QUERIES,
            "all_status_ok": all(row.get("status") == "ok" for row in rows),
        }
        if not all(valid.values()):
            raise RuntimeError(f"Frozen artifact integrity failure for {entry['key']}: {valid}")
        method = str(rows[0].get("method", "")) if rows else ""
        model = str(rows[0].get("model_key", entry.get("model_key", ""))) if rows else str(entry.get("model_key", ""))
        if method not in METHOD_ORDER or model not in MODEL_ORDER:
            raise RuntimeError(f"Unexpected method/model in {path}: {model}/{method}")
        checks.append({
            "key": entry["key"],
            "file": entry["file"],
            "model": model,
            "method": method,
            "record_count": len(rows),
            "sha256": actual_hash,
            "manifest_sha256": entry["artifact_sha256"],
            "metadata_sha256": actual_metadata_hash,
            "status": "pass",
            **valid,
        })
        loaded.append({"entry": entry, "rows": rows, "model": model, "method": method})
    by_pair = {(item["model"], item["method"]) for item in loaded}
    expected_pairs = {(model, method) for model in MODEL_ORDER for method in METHOD_ORDER}
    if by_pair != expected_pairs:
        raise RuntimeError(f"Frozen model/method pairs differ from expected: {by_pair}")
    return loaded, {
        "frozen_commit": FROZEN_COMMIT,
        "manifest_generation_commit": manifest.get("git_commit_at_manifest_generation"),
        "artifact_count": len(checks),
        "expected_artifact_count": EXPECTED_ARTIFACTS,
        "answer_count": sum(item["record_count"] for item in checks),
        "checks": checks,
        "status": "pass",
    }


def tokenize(value: str) -> List[str]:
    """Stable character/word tokens for reference-similarity proxies."""

    return re.findall(r"[A-Za-z]+|\d+|[\u3400-\u9fff]", str(value or "").lower())


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1)))


def sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not hyp or not ref:
        return 0.0
    precisions: List[float] = []
    for n in range(1, max_n + 1):
        hyp_counts = ngrams(hyp, n)
        ref_counts = ngrams(ref, n)
        total = sum(hyp_counts.values())
        clipped = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        # Sentence-level smoothing keeps the metric defined for short Chinese
        # answers, while the report labels it as a lexical similarity proxy.
        precisions.append((clipped + 1.0) / (total + 1.0))
    geometric = math.exp(sum(math.log(max(value, 1e-12)) for value in precisions) / len(precisions))
    brevity_penalty = 1.0 if len(hyp) >= len(ref) else math.exp(1.0 - len(ref) / max(1, len(hyp)))
    return float(brevity_penalty * geometric)


def meteor_proxy(reference: str, hypothesis: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    remaining = Counter(ref)
    matches: List[int] = []
    for index, token in enumerate(hyp):
        if remaining[token] > 0:
            remaining[token] -= 1
            matches.append(index)
    match_count = len(matches)
    if match_count == 0:
        return 0.0
    precision = match_count / len(hyp)
    recall = match_count / len(ref)
    fmean = 10 * precision * recall / (recall + 9 * precision) if precision and recall else 0.0
    chunks = 1
    for left, right in zip(matches, matches[1:]):
        if right != left + 1:
            chunks += 1
    penalty = 0.5 * (chunks / match_count) ** 3
    return float(fmean * (1.0 - penalty))


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token_left in left:
        current = [0]
        for index, token_right in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if token_left == token_right else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, hypothesis: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    length = lcs_length(ref, hyp)
    precision = length / len(hyp)
    recall = length / len(ref)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def quality_metrics(reference: str, answer: str) -> Dict[str, Any]:
    return {
        "bertscore_f1": None,
        "bleu": sentence_bleu(reference, answer),
        "meteor": meteor_proxy(reference, answer),
        "rouge_l": rouge_l(reference, answer),
        "bertscore_status": "unavailable_bert_score_package_not_installed",
    }


def load_gold_and_queries() -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    query_rows = {int(row["query_id"]): row for row in read_jsonl(ROOT / "data/processed/test.jsonl")}
    reconciliation = read_json(ROOT / "reports/data/lecoqa_gold_reconciliation.json")
    gold_rows = {int(row["query_id"]): row for row in reconciliation["rows"]}
    if set(query_rows) != set(gold_rows) or len(query_rows) != EXPECTED_QUERIES:
        raise RuntimeError("Query and reconciliation IDs are not the expected 309-query test split.")
    for query_id, row in query_rows.items():
        canonical = set(int(value) for value in gold_rows[query_id]["canonical_gold_ids"])
        original = set(int(value) for value in row.get("gold_statute_ids", []))
        if original != canonical:
            raise RuntimeError(f"Gold set mismatch for query {query_id}; positional gold mapping is forbidden.")
        row["canonical_gold_ids"] = sorted(canonical)
    return query_rows, gold_rows


def build_evaluation_rows(
    loaded: Sequence[Mapping[str, Any]],
    query_rows: Mapping[int, Mapping[str, Any]],
    corpus_index: Any,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, int]] = set()
    for item in loaded:
        method = str(item["method"])
        model = str(item["model"])
        for row in item["rows"]:
            query_id = int(row["query_id"])
            key = (model, method, query_id)
            if key in seen:
                raise RuntimeError(f"Duplicate model/method/query record: {key}")
            seen.add(key)
            query = query_rows[query_id]
            gold = set(int(value) for value in query["canonical_gold_ids"])
            answer = str(row.get("answer") or "")
            citations = extract_citations(answer, corpus_index)
            evidence_marker_ids = extract_statute_id_markers(answer)
            resolved_ids = [int(citation["canonical_statute_id"]) for citation in citations if citation.get("parse_status") == "resolved_unique"]
            if method == "direct":
                retrieved: List[int] = []
                visible: List[int] = []
                included: Optional[List[int]] = None
                evidence_applicable = False
            else:
                retrieved = as_int_list(row.get("retrieved_statute_ids"))
                visible = as_int_list(row.get("fully_visible_statute_ids"))
                included = as_int_list(row.get("included_statute_ids")) if method == "bm25" else None
                evidence_applicable = True
                if method == "bm25" and included != visible:
                    raise RuntimeError(f"BM25 evidence mismatch at {model}/{query_id}: included != fully_visible")
                if as_int_list(row.get("partially_visible_statute_ids")):
                    raise RuntimeError(f"Partial statute block found in frozen artifact at {model}/{method}/{query_id}")
                if bool(row.get("was_truncated", False)):
                    raise RuntimeError(f"Truncated frozen context found at {model}/{method}/{query_id}")
            visible_set = set(visible)
            resolved_set = set(resolved_ids)
            gold_visible = bool(gold & visible_set) if evidence_applicable else None
            citation_gold_match = bool(gold & resolved_set)
            all_resolved = bool(citations) and all(citation.get("parse_status") == "resolved_unique" for citation in citations)
            answer_has_all_existing_citations = all_resolved
            any_gold = bool(gold & resolved_set)
            all_gold = bool(citations) and all(citation.get("parse_status") == "resolved_unique" and int(citation["canonical_statute_id"]) in gold for citation in citations)
            answer_has_all_visible_citations = bool(citations) and all(citation.get("parse_status") == "resolved_unique" and int(citation["canonical_statute_id"]) in visible_set for citation in citations) if evidence_applicable else None
            visible_resolved_count = sum(1 for citation in citations if citation.get("parse_status") == "resolved_unique" and int(citation["canonical_statute_id"]) in visible_set)
            parse_uncertain = any(citation.get("parse_status") != "resolved_unique" for citation in citations)
            gold_citation_occurrence_count = sum(
                1 for citation in citations
                if citation.get("parse_status") == "resolved_unique"
                and int(citation["canonical_statute_id"]) in gold
            )
            marker_visible_count = sum(1 for statute_id in evidence_marker_ids if statute_id in visible_set)
            marker_gold_count = sum(1 for statute_id in evidence_marker_ids if statute_id in gold)
            any_attribution = bool(citations or evidence_marker_ids)
            quality = quality_metrics(str(query.get("reference_answer") or ""), answer)
            output.append({
                "query_id": query_id,
                "model": model,
                "method": method,
                "question": query["question"],
                "reference_answer": query.get("reference_answer", ""),
                "canonical_gold_ids": sorted(gold),
                "generated_answer": answer,
                "answer_status": row.get("status"),
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
                "any_attribution_presence": any_attribution,
                "citation_count": len(citations),
                "resolved_citation_count": len(resolved_ids),
                "unresolved_citation_count": sum(1 for citation in citations if citation.get("parse_status") == "unresolved"),
                "ambiguous_citation_count": sum(1 for citation in citations if citation.get("parse_status") == "ambiguous"),
                "malformed_citation_count": sum(1 for citation in citations if citation.get("parse_status") == "malformed"),
                "has_explicit_citation": bool(citations),
                "all_citations_resolved": all_resolved,
                "answer_has_all_existing_citations": answer_has_all_existing_citations,
                "any_gold_citation": any_gold,
                "all_citations_gold": all_gold,
                "gold_citation_occurrence_count": gold_citation_occurrence_count,
                "gold_citation_match_rate": occurrence_gold_match_rate(citations, gold),
                "visible_resolved_citation_count": visible_resolved_count if evidence_applicable else None,
                "answer_has_all_visible_citations": answer_has_all_visible_citations,
                "citation_parse_uncertain": parse_uncertain,
                "gold_visible": gold_visible,
                "gold_visible_count": len(gold & visible_set) if evidence_applicable else None,
                "full_gold_recall_in_visible_evidence": safe_rate(len(gold & visible_set), len(gold)) if evidence_applicable else None,
                "retrieval_failure": (not bool(gold & set(retrieved))) if evidence_applicable else None,
                # An unresolved or ambiguous mention is not automatically a
                # fabrication: it may be a missing alias or a law outside this
                # corpus.  Keep the flag tri-state and route it to semantic
                # review instead of silently labeling it erroneous.
                "citation_fabrication": None if parse_uncertain else False,
                "citation_outside_visible_evidence": (any(citation.get("parse_status") == "resolved_unique" and int(citation["canonical_statute_id"]) not in visible_set for citation in citations) if evidence_applicable else None),
                "gold_citation_mismatch": bool(citations) and not any_gold,
                "missing_citation": not bool(citations),
                "needs_semantic_review": True,
                "quality": quality,
            })
    if len(output) != EXPECTED_QUERIES * EXPECTED_ARTIFACTS:
        raise RuntimeError(f"Expected {EXPECTED_QUERIES * EXPECTED_ARTIFACTS} evaluation rows, got {len(output)}")
    return sorted(output, key=lambda row: (MODEL_ORDER.index(row["model"]), METHOD_ORDER.index(row["method"]), row["query_id"]))


def group_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], List[Mapping[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["method"]))].append(row)
    return grouped


def automatic_metrics(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for (model, method), group in sorted(group_rows(rows).items(), key=lambda item: (MODEL_ORDER.index(item[0][0]), METHOD_ORDER.index(item[0][1]))):
        citations = sum(int(row["citation_count"]) for row in group)
        resolved = sum(int(row["resolved_citation_count"]) for row in group)
        unresolved = sum(int(row["unresolved_citation_count"]) for row in group)
        ambiguous = sum(int(row["ambiguous_citation_count"]) for row in group)
        gold_matches = sum(
            sum(
                1
                for citation in row["citations"]
                if citation.get("parse_status") == "resolved_unique"
                and int(citation["canonical_statute_id"]) in set(row["canonical_gold_ids"])
            )
            for row in group
        )
        marker_count = sum(int(row["evidence_marker_count"]) for row in group)
        marker_visible = sum(int(row["evidence_marker_visible_count"] or 0) for row in group if row["visible_evidence_applicable"])
        marker_gold = sum(int(row["evidence_marker_gold_count"]) for row in group)
        visible_resolved = sum(int(row["visible_resolved_citation_count"] or 0) for row in group if row["visible_evidence_applicable"])
        visible_resolved_denominator = sum(int(row["resolved_citation_count"]) for row in group if row["visible_evidence_applicable"])
        applicable = [row for row in group if row["visible_evidence_applicable"]]
        citation_present = [row for row in group if row["has_explicit_citation"]]
        metric: Dict[str, Any] = {
            "model": model,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "answer_count": len(group),
            "citation_presence_rate": safe_rate(sum(bool(row["has_explicit_citation"]) for row in group), len(group)),
            "mean_citation_count": safe_rate(citations, len(group)),
            "citation_parse_success_rate": safe_rate(resolved, citations),
            "answer_all_citations_resolved_rate": safe_rate(sum(bool(row["all_citations_resolved"]) for row in citation_present), len(citation_present)),
            "unresolved_citation_rate": safe_rate(unresolved, citations),
            "ambiguous_citation_rate": safe_rate(ambiguous, citations),
            "citation_existence_rate": safe_rate(resolved, citations),
            "all_citations_exist_given_citation": safe_rate(sum(bool(row["answer_has_all_existing_citations"]) for row in citation_present), len(citation_present)),
            "answer_all_citations_exist_joint": safe_rate(sum(bool(row["answer_has_all_existing_citations"]) for row in group), len(group)),
            "gold_citation_match_rate": safe_rate(gold_matches, resolved),
            "answer_any_gold_citation_rate": safe_rate(sum(bool(row["any_gold_citation"]) for row in group), len(group)),
            "answer_all_citations_gold_rate": safe_rate(sum(bool(row["all_citations_gold"]) for row in citation_present), len(citation_present)),
            "evidence_marker_presence_rate": safe_rate(sum(bool(row["evidence_marker_count"]) for row in group), len(group)),
            "mean_evidence_marker_count": safe_rate(marker_count, len(group)),
            "marker_visible_consistency_rate": safe_rate(marker_visible, sum(int(row["evidence_marker_count"]) for row in applicable)) if applicable else None,
            "marker_gold_match_rate": safe_rate(marker_gold, marker_count),
            "any_attribution_presence_rate": safe_rate(sum(bool(row["any_attribution_presence"]) for row in group), len(group)),
            "visible_evidence_consistency_rate": safe_rate(visible_resolved, visible_resolved_denominator) if applicable else None,
            "all_citations_visible_given_citation": safe_rate(sum(bool(row["answer_has_all_visible_citations"]) for row in citation_present if row["visible_evidence_applicable"]), sum(1 for row in citation_present if row["visible_evidence_applicable"])) if applicable else None,
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
        results.append(metric)
    return results


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: Set[str] = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def diagnostic_matrix(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for (model, method), group in sorted(group_rows(rows).items(), key=lambda item: (MODEL_ORDER.index(item[0][0]), METHOD_ORDER.index(item[0][1]))):
        if method == "direct":
            continue
        counts = Counter()
        for row in group:
            if row["gold_visible"] and row["any_gold_citation"]:
                category = "A"
            elif row["gold_visible"] and not row["any_gold_citation"]:
                category = "B"
            elif not row["gold_visible"] and row["any_gold_citation"]:
                category = "C"
            else:
                category = "D"
            counts[category] += 1
        total = len(group)
        output.append({
            "model": model,
            "retriever": method,
            "retriever_label": METHOD_LABELS[method],
            **{f"{category}_count": counts[category] for category in "ABCD"},
            **{f"{category}_rate": safe_rate(counts[category], total) for category in "ABCD"},
            "total": total,
            "interpretation": "A=gold evidence visible and gold citation present; B=visible but not explicitly cited; C=gold citation present without gold evidence visible (manual review required); D=neither.",
        })
    return output


def paired_bootstrap(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if np is None:
        raise RuntimeError("numpy is required for the specified 10,000-sample bootstrap.")
    by_pair: Dict[Tuple[str, str], Dict[int, Mapping[str, Any]]] = {}
    for (model, method), group in group_rows(rows).items():
        by_pair[(model, method)] = {int(row["query_id"]): row for row in group}
    comparisons = [("direct", "bm25"), ("direct", "dense"), ("direct", "hybrid"), ("bm25", "dense"), ("dense", "hybrid")]
    metric_getters = {
        "citation_presence": lambda row: float(bool(row["has_explicit_citation"])),
        "answer_all_citations_exist_joint": lambda row: float(bool(row["answer_has_all_existing_citations"])),
        "any_gold_citation": lambda row: float(bool(row["any_gold_citation"])),
        "bleu": lambda row: float(row["quality"]["bleu"]),
        "meteor": lambda row: float(row["quality"]["meteor"]),
        "rouge_l": lambda row: float(row["quality"]["rouge_l"]),
        "answer_all_citations_visible_joint": lambda row: float(bool(row["answer_has_all_visible_citations"])),
    }
    output: List[Dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for model in MODEL_ORDER:
        for left, right in comparisons:
            query_ids = sorted(set(by_pair[(model, left)]) & set(by_pair[(model, right)]))
            for metric_name, getter in metric_getters.items():
                if metric_name == "answer_all_citations_visible_joint" and (left == "direct" or right == "direct"):
                    continue
                left_values = np.array([getter(by_pair[(model, left)][query_id]) for query_id in query_ids], dtype=float)
                right_values = np.array([getter(by_pair[(model, right)][query_id]) for query_id in query_ids], dtype=float)
                differences = right_values - left_values
                indices = rng.integers(0, len(differences), size=(BOOTSTRAP_SAMPLES, len(differences)))
                bootstrap_means = differences[indices].mean(axis=1)
                output.append({
                    "model": model,
                    "left_method": left,
                    "right_method": right,
                    "left_label": METHOD_LABELS[left],
                    "right_label": METHOD_LABELS[right],
                    "metric": metric_name,
                    "n": len(query_ids),
                    "difference_right_minus_left": float(differences.mean()),
                    "ci_low": float(np.quantile(bootstrap_means, 0.025)),
                    "ci_high": float(np.quantile(bootstrap_means, 0.975)),
                    "bootstrap_samples": BOOTSTRAP_SAMPLES,
                    "seed": SEED,
                })
    return output


def corpus_names_and_texts(corpus_index: Any, statute_ids: Sequence[int]) -> Tuple[List[str], List[str]]:
    names: List[str] = []
    texts: List[str] = []
    for statute_id in statute_ids:
        statute = corpus_index.statute_by_id.get(int(statute_id))
        if statute is None:
            continue
        names.append(str(statute["statute_name"]))
        texts.append(str(statute["statute_text"]))
    return names, texts


def make_manual_selection(rows: Sequence[Mapping[str, Any]], corpus_index: Any) -> Dict[str, Any]:
    dense = [row for row in rows if row["method"] == "dense"]
    by_query: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for row in dense:
        by_query[int(row["query_id"])][str(row["model"])] = row
    strata: Dict[str, List[int]] = defaultdict(list)
    for query_id, model_rows in by_query.items():
        # Query-level strata use the union of both frozen Dense model outputs;
        # this keeps the sample unique by query_id and avoids selecting it by
        # hand based on one model's answer.
        gold_visible = any(bool(row["gold_visible"]) for row in model_rows.values())
        gold_cited = any(bool(row["any_gold_citation"]) for row in model_rows.values())
        if gold_visible and gold_cited:
            stratum = "S1_gold_visible_and_gold_cited"
        elif gold_visible and not gold_cited:
            stratum = "S2_gold_visible_but_not_gold_cited"
        elif not gold_visible and gold_cited:
            stratum = "S3_gold_not_visible_but_gold_cited"
        else:
            stratum = "S4_gold_not_visible_and_not_gold_cited"
        strata[stratum].append(query_id)
    rng = random.Random(SEED)
    ordered_strata = ["S1_gold_visible_and_gold_cited", "S2_gold_visible_but_not_gold_cited", "S3_gold_not_visible_but_gold_cited", "S4_gold_not_visible_and_not_gold_cited"]
    sample_counts = {stratum: min(15, len(strata.get(stratum, []))) for stratum in ordered_strata}
    remaining_count = 60 - sum(sample_counts.values())
    # Redistribute an under-populated stratum's quota to S4 first.  For this
    # diagnostic design, S4 is the intended negative-control comparison
    # stratum; with the frozen populations this yields S1=15, S2=15, S3=1,
    # S4=29.  The source stratum is retained for every query; there is no
    # synthetic FILL stratum and every sampling probability is recoverable as
    # n_h / N_h.
    redistribution_priority = [
        "S4_gold_not_visible_and_not_gold_cited",
        "S1_gold_visible_and_gold_cited",
        "S2_gold_visible_but_not_gold_cited",
        "S3_gold_not_visible_but_gold_cited",
    ]
    while remaining_count > 0:
        candidates = [
            stratum
            for stratum in redistribution_priority
            if sample_counts[stratum] < len(strata.get(stratum, []))
        ]
        if not candidates:
            raise RuntimeError("Cannot allocate the requested 60-query sample across Dense strata.")
        stratum = candidates[0]
        sample_counts[stratum] += 1
        remaining_count -= 1
    selected: List[Dict[str, Any]] = []
    for stratum in ordered_strata:
        population = len(strata.get(stratum, []))
        take = sample_counts[stratum]
        chosen = sorted(rng.sample(sorted(strata.get(stratum, [])), take)) if take else []
        reason = "target_quota" if take <= 15 else "redistributed_quota_from_underpopulated_stratum"
        if population < 15:
            reason = "all_available_and_redistributed_quota"
        for query_id in chosen:
            selected.append({
                "query_id": query_id,
                "stratum": stratum,
                "original_stratum": stratum,
                "sampling_stratum": stratum,
                "stratum_population_N": population,
                "stratum_sample_n": take,
                "selection_probability": safe_rate(take, population),
                "sampling_weight": safe_rate(population, take),
                "selection_reason": reason,
            })
    selected = sorted(selected, key=lambda item: item["query_id"])
    if len(selected) != 60 or len({item["query_id"] for item in selected}) != 60:
        raise RuntimeError(f"Manual sample has {len(selected)} unique queries, expected 60.")
    selected_by_id = {item["query_id"]: item for item in selected}
    by_key = {(int(row["query_id"]), str(row["model"]), str(row["method"])): row for row in rows}
    primary_answers: List[Dict[str, Any]] = []
    additional_context: List[Dict[str, Any]] = []
    for item in selected:
        query_id = item["query_id"]
        for model in MODEL_ORDER:
            for method in ["dense"]:
                row = by_key[(query_id, model, method)]
                names, texts = corpus_names_and_texts(corpus_index, row["canonical_gold_ids"])
                visible_names, visible_texts = corpus_names_and_texts(corpus_index, row["visible_evidence_ids"])
                primary_answers.append({
                    "query_id": query_id,
                    "stratum": item["stratum"],
                    "original_stratum": item["original_stratum"],
                    "sampling_stratum": item["sampling_stratum"],
                    "stratum_population_N": item["stratum_population_N"],
                    "stratum_sample_n": item["stratum_sample_n"],
                    "selection_probability": item["selection_probability"],
                    "sampling_weight": item["sampling_weight"],
                    "selection_reason": item["selection_reason"],
                    "model": model,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "gold_statute_ids": json.dumps(row["canonical_gold_ids"], ensure_ascii=False),
                    "gold_statute_names": "\n".join(names),
                    "gold_statute_texts": "\n\n".join(texts),
                    "retrieved_statute_ids": json.dumps(row["retrieved_statute_ids"], ensure_ascii=False),
                    "visible_evidence_ids": json.dumps(row["visible_evidence_ids"], ensure_ascii=False),
                    "visible_evidence_names": "\n".join(visible_names),
                    "visible_evidence_texts": "\n\n".join(visible_texts),
                    "generated_answer": row["generated_answer"],
                    "auto_gold_visible": row["gold_visible"],
                    "auto_gold_citation_match": row["any_gold_citation"],
                    "auto_citation_fabrication": row["citation_fabrication"],
                    "auto_citation_outside_visible": row["citation_outside_visible_evidence"],
                    "auto_missing_citation": row["missing_citation"],
                    "overall_answer_quality": "",
                    "fabricated_citation_present": "",
                    "wrong_existing_citation_present": "",
                    "unsupported_by_cited_evidence_present": "",
                    "missing_relevant_citation_present": "",
                    "unsupported_extension_present": "",
                    "evidence_misuse_present": "",
                    "retrieval_failure_contributed": "",
                    "primary_failure_type": "",
                    "gold_evidence_semantically_sufficient": "",
                    "failure_origin": "",
                    "reviewer_id": "",
                    "annotation_round": "",
                    "reviewer_confidence": "",
                    "reviewer_notes": "",
                })
            for method in ["direct", "bm25", "hybrid"]:
                row = by_key[(query_id, model, method)]
                visible_names, visible_texts = corpus_names_and_texts(corpus_index, row["visible_evidence_ids"])
                additional_context.append({
                    "query_id": query_id,
                    "stratum": item["stratum"],
                    "original_stratum": item["original_stratum"],
                    "sampling_stratum": item["sampling_stratum"],
                    "stratum_population_N": item["stratum_population_N"],
                    "stratum_sample_n": item["stratum_sample_n"],
                    "selection_probability": item["selection_probability"],
                    "sampling_weight": item["sampling_weight"],
                    "selection_reason": item["selection_reason"],
                    "model": model,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "gold_statute_ids": json.dumps(row["canonical_gold_ids"], ensure_ascii=False),
                    "retrieved_statute_ids": json.dumps(row["retrieved_statute_ids"], ensure_ascii=False),
                    "visible_evidence_ids": json.dumps(row["visible_evidence_ids"], ensure_ascii=False),
                    "visible_evidence_names": "\n".join(visible_names),
                    "visible_evidence_texts": "\n\n".join(visible_texts),
                    "generated_answer": row["generated_answer"],
                    "auto_gold_visible": row["gold_visible"],
                    "auto_gold_citation_match": row["any_gold_citation"],
                    "auto_citation_fabrication": row["citation_fabrication"],
                    "auto_citation_outside_visible": row["citation_outside_visible_evidence"],
                    "auto_missing_citation": row["missing_citation"],
                })
    return {
        "seed": SEED,
        "selection_rule": "Pooled Dense CLS across both models; target 15 per S1-S4, with any under-populated quota redistributed to S4 first as the diagnostic negative-control stratum; source stratum and exact n/N weights are retained.",
        "stratum_available_counts": {key: len(value) for key, value in sorted(strata.items())},
        "stratum_selected_counts": dict(Counter(item["original_stratum"] for item in selected)),
        "sampling_design": "stratified diagnostic sample; unweighted manual proportions are not population prevalence",
        "selected_queries": selected,
        "primary_answers": primary_answers,
        "additional_context": additional_context,
    }


def make_parser_qa_sample(rows: Sequence[Mapping[str, Any]], corpus_index: Any) -> Dict[str, Any]:
    """Create a fixed, non-semantic 100-unit parser audit sample."""

    rng = random.Random(SEED)
    resolved_units: List[Dict[str, Any]] = []
    uncertain_units: List[Dict[str, Any]] = []
    for row in rows:
        for citation in row["citations"]:
            unit = {
                "unit_type": "resolved_citation" if citation.get("parse_status") == "resolved_unique" else "unresolved_or_uncertain_citation",
                "query_id": row["query_id"],
                "model": row["model"],
                "method": row["method"],
                "raw_answer": row["generated_answer"],
                "raw_citation": citation.get("raw_citation", ""),
                "parser_status": citation.get("parse_status"),
                "normalized_law_name": citation.get("normalized_law_name"),
                "article_number": citation.get("article_number"),
                "resolved_statute_id": citation.get("canonical_statute_id"),
                "resolved_statute_name": None,
                "candidate_statute_ids": json.dumps(citation.get("candidate_statute_ids", []), ensure_ascii=False),
                "extraction_correct": "",
                "law_name_correct": "",
                "article_number_correct": "",
                "resolution_correct": "",
                "normalization_correct": "",
                "statute_mapping_correct": "",
                "actually_resolvable": "" if citation.get("parse_status") == "resolved_unique" else "",
                "parser_error_type": "",
                "confidence": "",
                "short_reason": "",
                "error_reason": "",
                "notes": "",
            }
            if citation.get("canonical_statute_id") is not None:
                statute = corpus_index.statute_by_id.get(int(citation["canonical_statute_id"]))
                unit["resolved_statute_name"] = statute.get("statute_name") if statute else None
                resolved_units.append(unit)
            else:
                uncertain_units.append(unit)
    no_citation_answers = [row for row in rows if not row["has_explicit_citation"]]
    if len(resolved_units) < 40 or len(uncertain_units) < 40 or len(no_citation_answers) < 20:
        raise RuntimeError("Parser QA strata are smaller than the requested 40/40/20 sample.")
    selected = []
    selected.extend(rng.sample(resolved_units, 40))
    selected.extend(rng.sample(uncertain_units, 40))
    for row in rng.sample(no_citation_answers, 20):
        selected.append({
            "unit_type": "no_conventional_citation_answer",
            "query_id": row["query_id"],
            "model": row["model"],
            "method": row["method"],
            "raw_answer": row["generated_answer"],
            "raw_citation": "",
            "parser_status": "no_conventional_citation",
            "normalized_law_name": "",
            "article_number": "",
            "resolved_statute_id": "",
            "resolved_statute_name": "",
            "candidate_statute_ids": "[]",
            "extraction_correct": "",
            "law_name_correct": "",
            "article_number_correct": "",
            "resolution_correct": "",
            "normalization_correct": "",
            "statute_mapping_correct": "",
            "actually_resolvable": "",
            "parser_error_type": "",
            "confidence": "",
            "short_reason": "",
            "error_reason": "",
            "notes": "",
        })
    for index, row in enumerate(selected, start=1):
        row["audit_unit_id"] = index
    return {
        "seed": SEED,
        "sample_rule": "40 resolved_unique citations + 40 unresolved/ambiguous/malformed citations + 20 answers with no conventional citation",
        "available_counts": {
            "resolved_unique": len(resolved_units),
            "unresolved_or_uncertain": len(uncertain_units),
            "no_conventional_citation_answer": len(no_citation_answers),
        },
        "selected_counts": {"resolved_unique": 40, "unresolved_or_uncertain": 40, "no_conventional_citation_answer": 20},
        "manual_fields_blank": True,
        "rows": selected,
    }


def annotation_guideline() -> str:
    return """# Phase 3.5B 人工审核指南（仅供人工填写）

本文件对应 `manual_audit.xlsx`。Phase 3.5A 的自动结果只判断：引用是否被解析、法条是否存在、是否命中 gold 集合、是否位于模型实际可见 evidence，以及 gold evidence 是否进入上下文；它不判断法律结论是否正确。

## 填写范围

第一轮只填写 `primary_answers` 工作表中的 Direct 与 Dense 四类答案。`additional_context` 仅用于对照，不作为第一轮主要标注对象。

## 字段定义

- `overall_answer_quality`：`correct` / `partially_correct` / `incorrect` / `uncertain`
- `citation_failure_type`：`none` / `fabricated_citation` / `wrong_existing_citation` / `unsupported_by_cited_evidence` / `missing_relevant_citation` / `mixed` / `uncertain`
- `unsupported_extension`：`yes` / `no` / `uncertain`
- `retrieval_failure_contributed`：`yes` / `no` / `not_applicable` / `uncertain`
- `evidence_misuse`：`yes` / `no` / `uncertain`
- `reviewer_confidence`：`high` / `medium` / `low`

## 重要边界

“法条存在”“citation 命中 gold”“citation 在可见 evidence 中”都不能替代语义支持判断。只有在阅读问题、参考答案、法条正文和模型答案后，才填写 `unsupported_by_cited_evidence`、`evidence_misuse` 或法律结论质量。

- Fabricated Citation：corpus 中不存在该法条。
- Wrong Existing Citation：法条真实存在，但不是当前问题/结论对应的正确引用。
- Unsupported by Cited Evidence：法条存在，但正文不能支持答案中的相关法律主张。
- Missing Relevant Citation：答案提出了需要法律依据的结论，却没有给出应有的显式引用。
- Unsupported Extension：在法条或题目事实之外擅自增加期限、金额、条件、责任形式、例外或程序要求等具体结论。
- Evidence Misuse：模型看到了相关 evidence，但仍然错误解释、扩大或适用它。

不要把语言风格、答案长短或措辞不漂亮本身标为 citation failure。
"""


def annotation_guideline() -> str:
    return """# Phase 3.5B semantic-review guideline

This workbook is a human-review instrument. Phase 3.5A automatically checks citation parsing, corpus existence, set-based gold membership, visibility in the model's evidence, and whether gold evidence entered context. It does not decide legal semantic correctness.

## First-round scope

Review only `primary_answers`: Dense CLS outputs for 60 queries × 2 models = 120 answers. Direct, BM25, and Hybrid remain in `additional_context` for comparison only.

## Multi-label annotation fields

- `overall_answer_quality`: `correct` / `partially_correct` / `incorrect` / `uncertain`
- `fabricated_citation_present`: `yes` / `no` / `uncertain`
- `wrong_existing_citation_present`: `yes` / `no` / `uncertain`
- `unsupported_by_cited_evidence_present`: `yes` / `no` / `uncertain`
- `missing_relevant_citation_present`: `yes` / `no` / `uncertain`
- `unsupported_extension_present`: `yes` / `no` / `uncertain`
- `evidence_misuse_present`: `yes` / `no` / `uncertain`
- `retrieval_failure_contributed`: `yes` / `no` / `not_applicable` / `uncertain`
- `primary_failure_type`: `none` / `fabricated_citation` / `wrong_existing_citation` / `unsupported_by_cited_evidence` / `missing_relevant_citation` / `unsupported_extension` / `evidence_misuse` / `retrieval_failure` / `mixed` / `uncertain`
- `gold_evidence_semantically_sufficient`: `yes` / `partial` / `no` / `uncertain`
- `failure_origin`: `retrieval` / `generation` / `both` / `neither` / `uncertain`
- `reviewer_id` and `annotation_round`: leave blank in this round; reserved for inter-annotator agreement.

Use the multi-label columns to record every observed failure. Use `primary_failure_type` only for the main source; do not hide multiple failures inside `mixed`.

## Boundary

Citation existence, citation-gold membership, and citation visibility do not replace semantic-support judgment. A citation outside gold is not automatically legally wrong, and a citation inside gold is not automatically sufficient. When statutory conflicts or specialist interpretation is unclear, use `uncertain` and `reviewer_confidence=low`; do not guess.

The selected queries retain `original_stratum`, `stratum_population_N`, `stratum_sample_n`, `selection_probability`, and `sampling_weight`. This is a stratified diagnostic sample, not a simple random sample. Unweighted annotation proportions must not be reported as 309-query prevalence; use the stored weights for any post-stratified estimate.
"""


def format_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def build_report(
    integrity: Mapping[str, Any],
    index_audit: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    matrix: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]],
    manual: Mapping[str, Any],
    parser_qa: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    by_key = {(row["model"], row["method"]): row for row in metrics}
    lines = [
        "# Phase 3.5A 自动评估与引用失败诊断",
        "",
        f"- Frozen baseline release commit: `{FROZEN_COMMIT}`",
        "- Scope: 2 models × 4 frozen methods × 309 queries = 2472 answers",
        "- Phase 3.5A.1 / Phase 3.5B-0 status: **complete; stop before Phase 3.5B-1 and Phase 4**",
        "- No answer regeneration, retrieval change, prompt change, claim repair, verifier, or stress test was performed.",
        "",
        "## 1. 冻结 artifact 完整性",
        "",
        f"8 个正式 generation artifact 均通过 SHA256、309 条记录、query_id 唯一、status=ok 检查；总记录数为 `{integrity['answer_count']}`。",
        f"manifest 生成 commit 为 `{integrity['manifest_generation_commit']}`，本阶段按发布节点 `{FROZEN_COMMIT}` 记录。",
        "",
        "## 2. 确定性 citation index",
        "",
        f"corpus 共 `{index_audit['statute_count']}` 条 statute，结构化 canonical key `{index_audit['structured_count']}` 条，重复 key `{index_audit['duplicate_canonical_key_count']}`，无法结构化 `{index_audit['unstructured_count']}`。",
        "法律名称只允许 corpus 内唯一 alias 映射；缺少法律名称、未知法律、非唯一映射均保留为 unresolved/ambiguous。",
        "",
        "## 3. Parser QA 与 marker attribution",
        "",
        f"本轮生成固定 seed={SEED} 的 100-unit parser QA：40 个 resolved_unique、40 个 unresolved/uncertain、20 个无 conventional citation 的 answer。人工字段仍为空，当前不能据此估计 parser precision/false-negative rate。",
        f"答案内部 evidence marker 共单独记录；marker metrics 已写入 `citation_metrics.csv`，不与 conventional citation 合并。",
        "",
        "## 4. 自动 citation / answer 指标",
        "",
        "下表的 `citation gold match` 按 citation occurrence 计算，不等于法律语义正确；BLEU/METEOR/ROUGE-L 是字符/词元层面的 reference similarity proxy。BERTScore 在本环境未安装，因此不作虚构数值。答案中的内部 `[法条ID n]` evidence marker 单独记录，不当作传统法律 citation 计数。",
        "",
        "| Model | Method | Conventional citation presence | Any attribution presence | Citation existence | All exist given citation | All exist joint | Gold citation match | Marker presence | Gold evidence visible | ROUGE-L |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            metric = by_key[(model, method)]
            lines.append(
                f"| {model} | {METHOD_LABELS[method]} | {format_pct(metric['citation_presence_rate'])} | {format_pct(metric['any_attribution_presence_rate'])} | {format_pct(metric['citation_existence_rate'])} | {format_pct(metric['all_citations_exist_given_citation'])} | {format_pct(metric['answer_all_citations_exist_joint'])} | {format_pct(metric['gold_citation_match_rate'])} | {format_pct(metric['evidence_marker_presence_rate'])} | {format_pct(metric['gold_evidence_visible_rate'])} | {metric['quality_rouge_l']:.4f} |"
            )
    lines += [
        "",
        "解释：`citation_existence_rate` 是 citation-level；`all_citations_exist_given_citation` 是有 conventional citation 答案中的 conditional rate；`answer_all_citations_exist_joint` 是全部 309 个答案中的 joint answer-level rate。RAG 的 visible 指标也同时报告 conditional 与 joint 版本。Direct 的 evidence consistency / gold evidence visible 为 N/A，因为 Direct 没有检索 evidence。unresolved/ambiguous citation 不会被自动标成 fabricated，而是保留为 parse-uncertain 并进入人工审核。",
        "",
        "## 5. Dense-centered A/B/C/D diagnostic matrix",
        "",
        "A = gold evidence visible 且答案显式命中 gold；B = gold evidence visible 但没有显式命中 gold；C = gold evidence 不可见但答案命中 gold（只能标记为 requires manual review）；D = 两者均未发生。",
        "",
        "| Model | Retriever | A | B | C | D |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in matrix:
        lines.append(f"| {item['model']} | {item['retriever_label']} | {item['A_count']} ({format_pct(item['A_rate'])}) | {item['B_count']} ({format_pct(item['B_rate'])}) | {item['C_count']} ({format_pct(item['C_rate'])}) | {item['D_count']} ({format_pct(item['D_rate'])}) |")
    lines += [
        "",
        "C 类没有被自动解释为参数记忆、数据泄漏或模型错误；这些只能进入人工审核。",
        "",
        "## 6. Paired bootstrap",
        "",
        f"所有 paired bootstrap 使用同一 query_id 配对、seed={SEED}、{BOOTSTRAP_SAMPLES} 次重采样；`difference = right_method - left_method`。CI 跨 0 时不写 statistically significant。完整结果见 `citation_metric_bootstrap.csv`。",
        "",
        "主要比较（每个模型分别计算）：",
    ]
    for left, right in [("direct", "dense"), ("bm25", "dense"), ("dense", "hybrid")]:
        for model in MODEL_ORDER:
            selected = [row for row in bootstrap if row["model"] == model and row["left_method"] == left and row["right_method"] == right and row["metric"] in {"citation_presence", "answer_all_citations_exist_joint", "any_gold_citation", "answer_all_citations_visible_joint"}]
            if not selected:
                continue
            lines.append(f"- {model} {METHOD_LABELS[left]} → {METHOD_LABELS[right]}：")
            for item in selected:
                significance = "CI跨0，不称显著" if item["ci_low"] <= 0 <= item["ci_high"] else "CI不跨0"
                lines.append(f"  - {item['metric']}: difference={item['difference_right_minus_left']:.4f}, 95% CI=[{item['ci_low']:.4f}, {item['ci_high']:.4f}]（{significance}）")
    lines += [
        "",
        "## 7. 60-query manual audit sample",
        "",
        f"选取 `{len(manual['selected_queries'])}` 个唯一 query_id；按两套 Dense CLS 的 pooled gold_visible / gold_citation_match 分层，seed={SEED}。实际分层可用数：`{manual['stratum_available_counts']}`；选入数：`{manual['stratum_selected_counts']}`。每条样本保留 original_stratum、selection_probability 和 sampling_weight；该样本用于 failure diagnosis，不能把未加权的人工比例直接解释为 309-query population prevalence。",
        "",
        "`manual_audit.xlsx` 的 primary_answers 现在只包含 Dense CLS 的 60 queries × 2 models = 120 条首轮语义审核对象；Direct/BM25/Hybrid 保留在 additional_context 作为比较背景，所有人工字段保持空白。",
        "",
        "## 8. 当前可以支持的结论",
        "",
        "1. 可以报告不同 baseline 的显式 citation presence、解析成功率、corpus existence、set-based gold match，以及 RAG 的实际 visible-evidence consistency。",
        "2. 可以报告 Dense 的 retrieval gold visibility 是否传递到显式 citation 行为，并用 A/B/C/D 量化 retrieval 与 generation/citation 的关系。",
        "3. 可以报告 Direct、BM25、Dense、Hybrid 的 answer/reference lexical similarity proxy，但不得把这些 proxy 直接写成 legal correctness。",
        "",
        "## 9. 当前不能支持的结论",
        "",
        "1. 自动指标不能证明 citation 对答案主张具有法律语义支持，也不能自动判断 wrong legal conclusion、unsupported extension 或 evidence misuse。",
        "2. citation 在 gold 中不等于法律上正确；citation 不在 gold 中也不能自动等同于法律错误。",
        "3. C 类不能自动归因于参数记忆；Dense 的点估计优势不能据此写成显著优于 Hybrid。",
        "",
        "## 10. 阶段停止点",
        "",
        "Phase 3.5A.1 / Phase 3.5B-0 已完成：指标口径 hotfix、parser QA、sampling manifest、Dense-only manual_audit.xlsx、annotation_guideline.md、checkpoint report 均已生成。parser QA 与语义人工审核仍待填写；本次不进入 Phase 3.5B-1，不进入 Phase 4。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    report_dir = ROOT / "reports/phase3_5"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(ROOT / "reports/final_baseline_manifest.json")
    loaded, integrity = artifact_integrity(manifest)
    query_rows, _ = load_gold_and_queries()
    corpus_rows = read_jsonl(ROOT / "data/processed/corpus.jsonl")
    corpus_index = build_citation_index(corpus_rows)
    index_audit = corpus_index.audit()
    write_json(report_dir / "citation_index_audit.json", {
        "index_version": "phase3_5_canonical_law_article_v1",
        **index_audit,
        "ambiguous_key_examples": index_audit.get("ambiguous_key_examples", []),
    })
    evaluation_rows = build_evaluation_rows(loaded, query_rows, corpus_index)
    write_jsonl(report_dir / "citation_records.jsonl", evaluation_rows)
    # The requested per-answer quality artifact is intentionally separate from
    # citation_records so it can be loaded by statistical tooling directly.
    quality_rows = []
    for row in evaluation_rows:
        quality_rows.append({
            "query_id": row["query_id"],
            "model": row["model"],
            "method": row["method"],
            "reference_answer": row["reference_answer"],
            "generated_answer": row["generated_answer"],
            **row["quality"],
        })
    write_jsonl(report_dir / "per_answer_quality.jsonl", quality_rows)
    metrics = automatic_metrics(evaluation_rows)
    matrix = diagnostic_matrix(evaluation_rows)
    bootstrap = paired_bootstrap(evaluation_rows)
    write_csv(report_dir / "citation_metrics.csv", metrics, fieldnames=list(metrics[0].keys()))
    write_json(report_dir / "citation_metrics.json", {
        "analysis_version": "phase3_5B0_v1_analysis_audit_hotfix",
        "frozen_commit": FROZEN_COMMIT,
        "answer_count": len(evaluation_rows),
        "seed": SEED,
        "bertscore": "not_run: bert_score package unavailable",
        "metrics": metrics,
    })
    write_csv(report_dir / "retrieval_citation_matrix.csv", matrix, fieldnames=list(matrix[0].keys()))
    write_csv(report_dir / "citation_metric_bootstrap.csv", bootstrap, fieldnames=list(bootstrap[0].keys()))
    write_csv(report_dir / "answer_quality.csv", metrics, fieldnames=[
        "model", "method", "method_label", "answer_count", "quality_bertscore_f1", "quality_bleu", "quality_meteor", "quality_rouge_l", "quality_metric_note",
    ])
    manual = make_manual_selection(evaluation_rows, corpus_index)
    parser_qa = make_parser_qa_sample(evaluation_rows, corpus_index)
    write_json(report_dir / "manual_audit_data.json", manual)
    write_json(report_dir / "parser_qa_data.json", parser_qa)
    write_json(report_dir / "sampling_manifest.json", {
        "sampling_version": "phase3_5B0_stratified_dense_pooled_v2",
        "seed": SEED,
        "selection_rule": manual["selection_rule"],
        "stratum_available_counts": manual["stratum_available_counts"],
        "stratum_selected_counts": manual["stratum_selected_counts"],
        "sampling_design": manual["sampling_design"],
        "selected_queries": [
            {
                key: item[key]
                for key in [
                    "query_id", "original_stratum", "sampling_stratum",
                    "stratum_population_N", "stratum_sample_n",
                    "selection_probability", "sampling_weight", "selection_reason",
                ]
            }
            for item in manual["selected_queries"]
        ],
    })
    guideline = annotation_guideline()
    (report_dir / "annotation_guideline.md").write_text(guideline, encoding="utf-8")
    report = build_report(integrity, index_audit, metrics, matrix, bootstrap, manual, parser_qa, evaluation_rows)
    (report_dir / "phase3_5A_report_zh.md").write_text(report, encoding="utf-8")
    checkpoint = "# Phase 3.5A Checkpoint Report\n\n" + report.replace("# Phase 3.5A 自动评估与引用失败诊断", "## Phase 3.5A 自动评估与引用失败诊断")
    (report_dir / "phase3_5_checkpoint_report.md").write_text(checkpoint, encoding="utf-8")
    write_json(report_dir / "phase3_5A_run_manifest.json", {
        "analysis_version": "phase3_5B0_v1_analysis_audit_hotfix",
        "frozen_commit": FROZEN_COMMIT,
        "manifest_version": manifest.get("manifest_version"),
        "integrity": integrity,
        "index_audit": index_audit,
        "answer_count": len(evaluation_rows),
        "citation_count": sum(int(row["citation_count"]) for row in evaluation_rows),
        "resolved_citation_count": sum(int(row["resolved_citation_count"]) for row in evaluation_rows),
        "manual_sample": {
            "query_count": len(manual["selected_queries"]),
            "stratum_available_counts": manual["stratum_available_counts"],
            "stratum_selected_counts": manual["stratum_selected_counts"],
            "seed": manual["seed"],
        },
        "parser_qa_sample": {
            "selected_counts": parser_qa["selected_counts"],
            "available_counts": parser_qa["available_counts"],
            "manual_fields_blank": parser_qa["manual_fields_blank"],
        },
        "outputs": [
            "citation_index_audit.json",
            "citation_records.jsonl",
            "per_answer_quality.jsonl",
            "answer_quality.csv",
            "citation_metrics.csv",
            "citation_metrics.json",
            "retrieval_citation_matrix.csv",
            "citation_metric_bootstrap.csv",
            "manual_audit_data.json",
            "sampling_manifest.json",
            "parser_qa_data.json",
            "parser_qa.xlsx",
            "annotation_guideline.md",
            "phase3_5A_report_zh.md",
            "phase3_5_checkpoint_report.md",
        ],
        "status": "complete_stop_before_phase3_5B1",
    })
    print(json.dumps({
        "status": "complete_stop_before_phase3_5B1",
        "answer_count": len(evaluation_rows),
        "citation_count": sum(int(row["citation_count"]) for row in evaluation_rows),
        "resolved_citation_count": sum(int(row["resolved_citation_count"]) for row in evaluation_rows),
        "manual_query_count": len(manual["selected_queries"]),
        "stratum_available_counts": manual["stratum_available_counts"],
        "stratum_selected_counts": manual["stratum_selected_counts"],
        "outputs": str(report_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
