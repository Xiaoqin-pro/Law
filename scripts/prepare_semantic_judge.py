"""Prepare and validate the blinded Dense semantic-judgment set.

The script never fills a human workbook and never changes frozen generation or
retrieval artifacts. It creates anonymous judge inputs from the existing
60-query Dense diagnostic sample and keeps the identity map separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs/phase3_5/semantic_model_assisted"
REPORT_DIR = ROOT / "reports/phase3_5/semantic_judge"
MANUAL_DATA = ROOT / "reports/phase3_5/manual_audit_data.json"
V2_RECORDS = ROOT / "outputs/phase3_5/parser_v2/citation_records_v2.jsonl"
CORPUS = ROOT / "data/processed/corpus.jsonl"
FROZEN_COMMIT = "b3679ade8b233b71e5d3812dd44184aabee05f8f"
PARSER_VERSION = "citation_parser_v2"
SAMPLE_SIZE = 120

ALLOWED = {
    "overall_answer_quality": {"correct", "partially_correct", "incorrect", "uncertain"},
    "fabricated_citation_present": {"yes", "no", "uncertain"},
    "wrong_existing_citation_present": {"yes", "no", "uncertain"},
    "unsupported_by_cited_evidence_present": {"yes", "no", "uncertain"},
    "missing_relevant_citation_present": {"yes", "no", "uncertain"},
    "unsupported_extension_present": {"yes", "no", "uncertain"},
    "evidence_misuse_present": {"yes", "no", "uncertain"},
    "retrieval_failure_contributed": {"yes", "no", "not_applicable", "uncertain"},
    "gold_evidence_semantically_sufficient": {"yes", "partial", "no", "uncertain"},
    "failure_origin": {"retrieval", "generation", "both", "neither", "uncertain"},
    "primary_failure_type": {
        "none", "fabricated_citation", "wrong_existing_citation",
        "unsupported_by_cited_evidence", "missing_relevant_citation",
        "unsupported_extension", "evidence_misuse", "retrieval_failure",
        "mixed", "uncertain",
    },
    "reviewer_confidence": {"high", "medium", "low"},
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


def prepare() -> Dict[str, Any]:
    manual = read_json(MANUAL_DATA)
    primary = manual["primary_answers"]
    if len(primary) != SAMPLE_SIZE:
        raise RuntimeError(f"Expected {SAMPLE_SIZE} Dense primary answers, got {len(primary)}")
    v2_rows = {
        (row["model"], row["method"], int(row["query_id"])): row
        for row in read_jsonl(V2_RECORDS)
        if row["method"] == "dense"
    }
    corpus = {int(row["statute_id"]): row for row in read_jsonl(CORPUS)}
    blinded: List[Dict[str, Any]] = []
    identity: List[Dict[str, Any]] = []
    for index, row in enumerate(sorted(primary, key=lambda item: (int(item["query_id"]), item["model"])), start=1):
        case_id = f"answer_{index:03d}"
        evidence_set_id = f"evidence_{index:03d}"
        v2 = v2_rows[(row["model"], row["method"], int(row["query_id"]))]
        gold_ids = json.loads(row["gold_statute_ids"])
        visible_ids = json.loads(row["visible_evidence_ids"])
        parsed_citations = []
        for citation in v2["citations"]:
            item = {
                "raw_citation": citation.get("raw_citation"),
                "normalized_citation": citation.get("normalized_citation"),
                "resolved_statute_name": None,
                "resolved_statute_text": None,
                "parser_status": citation.get("parse_status"),
                "parser_confidence": citation.get("confidence"),
                "resolution_method": citation.get("resolution_method"),
                "repair_applied": citation.get("repair_applied"),
            }
            statute_id = citation.get("canonical_statute_id")
            if statute_id is not None and int(statute_id) in corpus:
                item["resolved_statute_name"] = corpus[int(statute_id)]["statute_name"]
                item["resolved_statute_text"] = corpus[int(statute_id)]["statute_text"]
            parsed_citations.append(item)
        blinded.append({
            "answer_id": case_id,
            "evidence_set_id": evidence_set_id,
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "gold_evidence": [
                {"statute_name": corpus[int(statute_id)]["statute_name"], "statute_text": corpus[int(statute_id)]["statute_text"]}
                for statute_id in gold_ids
                if int(statute_id) in corpus
            ],
            "visible_evidence": [
                {"statute_name": corpus[int(statute_id)]["statute_name"], "statute_text": corpus[int(statute_id)]["statute_text"]}
                for statute_id in visible_ids
                if int(statute_id) in corpus
            ],
            "generated_answer": row["generated_answer"],
            "parser_version": PARSER_VERSION,
            "parsed_citations": parsed_citations,
        })
        identity.append({
            "answer_id": case_id,
            "evidence_set_id": evidence_set_id,
            "query_id": int(row["query_id"]),
            "model": row["model"],
            "method": row["method"],
            "original_stratum": row["original_stratum"],
            "sampling_weight": row["sampling_weight"],
            "selection_probability": row["selection_probability"],
            "gold_visible": row["auto_gold_visible"],
            "auto_gold_citation_match": row["auto_gold_citation_match"],
            "auto_citation_outside_visible": row["auto_citation_outside_visible"],
            "auto_missing_citation": row["auto_missing_citation"],
            "v2_citation_record_key": [row["model"], row["method"], int(row["query_id"])],
        })
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(INPUT_DIR / "semantic_judge_inputs_blinded.jsonl", blinded)
    write_jsonl(INPUT_DIR / "semantic_judge_identity_map.jsonl", identity)
    prompt = """You are an independent legal-evidence judgment model.

Only use the question, benchmark reference answer, gold statute texts, visible evidence texts, generated answer, and parser metadata supplied in the case. Do not use outside legal knowledge. If the supplied evidence is insufficient or the legal interpretation requires outside knowledge, use uncertain. A citation outside gold is not automatically wrong; a gold citation is not automatically supported; parser unresolved is not automatically fabrication.

Return JSON with: overall_answer_quality, fabricated_citation_present, wrong_existing_citation_present, unsupported_by_cited_evidence_present, missing_relevant_citation_present, unsupported_extension_present, evidence_misuse_present, retrieval_failure_contributed, gold_evidence_semantically_sufficient, failure_origin, primary_failure_type, reviewer_confidence, evidence_for_judgment, short_reason, problematic_claims.

Do not infer model identity, method, stratum, sampling weight, or automatic failure category.
"""
    prompt_path = INPUT_DIR / "semantic_judge_prompt_v1.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    manifest = {
        "analysis_version": "phase3_5B1_independent_semantic_judge_prepare_v1",
        "status": "INPUTS_PREPARED_NO_LABELS_FILLED",
        "frozen_commit": FROZEN_COMMIT,
        "parser_version": PARSER_VERSION,
        "answer_count": len(blinded),
        "scope": "60 stratified queries x 2 Dense models = 120 answers",
        "methods_in_judge_input": ["dense"],
        "model_identity_blinded": True,
        "automatic_failure_categories_blinded": True,
        "prompt_sha256": sha256(prompt_path),
        "input_sha256": sha256(INPUT_DIR / "semantic_judge_inputs_blinded.jsonl"),
        "identity_map_sha256": sha256(INPUT_DIR / "semantic_judge_identity_map.jsonl"),
        "judge_model": "independent Codex model; interactive model-assisted mode",
        "judge_revision": "runtime-provided; no external completion API",
        "temperature": 0,
        "latency_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "telemetry_note": "No completion API telemetry is available in this local interactive judge mode.",
        "human_validation": False,
        "outputs": [
            "outputs/phase3_5/semantic_model_assisted/semantic_judge_inputs_blinded.jsonl",
            "outputs/phase3_5/semantic_model_assisted/semantic_judge_identity_map.jsonl",
            "outputs/phase3_5/semantic_model_assisted/semantic_judge_prompt_v1.txt",
        ],
    }
    write_json(REPORT_DIR / "semantic_judge_manifest.json", manifest)
    return manifest


def validate_annotations(path: Path) -> Dict[str, Any]:
    rows = read_jsonl(path)
    expected = {f"answer_{index:03d}" for index in range(1, SAMPLE_SIZE + 1)}
    observed = {str(row.get("answer_id")) for row in rows}
    errors: List[str] = []
    if observed != expected:
        errors.append("answer_id set is not exactly answer_001..answer_120")
    if len(rows) != SAMPLE_SIZE:
        errors.append(f"expected {SAMPLE_SIZE} rows, got {len(rows)}")
    for row in rows:
        for field, allowed in ALLOWED.items():
            value = row.get(field)
            if value not in allowed:
                errors.append(f"{row.get('answer_id')}: invalid {field}={value!r}")
        for required in ["evidence_for_judgment", "short_reason", "problematic_claims"]:
            if required not in row:
                errors.append(f"{row.get('answer_id')}: missing {required}")
        if not isinstance(row.get("problematic_claims"), list):
            errors.append(f"{row.get('answer_id')}: problematic_claims must be a list")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "row_count": len(rows),
        "unique_answer_ids": len(observed),
        "errors": errors,
        "model_assisted_only": True,
        "not_human_validation": True,
    }
    write_json(REPORT_DIR / "semantic_judge_validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    validate = sub.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(validate_annotations(args.input), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
