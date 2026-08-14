"""Validate the Dense-only manual audit workbook and summarize completion.

This script never fills annotation cells.  It verifies the immutable sample
against the JSON source, validates the annotation schema, and emits only a
progress report until all required human fields are complete.  Once complete,
it produces unweighted diagnostic counts and query-level post-stratified
estimates, keeping them separate from population claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PHASE_DIR = ROOT / "reports/phase3_5"
SEED = 42
BOOTSTRAP_SAMPLES = 10_000
MODELS = ["qwen25_7b", "qwen3_4b"]
STRATA = [
    "S1_gold_visible_and_gold_cited",
    "S2_gold_visible_but_not_gold_cited",
    "S3_gold_not_visible_but_gold_cited",
    "S4_gold_not_visible_and_not_gold_cited",
]
MULTI_LABEL_FIELDS = [
    "fabricated_citation_present",
    "wrong_existing_citation_present",
    "unsupported_by_cited_evidence_present",
    "missing_relevant_citation_present",
    "unsupported_extension_present",
    "evidence_misuse_present",
]
BINARY_OR_NA_FIELDS = MULTI_LABEL_FIELDS + ["retrieval_failure_contributed"]
REQUIRED_FIELDS = [
    "overall_answer_quality",
    *MULTI_LABEL_FIELDS,
    "retrieval_failure_contributed",
    "primary_failure_type",
    "gold_evidence_semantically_sufficient",
    "failure_origin",
    "reviewer_id",
    "annotation_round",
    "reviewer_confidence",
]
OPTIONAL_FIELDS = ["reviewer_notes"]
EDITABLE_FIELDS = [*REQUIRED_FIELDS, *OPTIONAL_FIELDS]
ALLOWED: Dict[str, set[str]] = {
    "overall_answer_quality": {"correct", "partially_correct", "incorrect", "uncertain"},
    **{field: {"yes", "no", "uncertain"} for field in MULTI_LABEL_FIELDS},
    "retrieval_failure_contributed": {"yes", "no", "not_applicable", "uncertain"},
    "primary_failure_type": {"none", "fabricated_citation", "wrong_existing_citation", "unsupported_by_cited_evidence", "missing_relevant_citation", "unsupported_extension", "evidence_misuse", "retrieval_failure", "mixed", "uncertain"},
    "gold_evidence_semantically_sufficient": {"yes", "partial", "no", "uncertain"},
    "failure_origin": {"retrieval", "generation", "both", "neither", "uncertain"},
    "reviewer_confidence": {"high", "medium", "low"},
}
IMMUTABLE_FIELDS = [
    "query_id", "model", "method", "original_stratum", "sampling_stratum",
    "stratum_population_N", "stratum_sample_n", "selection_probability",
    "sampling_weight", "question", "reference_answer", "generated_answer",
]


def node_executable() -> str:
    configured = os.environ.get("CODEX_NODE")
    if configured:
        return configured
    bundled = Path(r"C:\Users\111\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if bundled.exists():
        return str(bundled)
    return shutil.which("node") or "node"


def load_workbook_sheet(path: Path, sheet_name: str) -> List[Dict[str, Any]]:
    bridge = ROOT / "scripts/workbook_to_json.mjs"
    result = subprocess.run(
        [node_executable(), str(bridge), str(path), sheet_name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "artifact-tool workbook import failed")
    return json.loads(result.stdout)["rows"]


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalized(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else value
    return text(value)


def equal_value(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return normalized(left) == normalized(right)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def row_key(row: Mapping[str, Any]) -> Tuple[int, str]:
    return int(float(row["query_id"])), text(row["model"])


def validate_structure(rows: Sequence[Mapping[str, Any]], source_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    if len(rows) != 120:
        errors.append({"type": "row_count", "expected": 120, "actual": len(rows)})
    keys = [row_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        errors.append({"type": "duplicate_query_model_rows"})
    if any(text(row.get("method")) != "dense" for row in rows):
        errors.append({"type": "non_dense_primary_row"})
    if Counter(text(row.get("model")) for row in rows) != Counter({"qwen25_7b": 60, "qwen3_4b": 60}):
        errors.append({"type": "model_counts", "counts": dict(Counter(text(row.get("model")) for row in rows))})
    source_by_key = {row_key(row): row for row in source_rows}
    for row in rows:
        key = row_key(row)
        source = source_by_key.get(key)
        if source is None:
            errors.append({"type": "unexpected_row", "key": key})
            continue
        for field in IMMUTABLE_FIELDS:
            if field in source and not equal_value(row.get(field), source.get(field)):
                errors.append({"type": "immutable_mismatch", "key": key, "field": field})
    manifest_by_query = {int(item["query_id"]): item for item in manifest.get("selected_queries", [])}
    for row in rows:
        query_id = int(float(row["query_id"]))
        selected = manifest_by_query.get(query_id)
        if selected is None:
            errors.append({"type": "query_missing_from_sampling_manifest", "query_id": query_id})
            continue
        for field in ("original_stratum", "sampling_weight", "selection_probability"):
            if field in selected and not equal_value(row.get(field), selected[field]):
                errors.append({"type": "sampling_metadata_mismatch", "query_id": query_id, "field": field})
    return errors


def validate_annotations(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, Any]] = []
    for row in rows:
        key = f"{text(row.get('query_id'))}/{text(row.get('model'))}"
        missing = []
        for field in REQUIRED_FIELDS:
            current = text(row.get(field))
            if not current:
                missing.append(field)
            elif field in ALLOWED and current not in ALLOWED[field]:
                errors.append({"type": "invalid_annotation_value", "key": key, "field": field, "value": current})
        if missing:
            incomplete.append({"key": key, "query_id": text(row.get("query_id")), "model": text(row.get("model")), "missing_fields": missing})
    return errors, incomplete


def group_label_counts(rows: Sequence[Mapping[str, Any]], group_field: Optional[str], group_value: Optional[str]) -> List[Dict[str, Any]]:
    selected = [row for row in rows if group_field is None or text(row.get(group_field)) == group_value]
    output: List[Dict[str, Any]] = []
    metrics = ["overall_answer_quality", *BINARY_OR_NA_FIELDS, "gold_evidence_semantically_sufficient", "failure_origin", "primary_failure_type", "reviewer_confidence"]
    for metric_name in metrics:
        counts = Counter(text(row.get(metric_name)) for row in selected)
        for label, count in sorted(counts.items()):
            output.append({
                "scope": "overall" if group_field is None else group_field,
                "group": "all" if group_value is None else group_value,
                "metric": metric_name,
                "label": label,
                "count": count,
                "denominator": len(selected),
            })
    return output


def query_rows(rows: Sequence[Mapping[str, Any]], model: str) -> Dict[int, Mapping[str, Any]]:
    return {int(float(row["query_id"])): row for row in rows if text(row.get("model")) == model}


def weighted_estimate(query_map: Mapping[int, Mapping[str, Any]], field: str, positive_labels: set[str] = {"yes"}) -> Dict[str, Any]:
    observed = [row for row in query_map.values() if text(row.get(field)) in {"yes", "no"}]
    weights = [float(row.get("sampling_weight", 0) or 0) for row in observed]
    denom = sum(weights)
    estimate = (sum(weight for weight, row in zip(weights, observed) if text(row.get(field)) in positive_labels) / denom) if denom else None
    uncertain_rate = sum(float(row.get("sampling_weight", 0) or 0) for row in query_map.values() if text(row.get(field)) in {"uncertain", ""}) / sum(float(row.get("sampling_weight", 0) or 0) for row in query_map.values()) if query_map else None
    rng = random.Random(SEED)
    ids = list(query_map)
    bootstrap: List[float] = []
    if ids and estimate is not None:
        for _ in range(BOOTSTRAP_SAMPLES):
            sample = [query_map[rng.choice(ids)] for _ in ids]
            sample_observed = [row for row in sample if text(row.get(field)) in {"yes", "no"}]
            sample_weights = [float(row.get("sampling_weight", 0) or 0) for row in sample_observed]
            sample_denom = sum(sample_weights)
            if sample_denom:
                bootstrap.append(sum(weight for weight, row in zip(sample_weights, sample_observed) if text(row.get(field)) in positive_labels) / sample_denom)
    return {
        "field": field,
        "weighted_estimate": estimate,
        "weighted_uncertain_or_unobserved_rate": uncertain_rate,
        "observed_query_count": len(observed),
        "total_query_count": len(query_map),
        "bootstrap_samples": BOOTSTRAP_SAMPLES if bootstrap else 0,
        "bootstrap_ci_low": sorted(bootstrap)[int(len(bootstrap) * 0.025)] if bootstrap else None,
        "bootstrap_ci_high": sorted(bootstrap)[min(len(bootstrap) - 1, int(len(bootstrap) * 0.975))] if bootstrap else None,
        "bootstrap_unit": "query_id",
        "seed": SEED,
    }


def paired_model_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_model = {model: query_rows(rows, model) for model in MODELS}
    common = sorted(set(by_model[MODELS[0]]) & set(by_model[MODELS[1]]))
    output = []
    for field in ["overall_answer_quality", *BINARY_OR_NA_FIELDS, "gold_evidence_semantically_sufficient", "failure_origin", "primary_failure_type"]:
        pairs = [(text(by_model[MODELS[0]][query_id].get(field)), text(by_model[MODELS[1]][query_id].get(field))) for query_id in common]
        output.append({
            "metric": field,
            "n_queries": len(pairs),
            "qwen25_distribution": json.dumps(dict(Counter(left for left, _ in pairs)), ensure_ascii=False, sort_keys=True),
            "qwen3_distribution": json.dumps(dict(Counter(right for _, right in pairs)), ensure_ascii=False, sort_keys=True),
            "exact_agreement_count": sum(left == right for left, right in pairs),
            "exact_agreement_rate": sum(left == right for left, right in pairs) / len(pairs) if pairs else None,
            "note": "Paired descriptive comparison; this is not a claim of independent samples.",
        })
    return output


def write_complete_outputs(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    raw_rows: List[Dict[str, Any]] = []
    raw_rows.extend(group_label_counts(rows, None, None))
    for model in MODELS:
        raw_rows.extend(group_label_counts(rows, "model", model))
    for stratum in STRATA:
        raw_rows.extend(group_label_counts(rows, "original_stratum", stratum))
    write_csv(output_dir / "manual_failure_counts.csv", raw_rows, ["scope", "group", "metric", "label", "count", "denominator"])

    weighted_rows = []
    for model in MODELS:
        model_map = query_rows(rows, model)
        for field in [*BINARY_OR_NA_FIELDS, "gold_evidence_semantically_sufficient"]:
            weighted_rows.append({"model": model, **weighted_estimate(model_map, field)})
    write_csv(output_dir / "manual_failure_prevalence.csv", weighted_rows, ["model", "field", "weighted_estimate", "weighted_uncertain_or_unobserved_rate", "observed_query_count", "total_query_count", "bootstrap_samples", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_unit", "seed"])
    write_csv(output_dir / "semantic_review_summary.csv", raw_rows, ["scope", "group", "metric", "label", "count", "denominator"])
    write_csv(output_dir / "semantic_review_weighted.csv", weighted_rows, ["model", "field", "weighted_estimate", "weighted_uncertain_or_unobserved_rate", "observed_query_count", "total_query_count", "bootstrap_samples", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_unit", "seed"])
    write_csv(output_dir / "manual_model_comparison.csv", paired_model_rows(rows), ["metric", "n_queries", "qwen25_distribution", "qwen3_distribution", "exact_agreement_count", "exact_agreement_rate", "note"])

    origin_rows = []
    for stratum in STRATA:
        for origin in sorted({text(row.get("failure_origin")) for row in rows if text(row.get("original_stratum")) == stratum}):
            subset = [row for row in rows if text(row.get("original_stratum")) == stratum and text(row.get("failure_origin")) == origin]
            origin_rows.append({"dimension": "failure_origin", "stratum": stratum, "label": origin, "count": len(subset), "denominator": sum(text(row.get("original_stratum")) == stratum for row in rows)})
    for sufficiency in sorted({text(row.get("gold_evidence_semantically_sufficient")) for row in rows}):
        for origin in sorted({text(row.get("failure_origin")) for row in rows}):
            subset = [row for row in rows if text(row.get("gold_evidence_semantically_sufficient")) == sufficiency and text(row.get("failure_origin")) == origin]
            origin_rows.append({"dimension": "gold_evidence_semantically_sufficient_x_failure_origin", "stratum": sufficiency, "label": origin, "count": len(subset), "denominator": sum(text(row.get("gold_evidence_semantically_sufficient")) == sufficiency for row in rows)})
    write_csv(output_dir / "failure_origin_analysis.csv", origin_rows, ["dimension", "stratum", "label", "count", "denominator"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PHASE_DIR / "manual_audit.xlsx")
    parser.add_argument("--source", type=Path, default=PHASE_DIR / "manual_audit_data.json")
    parser.add_argument("--sampling", type=Path, default=PHASE_DIR / "sampling_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=PHASE_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_workbook_sheet(args.input, "primary_answers")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    sampling = json.loads(args.sampling.read_text(encoding="utf-8"))
    structural_errors = validate_structure(rows, source["primary_answers"], sampling)
    annotation_errors, incomplete = validate_annotations(rows)
    completed = len(rows) - len(incomplete)
    progress = {
        "analysis_version": "phase3_5B1_manual_validation_v1",
        "status": "WAITING_FOR_HUMAN_ANNOTATION" if incomplete else "COMPLETE",
        "total_rows": len(rows),
        "completed_rows": completed,
        "incomplete_rows": len(incomplete),
        "structural_errors": structural_errors,
        "annotation_errors": annotation_errors,
        "incomplete_examples": incomplete[:50],
        "by_model_completed": {model: sum(not any(item["key"] == f"{text(row.get('query_id'))}/{model}" for item in incomplete) for row in rows if text(row.get("model")) == model) for model in MODELS},
        "by_stratum_completed": {stratum: sum(not any(item["key"] == f"{text(row.get('query_id'))}/{text(row.get('model'))}" for item in incomplete) for row in rows if text(row.get("original_stratum")) == stratum) for stratum in STRATA},
        "uncertain_counts": {field: sum(text(row.get(field)) == "uncertain" for row in rows) for field in ["overall_answer_quality", *BINARY_OR_NA_FIELDS, "gold_evidence_semantically_sufficient", "failure_origin", "primary_failure_type"]},
        "low_confidence_count": sum(text(row.get("reviewer_confidence")) == "low" for row in rows),
        "input_sha256": sha256(args.input),
        "sampling_design": "stratified diagnostic sample; weighted outputs are query-level post-stratified estimates, not raw 120-row prevalence",
    }
    write_json(args.output_dir / "manual_annotation_progress.json", progress)
    if not structural_errors and not annotation_errors and not incomplete:
        write_complete_outputs(rows, args.output_dir)
    print(json.dumps({key: progress[key] for key in ("status", "total_rows", "completed_rows", "incomplete_rows", "structural_errors", "annotation_errors")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
