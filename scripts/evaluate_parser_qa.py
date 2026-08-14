"""Evaluate the human-entered parser QA workbook without filling it.

The workbook is read through ``workbook_to_json.mjs`` and artifact-tool.  This
script deliberately treats blank manual fields as incomplete rather than
silently substituting the parser's own output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE_DIR = ROOT / "reports/phase3_5"
VALID_TRI_STATE = {"yes", "no", "uncertain"}


def node_executable() -> str:
    configured = os.environ.get("CODEX_NODE")
    if configured:
        return configured
    bundled = Path(r"C:\Users\111\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if bundled.exists():
        return str(bundled)
    return shutil.which("node") or "node"


def load_workbook_sheet(path: Path, sheet_name: str) -> Dict[str, Any]:
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
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"artifact-tool bridge returned invalid JSON: {result.stdout[:500]}") from exc


def value(row: Mapping[str, Any], key: str) -> str:
    raw = row.get(key, "")
    return "" if raw is None else str(raw).strip()


def first_value(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        current = value(row, key)
        if current:
            return current
    return ""


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


def field_values(row: Mapping[str, Any]) -> Dict[str, str]:
    # Support both the original B-0 names and the more granular B-1 names.
    return {
        "extraction_correct": value(row, "extraction_correct"),
        "law_name_correct": first_value(row, ("law_name_correct", "normalization_correct")),
        "article_number_correct": first_value(row, ("article_number_correct", "normalization_correct")),
        "resolution_correct": first_value(row, ("resolution_correct", "statute_mapping_correct")),
        "actually_resolvable": value(row, "actually_resolvable"),
        "parser_error_type": value(row, "parser_error_type"),
        "notes": value(row, "notes"),
    }


def annotation_complete(row: Mapping[str, Any]) -> bool:
    fields = field_values(row)
    unit_type = value(row, "unit_type")
    if unit_type == "resolved_citation":
        required = [fields[name] for name in ("extraction_correct", "law_name_correct", "article_number_correct", "resolution_correct")]
    elif unit_type == "unresolved_or_uncertain_citation":
        required = [fields[name] for name in ("extraction_correct", "law_name_correct", "article_number_correct", "resolution_correct", "actually_resolvable")]
    elif unit_type == "no_conventional_citation_answer":
        required = [fields["extraction_correct"]]
    else:
        return False
    return all(item in VALID_TRI_STATE for item in required)


def validate_allowed_values(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for row in rows:
        row_id = value(row, "audit_unit_id") or value(row, "query_id")
        fields = field_values(row)
        for field in ("extraction_correct", "law_name_correct", "article_number_correct", "resolution_correct", "actually_resolvable"):
            current = fields[field]
            if current and current not in VALID_TRI_STATE:
                errors.append({"audit_unit_id": row_id, "field": field, "value": current, "reason": "invalid_value"})
    return errors


def metric(labels: Sequence[str]) -> Dict[str, Any]:
    observed = [label for label in labels if label in {"yes", "no"}]
    return {
        "yes": labels.count("yes"),
        "no": labels.count("no"),
        "uncertain": labels.count("uncertain"),
        "blank": labels.count(""),
        "observed_n": len(observed),
        "accuracy": (observed.count("yes") / len(observed)) if observed else None,
    }


def evaluate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups = {
        "resolved_citation": [row for row in rows if value(row, "unit_type") == "resolved_citation"],
        "unresolved_or_uncertain_citation": [row for row in rows if value(row, "unit_type") == "unresolved_or_uncertain_citation"],
        "no_conventional_citation_answer": [row for row in rows if value(row, "unit_type") == "no_conventional_citation_answer"],
    }
    fields_by_group: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, group in groups.items():
        field_labels = {field: [field_values(row)[field] for row in group] for field in field_values(group[0]).keys()} if group else {}
        fields_by_group[name] = {field: metric(labels) for field, labels in field_labels.items()}

    resolved = groups["resolved_citation"]
    resolved_resolution = [field_values(row)["resolution_correct"] for row in resolved]
    unresolved = groups["unresolved_or_uncertain_citation"]
    unresolved_resolvable = [field_values(row)["actually_resolvable"] for row in unresolved]
    no_citation = groups["no_conventional_citation_answer"]
    no_citation_extraction = [field_values(row)["extraction_correct"] for row in no_citation]

    error_counter = Counter(
        field_values(row)["parser_error_type"]
        for row in rows
        if field_values(row)["parser_error_type"]
    )
    systematic_flags = [
        {"parser_error_type": error_type, "count": count}
        for error_type, count in error_counter.most_common()
        if count >= 3
    ]
    completed = sum(annotation_complete(row) for row in rows)
    total = len(rows)
    manual_errors = validate_allowed_values(rows)
    resolved_precision = metric(resolved_resolution)
    unresolved_metric = metric(unresolved_resolvable)
    no_citation_metric = metric(no_citation_extraction)
    all_required_complete = completed == total and not manual_errors
    acceptance_pass = bool(
        all_required_complete
        and resolved_precision["accuracy"] is not None
        and resolved_precision["accuracy"] >= 0.95
        and not systematic_flags
    )
    status = "PASS" if acceptance_pass else "FAIL" if all_required_complete else "WAITING_FOR_HUMAN_ANNOTATION"
    return {
        "analysis_version": "phase3_5B1a_parser_qa_v1",
        "status": status,
        "total_rows": total,
        "completed_rows": completed,
        "incomplete_rows": total - completed,
        "manual_field_errors": manual_errors,
        "group_counts": {name: len(group) for name, group in groups.items()},
        "group_metrics": fields_by_group,
        "resolved_precision": resolved_precision,
        "unresolved_actually_resolvable": unresolved_metric,
        "no_citation_false_negative_control": no_citation_metric,
        "error_reason_counts": dict(Counter(value(row, "error_reason") for row in rows if value(row, "error_reason"))),
        "parser_error_type_counts": dict(error_counter),
        "systematic_failure_flags": systematic_flags,
        "acceptance_rule": "PASS iff all manual fields are complete, resolved precision >= 0.95, and no repeated parser_error_type flag is present.",
        "note": "This report does not fill or infer human fields. Unresolved/uncertain rows require manual actually_resolvable judgments.",
    }


def failure_cases(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for row in rows:
        fields = field_values(row)
        if any(fields[field] == "no" for field in ("extraction_correct", "law_name_correct", "article_number_correct", "resolution_correct")) or fields["actually_resolvable"] == "yes":
            cases.append({
                "audit_unit_id": value(row, "audit_unit_id"),
                "unit_type": value(row, "unit_type"),
                "query_id": value(row, "query_id"),
                "raw_citation": value(row, "raw_citation"),
                **fields,
            })
    return cases


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Parser QA Summary",
        "",
        f"- Status: **{summary['status']}**",
        f"- Rows: {summary['total_rows']} total; {summary['completed_rows']} complete; {summary['incomplete_rows']} incomplete",
        "- Acceptance rule: resolved precision >= 0.95 and no repeated systematic parser failure flag.",
        "",
        "## Current state",
        "",
        summary["note"],
        "",
        "## Group counts",
        "",
    ]
    for name, count in summary["group_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines += [
        "",
        "## Key metrics",
        "",
        f"- Resolved citation precision: `{summary['resolved_precision']['accuracy']}` (observed n={summary['resolved_precision']['observed_n']})",
        f"- Originally unresolved rows marked actually resolvable: `{summary['unresolved_actually_resolvable']['accuracy']}` (observed n={summary['unresolved_actually_resolvable']['observed_n']})",
        f"- No-citation controls marked as parser false negatives: `{summary['no_citation_false_negative_control']['no']}`" if summary["no_citation_false_negative_control"]["observed_n"] else "- No-citation control false-negative rate: **not assessed; manual fields are blank**",
        "",
        "## Systematic-failure flags",
        "",
    ]
    if summary["systematic_failure_flags"]:
        lines.extend(f"- `{item['parser_error_type']}`: {item['count']}" for item in summary["systematic_failure_flags"])
    else:
        lines.append("- None recorded; this remains unassessed while manual fields are blank.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_PHASE_DIR / "parser_qa.xlsx")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PHASE_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workbook = load_workbook_sheet(args.input, "parser_qa")
    rows = workbook["rows"]
    summary = evaluate(rows)
    summary["input_file"] = str(args.input)
    summary["input_sha256"] = sha256(args.input)
    write_json(args.output_dir / "parser_qa_summary.json", summary)
    (args.output_dir / "parser_qa_summary_zh.md").write_text(markdown_report(summary), encoding="utf-8")
    if summary["status"] == "FAIL":
        write_csv(
            args.output_dir / "parser_qa_failure_cases.csv",
            failure_cases(rows),
            ["audit_unit_id", "unit_type", "query_id", "raw_citation", "extraction_correct", "law_name_correct", "article_number_correct", "resolution_correct", "actually_resolvable", "parser_error_type", "notes"],
        )
    print(json.dumps({key: summary[key] for key in ("status", "total_rows", "completed_rows", "incomplete_rows", "resolved_precision", "systematic_failure_flags")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
