"""Evaluate a model-assisted Parser QA overlay without treating it as human QA.

The uploaded workbook is an annotation overlay keyed by ``audit_unit_id``.  It
is intentionally evaluated in a separate output directory and is never merged
into or used to overwrite the blank human-review workbook.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from evaluate_parser_qa import (
    annotation_complete,
    evaluate,
    field_values,
    load_workbook_sheet,
    value,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "reports/phase3_5/parser_qa_data.json"
DEFAULT_OUTPUT = ROOT / "reports/phase3_5/model_assisted_parser_qa"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_annotation_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if value(row, "audit_unit_id").isdigit()]


def load_sample(path: Path) -> Dict[int, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["audit_unit_id"]): row for row in payload["rows"]}


def overlay_integrity(
    all_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    sample: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    ids = [int(value(row, "audit_unit_id")) for row in rows]
    duplicate_ids = sorted({row_id for row_id in ids if ids.count(row_id) > 1})
    observed = set(ids)
    expected = set(sample)
    return {
        "sheet_rows_total": len(all_rows),
        "numeric_annotation_rows": len(rows),
        "non_sample_or_blank_rows": len(all_rows) - len(rows),
        "annotation_id_min": min(ids) if ids else None,
        "annotation_id_max": max(ids) if ids else None,
        "unique_annotation_ids": len(observed),
        "duplicate_ids": duplicate_ids,
        "missing_expected_ids": sorted(expected - observed),
        "unexpected_ids": sorted(observed - expected),
        "sample_id_set_exact": observed == expected and not duplicate_ids,
        "review_source_counts": _counts(rows, "review_source"),
        "notes_contain_model_assisted_disclaimer": sum(
            (
                "model-assisted" in value(row, "notes").lower()
                or "模型辅助" in value(row, "notes")
            )
            for row in rows
        ),
    }


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        current = value(row, field) or "<blank>"
        counts[current] = counts.get(current, 0) + 1
    return counts


def priority_review_rows(
    rows: Sequence[Mapping[str, Any]], sample: Mapping[int, Mapping[str, Any]]
) -> list[Dict[str, Any]]:
    output: list[Dict[str, Any]] = []
    for row in rows:
        row_id = int(value(row, "audit_unit_id"))
        source = sample.get(row_id, {})
        fields = field_values(row)
        if value(row, "unit_type") != "unresolved_or_uncertain_citation":
            continue
        output.append(
            {
                "priority": "high" if fields["parser_error_type"] else "medium",
                "audit_unit_id": row_id,
                "query_id": source.get("query_id", ""),
                "model": source.get("model", ""),
                "method": source.get("method", ""),
                "raw_citation": source.get("raw_citation", ""),
                "parser_status": source.get("parser_status", ""),
                "model_assisted_extraction_correct": fields["extraction_correct"],
                "model_assisted_law_name_correct": fields["law_name_correct"],
                "model_assisted_article_number_correct": fields["article_number_correct"],
                "model_assisted_resolution_correct": fields["resolution_correct"],
                "model_assisted_actually_resolvable": fields["actually_resolvable"],
                "model_assisted_parser_error_type": fields["parser_error_type"],
                "model_assisted_short_reason": value(row, "short_reason"),
                "human_validation_required": "yes",
            }
        )
    return output


def make_report(
    summary: Mapping[str, Any],
    integrity: Mapping[str, Any],
    input_path: Path,
    sample_path: Path,
) -> str:
    lines = [
        "# Parser QA — Model-Assisted Overlay Evaluation",
        "",
        "> This is a model-assisted annotation report, not an independent human QA result.",
        "",
        f"- Label source: `{summary['label_source']}`",
        f"- Overlay status: **{summary['status']}**",
        f"- Human acceptance preview: **{summary['human_acceptance_preview']}**",
        f"- Input: `{input_path}`",
        f"- Reference sample: `{sample_path}`",
        "",
        "## Structural checks",
        "",
        f"- Numeric annotation rows: `{integrity['numeric_annotation_rows']}`",
        f"- Extra/blank rows ignored: `{integrity['non_sample_or_blank_rows']}`",
        f"- Exact audit-unit coverage 1–100: `{integrity['sample_id_set_exact']}`",
        f"- Duplicate IDs: `{integrity['duplicate_ids']}`",
        f"- Missing expected IDs: `{integrity['missing_expected_ids']}`",
        f"- Model-assisted disclaimer rows: `{integrity['notes_contain_model_assisted_disclaimer']}`",
        "",
        "## Recomputed metrics",
        "",
        f"- Resolved-citation precision: `{summary['resolved_precision']['accuracy']}` (n={summary['resolved_precision']['observed_n']})",
        f"- Unresolved rows judged actually resolvable: `{summary['unresolved_actually_resolvable']['accuracy']}` (excluding uncertain; n={summary['unresolved_actually_resolvable']['observed_n']})",
        f"- No-citation control extraction accuracy: `{summary['no_citation_false_negative_control']['accuracy']}` (n={summary['no_citation_false_negative_control']['observed_n']})",
        "",
        "## Repeated parser failure signals",
        "",
    ]
    if summary["systematic_failure_flags"]:
        lines.extend(
            f"- `{item['parser_error_type']}`: {item['count']} rows"
            for item in summary["systematic_failure_flags"]
        )
    else:
        lines.append("- None in this overlay.")
    lines += [
        "",
        "## Decision",
        "",
        "The overlay is complete enough to prioritize human checking, but it cannot be reported as human Parser QA. The repeated failure types mean the model-assisted evidence should be independently validated before Phase 3.5B-1 is accepted. The original blank human-review workbook remains unchanged.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sheet", default="parser_qa_annotations")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    workbook = load_workbook_sheet(args.input, args.sheet)
    all_rows = workbook["rows"]
    rows = numeric_annotation_rows(all_rows)
    sample = load_sample(args.sample)
    base = evaluate(rows)
    integrity = overlay_integrity(all_rows, rows, sample)
    source_counts = integrity["review_source_counts"]
    source_is_model_assisted = (
        len(source_counts) == 1
        and next(iter(source_counts), "").lower().find("model-assisted") >= 0
    )
    structurally_valid = bool(integrity["sample_id_set_exact"] and source_is_model_assisted)
    human_preview = (
        "BLOCKED_STRUCTURE"
        if not structurally_valid
        else "FAIL_SYSTEMATIC_FAILURES"
        if base["systematic_failure_flags"]
        else "CANDIDATE_PASS_PENDING_HUMAN_VALIDATION"
    )
    summary = dict(base)
    summary.update(
        {
            "analysis_version": "phase3_5B1a_parser_qa_model_assisted_v1",
            "status": "MODEL_ASSISTED_ONLY",
            "label_source": "ChatGPT model-assisted review",
            "human_validation_required": True,
            "human_acceptance_preview": human_preview,
            "structural_valid": structurally_valid,
            "input_file": str(args.input),
            "input_sha256": sha256(args.input),
            "sheet_name": args.sheet,
            "reference_sample": str(args.sample),
            "overlay_integrity": integrity,
            "next_gate": "Independent human validation of Parser QA before formal B-1 acceptance.",
        }
    )
    write_json(args.output_dir / "parser_qa_model_assisted_summary.json", summary)
    (args.output_dir / "parser_qa_model_assisted_summary_zh.md").write_text(
        make_report(summary, integrity, args.input, args.sample), encoding="utf-8"
    )
    queue = priority_review_rows(rows, sample)
    write_csv(
        args.output_dir / "parser_qa_model_assisted_priority_review.csv",
        queue,
        list(queue[0].keys()) if queue else ["audit_unit_id"],
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "human_acceptance_preview": summary["human_acceptance_preview"],
                "numeric_annotation_rows": integrity["numeric_annotation_rows"],
                "resolved_precision": summary["resolved_precision"],
                "unresolved_actually_resolvable": summary["unresolved_actually_resolvable"],
                "no_citation_false_negative_control": summary["no_citation_false_negative_control"],
                "systematic_failure_flags": summary["systematic_failure_flags"],
                "priority_review_rows": len(queue),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
