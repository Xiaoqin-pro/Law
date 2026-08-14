"""Run the non-destructive Phase 3.5B-1 validation checks in order."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE_DIR = ROOT / "reports/phase3_5"


def run(script: str) -> None:
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    # The independent precheck is only a convenience aid. Parser QA must still
    # be checked before the semantic workbook. None of these commands writes
    # manual annotation cells.
    run("parser_qa_independent_precheck.py")
    run("evaluate_parser_qa.py")
    run("validate_manual_audit.py")
    precheck = json.loads((PHASE_DIR / "parser_qa_independent_precheck.json").read_text(encoding="utf-8"))
    parser_summary = json.loads((PHASE_DIR / "parser_qa_summary.json").read_text(encoding="utf-8"))
    manual_progress = json.loads((PHASE_DIR / "manual_annotation_progress.json").read_text(encoding="utf-8"))
    manifest = {
        "analysis_version": "phase3_5B1_validation_infrastructure_v1",
        "frozen_commit": "b3679ade8b233b71e5d3812dd44184aabee05f8f",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "3.5B-1 Human Validation",
        "parser_qa": {
            "status": parser_summary["status"],
            "total_rows": parser_summary["total_rows"],
            "completed_rows": parser_summary["completed_rows"],
            "summary_json": "parser_qa_summary.json",
            "summary_report": "parser_qa_summary_zh.md",
            "independent_precheck": {
                "status": precheck["summary"]["status"],
                "resolved_mapping_yes": precheck["summary"]["resolved_mapping_yes"],
                "warning": "precheck only; not human validation or acceptance",
            },
        },
        "manual_semantic_review": {
            "status": manual_progress["status"],
            "total_rows": manual_progress["total_rows"],
            "completed_rows": manual_progress["completed_rows"],
            "progress_json": "manual_annotation_progress.json",
        },
        "scripts": [
            "scripts/workbook_to_json.mjs",
            "scripts/parser_qa_independent_precheck.py",
            "scripts/evaluate_parser_qa.py",
            "scripts/validate_manual_audit.py",
            "scripts/run_phase3_5b1_validation.py",
        ],
        "inputs": [
            "parser_qa.xlsx",
            "manual_audit.xlsx",
            "manual_audit_data.json",
            "sampling_manifest.json",
        ],
        "outputs": [
            "parser_qa_summary.json",
            "parser_qa_summary_zh.md",
            "parser_qa_independent_precheck.json",
            "parser_qa_independent_precheck.md",
            "manual_annotation_progress.json",
            "phase3_5B1_checkpoint_report.md",
        ],
        "stop_rule": "WAITING_FOR_HUMAN_ANNOTATION; do not calculate final semantic prevalence, merge PR #1, or enter Phase 4.",
    }
    (PHASE_DIR / "phase3_5B1_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "WAITING_FOR_HUMAN_ANNOTATION", "parser": parser_summary["status"], "manual": manual_progress["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
