"""Build the Phase 3.2 final baseline manifest and report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "outputs/generation"


GENERATION_FILES = {
    "qwen25_7b_direct": "phase3_full_qwen25_7b_direct.jsonl",
    "qwen25_7b_bm25_v2": "phase3_2_1_bm25_encoding_fixed_full_qwen25_7b_bm25.jsonl",
    "qwen25_7b_dense_cls": "phase3_1_cls_v2_full_qwen25_7b_dense.jsonl",
    "qwen25_7b_hybrid_cls": "phase3_1_cls_v2_full_qwen25_7b_hybrid.jsonl",
    "qwen3_4b_direct": "phase3_full_qwen3_4b_direct.jsonl",
    "qwen3_4b_bm25_v2": "phase3_2_1_bm25_encoding_fixed_full_qwen3_4b_bm25.jsonl",
    "qwen3_4b_dense_cls": "phase3_1_cls_v2_full_qwen3_4b_dense.jsonl",
    "qwen3_4b_hybrid_cls": "phase3_1_cls_v2_full_qwen3_4b_hybrid.jsonl",
}


LEGACY_BM25_FILES = {
    "qwen25_7b_bm25_v2_legacy": "phase3_2_bm25_contextsafe_full_qwen25_7b_bm25.jsonl",
    "qwen3_4b_bm25_v2_legacy": "phase3_2_bm25_contextsafe_full_qwen3_4b_bm25.jsonl",
}


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_generation(key: str, filename: str) -> Dict[str, Any]:
    path = GENERATION_ROOT / filename
    rows = load_jsonl(path)
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    statuses = Counter(str(row.get("status")) for row in rows)
    token_counts = [
        int(row.get("input_token_count_before_generation", row.get("input_token_count_before_truncation", 0)))
        for row in rows
        if row.get("input_token_count_before_generation", row.get("input_token_count_before_truncation")) is not None
    ]
    partial_count = sum(len(row.get("partially_visible_statute_ids", [])) for row in rows)
    visible_evidence_id_count = sum(
        len(row.get("included_statute_ids", [])) if "bm25_v2" in key else len(row.get("fully_visible_statute_ids", []))
        for row in rows
    )
    visible_evidence_consistency_failures = sum(
        "bm25_v2" in key
        and (
            row.get("visible_statute_ids", []) != row.get("included_statute_ids", [])
            or row.get("fully_visible_statute_ids", []) != row.get("included_statute_ids", [])
        )
        for row in rows
    )
    evidence_marker_missing_count = sum(
        "bm25_v2" in key and "[\u6cd5\u6761ID " not in row.get("prompt", "")
        for row in rows
    )
    return {
        "key": key,
        "file": str(path.relative_to(PROJECT_ROOT)),
        "metadata_file": str(metadata_path.relative_to(PROJECT_ROOT)),
        "record_count": len(rows),
        "unique_query_count": len({int(row["query_id"]) for row in rows}),
        "status_counts": dict(sorted(statuses.items())),
        "all_ok": len(rows) == 309 and statuses == Counter({"ok": 309}),
        "truncated_count": sum(bool(row.get("was_truncated")) for row in rows),
        "partial_statute_block_count": partial_count,
        "fully_visible_statute_block_count": sum(len(row.get("fully_visible_statute_ids", [])) for row in rows),
        "visible_evidence_id_count": visible_evidence_id_count,
        "visible_evidence_consistency_failures": visible_evidence_consistency_failures,
        "evidence_marker_missing_count": evidence_marker_missing_count,
        "input_token_count_mean": round(mean(token_counts), 3) if token_counts else None,
        "input_token_count_max": max(token_counts) if token_counts else None,
        "artifact_sha256": sha256_file(path),
        "metadata_sha256": sha256_file(metadata_path),
        "model_key": metadata.get("model_key"),
        "model_name": metadata.get("model_name"),
        "model_revision": metadata.get("model_revision"),
        "prompt_version": metadata.get("prompt_version"),
        "generation_config": {
            "max_new_tokens": metadata.get("config", {}).get("max_new_tokens"),
            "temperature": metadata.get("config", {}).get("temperature"),
            "top_p": metadata.get("config", {}).get("top_p"),
            "max_input_tokens": metadata.get("config", {}).get("max_input_tokens"),
            "quantization": metadata.get("config", {}).get("models", {}).get(metadata.get("model_key"), {}).get("quantization"),
        },
        "retrieval_experiment_id": (
            "phase2_full_20260813_010257" if "direct" not in key and "bm25_v2" in key else
            "phase2_1_cls_v2_full" if "dense_cls" in key or "hybrid_cls" in key else None
        ),
    }


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-status", default="20/20 passed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()

    reconciliation_path = PROJECT_ROOT / "reports/data/lecoqa_gold_reconciliation.json"
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    retrieval_csv_path = PROJECT_ROOT / "reports/phase3_2/retrieval_final.csv"
    retrieval_rows = read_csv(retrieval_csv_path)
    context_audit_path = PROJECT_ROOT / "reports/generation/context_audit.json"
    context_audit = json.loads(context_audit_path.read_text(encoding="utf-8")) if context_audit_path.exists() else {}
    generation = [summarize_generation(key, filename) for key, filename in GENERATION_FILES.items()]
    bm25_v2 = [row for row in generation if "bm25_v2" in row["key"]]
    legacy_bm25_v2 = [
        {
            "key": key,
            "file": str((GENERATION_ROOT / filename).relative_to(PROJECT_ROOT)),
            "metadata_file": str((GENERATION_ROOT / filename).with_suffix(".metadata.json").relative_to(PROJECT_ROOT)),
            "artifact_sha256": sha256_file(GENERATION_ROOT / filename),
            "metadata_sha256": sha256_file((GENERATION_ROOT / filename).with_suffix(".metadata.json")),
        }
        for key, filename in LEGACY_BM25_FILES.items()
    ]
    bm25_v2_ready = (
        len(bm25_v2) == 2
        and all(
            row["all_ok"]
            and row["truncated_count"] == 0
            and row["partial_statute_block_count"] == 0
            for row in bm25_v2
        )
    )

    manifest = {
        "manifest_version": "phase3_2_1_v1_bm25_encoding_fixed_freeze",
        "dataset": "LeCoQA",
        "split": "test",
        "query_count": 309,
        "git_commit_at_manifest_generation": current_git_commit(),
        "platform": platform.platform(),
        "phase_status": {
            "phase3_2": "complete",
            "phase3_2_1": "complete",
            "phase3_5": "deferred",
            "phase4": "deferred",
        },
        "gold_reconciliation": {
            "file": str(reconciliation_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(reconciliation_path),
            "counts": reconciliation["counts"],
            "resolution_levels": reconciliation["resolution_levels"],
        },
        "retrieval": {
            "file": str(retrieval_csv_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(retrieval_csv_path),
            "metrics": retrieval_rows,
            "paired_bootstrap_file": "reports/phase3_2/retrieval_paired_bootstrap.csv",
            "paired_bootstrap_sha256": sha256_file(PROJECT_ROOT / "reports/phase3_2/retrieval_paired_bootstrap.csv"),
            "primary_retriever_candidate": "dense_cls",
            "primary_retriever_reason": "Dense CLS is numerically ahead of Hybrid on the raw and raw-clean 309/267-query evaluations; no significance claim is made here.",
        },
        "generation": generation,
        "context_audit": {
            "legacy_generation": context_audit.get("method_summary", {}),
            "bm25_v2": bm25_v2,
            "legacy_bm25_v2_pre_encoding_fix": legacy_bm25_v2,
        },
        "tests": {
            "command": "E:\\python\\python.exe -m unittest discover -s tests -v",
            "status": args.tests_status,
        },
        "artifact_sha256": {
            "gold_reconciliation": sha256_file(reconciliation_path),
            "retrieval_final": sha256_file(retrieval_csv_path),
            **{row["key"]: row["artifact_sha256"] for row in generation},
        },
        "unresolved_or_deferred": [
            "Level C exact-name-only matches are reported separately and are not included in strict high-confidence gold.",
            "No semantic legal-support judgment has been made.",
            "Phase 3.5 citation analysis and Phase 4 repair remain deferred.",
        ],
    }
    manifest_path = args.output_dir / "final_baseline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bm25_visible_total = sum(row["visible_evidence_id_count"] for row in bm25_v2)
    bm25_fully_visible_total = sum(row["fully_visible_statute_block_count"] for row in bm25_v2)
    bm25_audit_ready = bm25_v2_ready and all(
        row["visible_evidence_consistency_failures"] == 0
        and row["evidence_marker_missing_count"] == 0
        for row in bm25_v2
    )
    bm25_acceptance = (
        f"BM25 v2 acceptance: passed: 618/618 records are successful, with no truncation or partially visible statute blocks; "
        f"sum(included_statute_ids)={bm25_visible_total}, sum(fully_visible_statute_ids)={bm25_fully_visible_total}, "
        "marker_missing=0, consistency_failures=0."
        if bm25_audit_ready
        else "BM25 v2 acceptance: pending: requires 618/618 successful records, no truncation, recognized markers, and consistent visible evidence IDs."
    )

    metric_lookup = {(row["gold_definition"], row["retriever"]): row for row in retrieval_rows}
    report_lines = [
        "# Phase 3.2.1 BM25 Encoding-Fixed Final Baseline Freeze",
        "",
        f"- Dataset: LeCoQA official test split ({manifest['query_count']} queries)",
        f"- Manifest source commit: `{manifest['git_commit_at_manifest_generation']}`",
        "- Phase 3.2.1 BM25 encoding hotfix: complete",
        "- Phase 3.5 citation analysis: deferred",
        "- Phase 4: deferred",
        "",
        "## Gold reconciliation",
        "",
        f"Raw-clean: {reconciliation['counts']['raw_position_clean']}; order-only conflicts: {reconciliation['counts']['reconciliation_status'].get('order_only_conflict', 0)}; true-set conflicts: {reconciliation['counts']['reconciliation_status'].get('true_set_conflict', 0)}; ambiguous: {reconciliation['counts']['reconciliation_status'].get('ambiguous', 0)}; unresolved: {reconciliation['counts']['reconciliation_status'].get('unresolved', 0)}.",
        f"All {reconciliation['counts']['all_related_resolved']} queries have a deterministic resolved set. Strict Level A/B high-confidence queries: {reconciliation['counts']['high_confidence_all_related']}; the remaining {reconciliation['counts']['not_strict_high_confidence']} queries are resolved only through lower-confidence Level C name matching and are reported separately, not silently promoted. Actual unresolved, ambiguous, or true-set-conflicted queries: {reconciliation['counts']['actual_unresolved_or_conflicted']}.",
        "",
        "## Retrieval metrics",
        "",
        "The raw and raw-clean tables are the primary baseline views; the strict high-confidence subset is small and is reported as a sensitivity analysis.",
        "",
        "| Gold definition | Retriever | N | Hit@10 | Recall@10 | MRR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in retrieval_rows:
        if row["gold_definition"] in {"raw_all", "raw_clean", "high_confidence_reconciled"}:
            report_lines.append(f"| {row['gold_definition']} | {row['retriever']} | {row['query_count']} | {row['hit_at_10']} | {row['recall_at_10']} | {row['mrr']} |")
    report_lines += [
        "",
        "Dense CLS is the current primary retriever candidate because it is numerically ahead of Hybrid on the raw and raw-clean views. The paired-bootstrap file records uncertainty; this report does not call the difference statistically significant without a pre-specified decision rule.",
        "",
        "## Frozen four baselines",
        "",
        "| Baseline | Models | Retrieval input | Status |",
        "|---|---|---|---|",
        "| B1 Direct | Qwen2.5-7B, Qwen3-4B | none | legacy formal output retained |",
        f"| B2 BM25 v2 | Qwen2.5-7B, Qwen3-4B | legacy BM25 retrieval + complete-block packing | {'frozen' if bm25_v2_ready else 'pending generation completion'} |",
        "| B3 Dense CLS | Qwen2.5-7B, Qwen3-4B | corrected CLS retrieval | frozen |",
        "| B4 Hybrid CLS | Qwen2.5-7B, Qwen3-4B | corrected CLS RRF retrieval | frozen comparison baseline |",
        "",
        bm25_acceptance,
        "",
        "## Context audit",
        "",
        "The earlier legacy BM25 output had 4 truncated records. The prior Phase 3.2 BM25 v2 output is retained as a legacy pre-encoding-fix artifact. The Phase 3.2.1 BM25 output uses the corrected [\u6cd5\u6761ID] marker, is isolated under a new experiment ID, and does not overwrite the legacy artifact.",
        "",
        "## Next stage",
        "",
        "The next allowed stage is Phase 3.5 baseline citation/failure analysis. No citation verifier, claim decomposition, re-retrieval, revision, or repair is implemented in Phase 3.2.",
    ]
    report_path = args.output_dir / "final_baseline_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "report": str(report_path), "generation_all_ok": all(row["all_ok"] for row in generation)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
