"""Build a compact, tracked scientific audit report from generated artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-metadata", type=Path, default=PROJECT_ROOT / "outputs/retrieval/phase2_1_cls_v2_full.metadata.json")
    parser.add_argument("--gold-audit", type=Path, default=PROJECT_ROOT / "reports/data/lecoqa_gold_audit.json")
    parser.add_argument("--context-audit", type=Path, default=PROJECT_ROOT / "reports/generation/context_audit.json")
    parser.add_argument("--generation-summary", type=Path, default=PROJECT_ROOT / "reports/phase3_1_summary.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/scientific_audit.json")
    args = parser.parse_args()
    payload: Dict[str, Any] = {
        "report_version": "phase2_1_phase3_1_scientific_audit_v1",
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "scope": {
            "retrieval": "BGE-M3 CLS pooling, normalized dense vectors, Dense/Hybrid rerun",
            "generation": "Dense/Hybrid rerun only after retrieval correction",
            "deferred": ["Phase 3.5 citation evaluator", "Phase 4 training/ablation"],
        },
        "official_embedding_contract": {
            "model": "BAAI/bge-m3",
            "pooling": "normalized last_hidden_state[:, 0] (CLS)",
            "legacy_mean_pooling": "retained only as explicitly named legacy configuration",
        },
        "gold_audit": read_json(args.gold_audit) if args.gold_audit.exists() else {"status": "pending"},
        "context_audit": read_json(args.context_audit) if args.context_audit.exists() else {"status": "pending"},
        "retrieval": read_json(args.retrieval_metadata) if args.retrieval_metadata.exists() else {"status": "pending"},
        "generation_summary": read_json(args.generation_summary) if args.generation_summary.exists() else {"status": "pending"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "gold_audit": payload["gold_audit"].get("classification_counts"),
        "context_audit": payload["context_audit"].get("method_summary"),
        "retrieval_status": payload["retrieval"].get("metrics", "pending"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
