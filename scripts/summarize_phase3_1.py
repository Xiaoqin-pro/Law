"""Summarize corrected retrieval and generation artifacts without changing them."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_generation(path: Path) -> Dict[str, Any]:
    rows = load_jsonl(path)
    statuses = Counter(str(row.get("status")) for row in rows)
    truncated = [row for row in rows if row.get("was_truncated")]
    partial = sum(len(row.get("partially_visible_statute_ids", [])) for row in rows)
    full_blocks = sum(len(row.get("fully_visible_statute_ids", [])) for row in rows)
    token_counts = [int(row["input_token_count_before_truncation"]) for row in rows if "input_token_count_before_truncation" in row]
    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "record_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "truncated_count": len(truncated),
        "truncation_rate": round(len(truncated) / len(rows), 6) if rows else None,
        "partial_statute_block_count": partial,
        "fully_visible_statute_block_count": full_blocks,
        "input_token_count_mean": round(statistics.mean(token_counts), 3) if token_counts else None,
        "input_token_count_max": max(token_counts) if token_counts else None,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
    }


def main() -> int:
    retrieval_metadata_path = PROJECT_ROOT / "outputs/retrieval/phase2_1_cls_v2_full.metadata.json"
    retrieval = json.loads(retrieval_metadata_path.read_text(encoding="utf-8")) if retrieval_metadata_path.exists() else {"status": "pending"}
    generation_root = PROJECT_ROOT / "outputs/generation"
    generation: Dict[str, Any] = {}
    for model_key in ("qwen3_4b", "qwen25_7b"):
        for method in ("dense", "hybrid"):
            path = generation_root / f"phase3_1_cls_v2_full_{model_key}_{method}.jsonl"
            generation[f"{model_key}_{method}"] = summarize_generation(path)
    payload = {
        "summary_version": "phase3_1_v1",
        "retrieval_experiment": retrieval.get("experiment_id"),
        "retrieval_metrics": retrieval.get("metrics", {}),
        "generation": generation,
        "completion_criteria": {
            "retrieval_rows_per_method": 309,
            "generation_rows_per_model_method": 309,
            "required_status": "ok",
            "required_context_fields": [
                "input_token_count_before_truncation",
                "was_truncated",
                "visible_statute_ids",
                "fully_visible_statute_ids",
                "partially_visible_statute_ids",
            ],
        },
    }
    output = PROJECT_ROOT / "reports/phase3_1_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
