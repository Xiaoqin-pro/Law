"""Audit existing generation prompts for model-input truncation and visibility."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.generation import audit_prompt_context, load_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, default=PROJECT_ROOT / "outputs/generation")
    parser.add_argument("--qwen25-model-path", type=Path, default=PROJECT_ROOT / "outputs/generation/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--qwen3-model-path", type=Path, default=PROJECT_ROOT / "outputs/generation/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/generation/context_audit.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer_by_model = {
        "qwen25_7b": AutoTokenizer.from_pretrained(str(args.qwen25_model_path.resolve()), local_files_only=True),
        "qwen3_4b": AutoTokenizer.from_pretrained(str(args.qwen3_model_path.resolve()), local_files_only=True),
    }
    target_files = sorted(
        path for path in args.generation_root.glob("phase3_full_*_*.jsonl")
        if path.stem.rsplit("_", 1)[-1] in {"bm25", "dense", "hybrid"}
    )
    records: List[Dict[str, Any]] = []
    method_summary: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"record_count": 0, "truncated_count": 0, "partial_statute_blocks": 0, "full_statute_blocks": 0})
    for path in target_files:
        for row in load_jsonl(path):
            audit = audit_prompt_context(
                tokenizer_by_model[str(row["model_key"])],
                row["prompt"],
                max_input_tokens=args.max_input_tokens,
            )
            method = str(row.get("method", "unknown"))
            summary = method_summary[method]
            summary["record_count"] += 1
            summary["truncated_count"] += int(audit["was_truncated"])
            summary["partial_statute_blocks"] += len(audit["partially_visible_statute_ids"])
            summary["full_statute_blocks"] += len(audit["fully_visible_statute_ids"])
            records.append({
                "source_file": path.name,
                "query_id": int(row["query_id"]),
                "method": method,
                "model_key": row.get("model_key"),
                **audit,
            })
    payload = {
        "audit_version": "v1_offsets_chat_template",
        "max_input_tokens": args.max_input_tokens,
        "model_paths": {
            "qwen25_7b": str(args.qwen25_model_path.resolve()),
            "qwen3_4b": str(args.qwen3_model_path.resolve()),
        },
        "file_count": len(target_files),
        "record_count": len(records),
        "method_summary": dict(sorted(method_summary.items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"file_count": len(target_files), "record_count": len(records), "method_summary": dict(sorted(method_summary.items()))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
