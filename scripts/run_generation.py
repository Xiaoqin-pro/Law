"""Run resumable Phase 3 Direct/BM25/Dense/Hybrid generation baselines."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.generation import (  # noqa: E402
    LocalQwenGenerator,
    build_evidence,
    classify_runtime_error,
    load_generation_records,
    load_processed_queries,
    load_jsonl,
    render_prompt,
    write_generation_record,
)


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "generation_phase3.json")
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "generation")
    parser.add_argument("--model-key", choices=("qwen25_7b", "qwen3_4b"), required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local model directory; when set, no remote model lookup is used.",
    )
    parser.add_argument("--methods", nargs="+", choices=("direct", "bm25", "dense", "hybrid"), default=["direct", "bm25", "dense", "hybrid"])
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--retrieval-root", type=Path, default=PROJECT_ROOT / "outputs" / "retrieval")
    parser.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "outputs" / "generation" / "cache" / "model")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    random.seed(seed)

    queries = load_processed_queries(args.processed_root / f"{config['split']}.jsonl")
    query_rows = list(queries.values())
    if args.mode == "smoke":
        query_rows = random.Random(seed).sample(query_rows, min(10, len(query_rows)))
    corpus = {int(row["statute_id"]): row for row in load_jsonl(args.processed_root / "corpus.jsonl")}
    model_config = config["models"][args.model_key]
    model_name = str(args.model_path.resolve()) if args.model_path else model_config["model_name"]
    model = LocalQwenGenerator(
        model_name,
        cache_dir=args.model_cache,
        max_input_tokens=config["max_input_tokens"],
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
    )

    retrieval_rows: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for method in ("bm25", "dense", "hybrid"):
        if method in args.methods:
            path = PROJECT_ROOT / config["retrieval_inputs"][method]
            retrieval_rows[method] = {int(row["query_id"]): row for row in load_jsonl(path)}

    args.output_root.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        run_id = f"phase3_{args.mode}_{args.model_key}_{method}"
        output_path = args.output_root / f"{run_id}.jsonl"
        metadata_path = args.output_root / f"{run_id}.metadata.json"
        completed = load_generation_records(output_path) if not args.no_resume else {}
        metadata = {
            "experiment_id": run_id,
            "git_commit": current_git_commit(),
            "config_path": str(args.config),
            "config": config,
            "random_seed": seed,
            "model_key": args.model_key,
            "model_name": model_name,
            "model_source": "local_path" if args.model_path else "huggingface",
            "model_revision": model.model_revision,
            "dataset_split": config["split"],
            "prompt_version": config["prompt_version"],
            "method": method,
            "query_count": len(query_rows),
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python": sys.version,
            "platform": platform.platform(),
            "cpu": platform.processor(),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        for index, query in enumerate(query_rows, start=1):
            query_id = int(query["query_id"])
            if query_id in completed:
                continue
            evidence = ""
            retrieval_row = None
            if method in retrieval_rows:
                retrieval_row = retrieval_rows[method][query_id]
                evidence = build_evidence(retrieval_row, corpus, top_k=config["top_k"])
            prompt = render_prompt(method, query, prompt_templates=config["prompt"], evidence=evidence)
            started = time.perf_counter()
            try:
                answer = model.generate(prompt)
                status = "ok"
                error = None
            except Exception as exc:  # record one failed item and continue
                answer = ""
                status = "error"
                error = {"type": classify_runtime_error(exc), "message": str(exc)}
            record = {
                "query_id": query_id,
                "question": query["question"],
                "method": method,
                "model_key": args.model_key,
                "prompt_version": config["prompt_version"],
                "prompt": prompt,
                "answer": answer,
                "status": status,
                "error": error,
                "gold_statute_ids": query["gold_statute_ids"],
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 4),
            }
            if retrieval_row is not None:
                record["retrieved_statute_ids"] = [int(r["statute_id"]) for r in retrieval_row["results"]]
            write_generation_record(output_path, record)
            if index == 1 or index % 10 == 0 or index == len(query_rows):
                print(f"{run_id}: {index}/{len(query_rows)}")
        metadata["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        metadata["completed_count"] = len(load_generation_records(output_path))
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
