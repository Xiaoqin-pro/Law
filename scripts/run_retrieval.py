"""Run Phase 2 BM25, Dense, and Hybrid retrieval with caches and resume."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.data import load_jsonl, write_json, write_jsonl  # noqa: E402
from citelaw.retrieval import (  # noqa: E402
    BM25Index,
    DenseEmbedder,
    DenseIndex,
    corpus_fingerprint,
    error_analysis,
    now_timestamp,
    query_fingerprint,
    reciprocal_rank_fusion,
    retrieval_metrics,
)


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl_incremental(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_existing_results(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["query_id"])] = row
    return rows


def run_retriever(
    name: str,
    queries: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    output_path: Path,
    search_fn: Any,
    *,
    top_k: int,
    resume: bool,
) -> List[Dict[str, Any]]:
    existing = load_existing_results(output_path) if resume else {}
    rows: List[Dict[str, Any]] = []
    for index, query in enumerate(queries, start=1):
        query_id = int(query["query_id"])
        if query_id in existing:
            rows.append(existing[query_id])
            continue
        started = time.perf_counter()
        results = search_fn(query, top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        row = {
            "query_id": query_id,
            "question": str(query["question"]),
            "gold_statute_ids": [int(value) for value in query["gold_statute_ids"]],
            "retriever": name,
            "results": results,
            "latency_ms": round(elapsed_ms, 4),
        }
        rows.append(row)
        write_jsonl_incremental(output_path, rows)
        if index == 1 or index % 25 == 0 or index == len(queries):
            print(f"{name}: {index}/{len(queries)}")
    rows.sort(key=lambda row: int(row["query_id"]))
    write_jsonl_incremental(output_path, rows)
    return rows


def save_metrics(path: Path, result_by_retriever: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["retriever", "query_count", "recall_at_1", "recall_at_5", "recall_at_10", "mrr", "avg_latency_ms"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, rows in result_by_retriever.items():
            metrics = retrieval_metrics(rows)
            latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
            writer.writerow({
                "retriever": name,
                "query_count": len(rows),
                **metrics,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
            })


def smoke_sample(queries: Sequence[Mapping[str, Any]], count: int, seed: int) -> List[Mapping[str, Any]]:
    if len(queries) <= count:
        return list(queries)
    return random.Random(seed).sample(list(queries), count)


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "retrieval_phase2.json")
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "retrieval")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "outputs" / "retrieval" / "cache")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--retrievers", nargs="+", choices=("bm25", "dense", "hybrid"), default=["bm25", "dense", "hybrid"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    np.random.seed(seed)
    random.seed(seed)

    corpus = load_jsonl(args.processed_root / "corpus.jsonl")
    all_queries = load_jsonl(args.processed_root / f"{config['split']}.jsonl")
    queries = smoke_sample(all_queries, 10, seed) if args.mode == "smoke" else all_queries
    run_id = f"phase2_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_id": run_id,
        "git_commit": current_git_commit(),
        "config_path": str(args.config),
        "config": config,
        "random_seed": seed,
        "dataset_split": config["split"],
        "query_count": len(queries),
        "query_fingerprint": query_fingerprint(queries),
        "corpus_fingerprint": corpus_fingerprint(corpus),
        "timestamp_start": now_timestamp(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "retrievers": args.retrievers,
    }
    write_json(args.output_root / f"{run_id}.metadata.json", metadata)

    bm25: BM25Index | None = None
    dense: DenseIndex | None = None
    embedder: DenseEmbedder | None = None
    dense_manifest: Dict[str, Any] = {}
    if "bm25" in args.retrievers or "hybrid" in args.retrievers:
        bm25_cache = args.cache_root / f"bm25_{corpus_fingerprint(corpus)[:12]}_{config['bm25']['k1']}_{config['bm25']['b']}.pkl"
        if bm25_cache.exists() and not args.no_resume:
            bm25 = BM25Index.load(bm25_cache)
            print(f"BM25 cache hit: {bm25_cache}")
        else:
            started = time.perf_counter()
            bm25 = BM25Index(k1=config["bm25"]["k1"], b=config["bm25"]["b"]).build(corpus)
            bm25.save(bm25_cache)
            print(f"BM25 index built in {time.perf_counter() - started:.2f}s")

    if "dense" in args.retrievers or "hybrid" in args.retrievers:
        dense_config = config["dense"]
        model_cache = args.cache_root / "model"
        embedder = DenseEmbedder(
            dense_config["model_name"],
            max_length=dense_config["max_length"],
            batch_size=dense_config["batch_size"],
            device=args.device,
            cache_dir=model_cache,
            normalize_embeddings=dense_config["normalize_embeddings"],
        )
        vectors, dense_manifest = embedder.encode_corpus_cached(corpus, args.cache_root / "embeddings", batch_size=dense_config["batch_size"])
        dense_index_path = args.cache_root / f"dense_{dense_manifest['corpus_fingerprint'][:12]}_{dense_manifest['model_revision']}.faiss"
        if dense_index_path.exists() and not args.no_resume:
            dense = DenseIndex.load(dense_index_path, corpus)
            print(f"Dense FAISS cache hit: {dense_index_path}")
        else:
            started = time.perf_counter()
            dense = DenseIndex(vectors, corpus)
            dense.save(dense_index_path)
            print(f"Dense FAISS index built in {time.perf_counter() - started:.2f}s")

    result_by_retriever: Dict[str, List[Dict[str, Any]]] = {}
    if bm25 is not None and "bm25" in args.retrievers:
        path = args.output_root / f"{run_id}.bm25.jsonl"
        result_by_retriever["bm25"] = run_retriever(
            "bm25",
            queries,
            corpus,
            path,
            lambda query, top_k: bm25.search(str(query["question"]), top_k),
            top_k=config["top_k"],
            resume=not args.no_resume,
        )
    dense_by_query: Dict[int, np.ndarray] = {}
    if dense is not None and embedder is not None and ("dense" in args.retrievers or "hybrid" in args.retrievers):
        dense_vectors = embedder.encode([str(query["question"]) for query in queries], batch_size=dense_config["batch_size"])
        dense_by_query = {int(query["query_id"]): vector for query, vector in zip(queries, dense_vectors)}
    if dense is not None and "dense" in args.retrievers:
        result_by_retriever["dense"] = run_retriever(
            "dense",
            queries,
            corpus,
            args.output_root / f"{run_id}.dense.jsonl",
            lambda query, top_k: dense.search(dense_by_query[int(query["query_id"])], top_k),
            top_k=config["top_k"],
            resume=not args.no_resume,
        )
    if dense is not None and embedder is not None and "hybrid" in args.retrievers:
        hybrid_rows: List[Dict[str, Any]] = []
        output_path = args.output_root / f"{run_id}.hybrid.jsonl"
        existing = load_existing_results(output_path) if not args.no_resume else {}
        for index, query in enumerate(queries, start=1):
            query_id = int(query["query_id"])
            if query_id in existing:
                hybrid_rows.append(existing[query_id])
                continue
            started = time.perf_counter()
            bm25_top = bm25.search(str(query["question"]), config["fusion_pool_k"]) if bm25 else []
            dense_top = dense.search(dense_by_query[query_id], config["fusion_pool_k"])
            fused = reciprocal_rank_fusion([bm25_top, dense_top], rrf_k=config["rrf_k"], top_k=config["top_k"])
            hybrid_rows.append({
                "query_id": query_id,
                "question": str(query["question"]),
                "gold_statute_ids": [int(value) for value in query["gold_statute_ids"]],
                "retriever": "hybrid",
                "results": fused,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 4),
            })
            write_jsonl_incremental(output_path, hybrid_rows)
            if index == 1 or index % 25 == 0 or index == len(queries):
                print(f"hybrid: {index}/{len(queries)}")
        hybrid_rows.sort(key=lambda row: int(row["query_id"]))
        write_jsonl_incremental(output_path, hybrid_rows)
        result_by_retriever["hybrid"] = hybrid_rows

    save_metrics(args.output_root / f"{run_id}.retrieval_results.csv", result_by_retriever)
    if set(result_by_retriever) >= {"bm25", "dense", "hybrid"}:
        analysis = error_analysis(result_by_retriever)
        write_json(args.output_root / f"{run_id}.error_analysis.json", {key: value[:50] for key, value in analysis.items()})
    smoke_rows = []
    for query in queries:
        payload: Dict[str, Any] = {"query_id": query["query_id"], "question": query["question"], "gold_statute_ids": query["gold_statute_ids"]}
        for name, rows in result_by_retriever.items():
            match = next((row for row in rows if int(row["query_id"]) == int(query["query_id"])), None)
            payload[name] = {"top5": match["results"][:5] if match else []}
        smoke_rows.append(payload)
    write_json(args.output_root / f"{run_id}.smoke_readable.json", smoke_rows)
    metadata["timestamp_end"] = now_timestamp()
    metadata["dense_manifest"] = dense_manifest
    metadata["metrics"] = {name: retrieval_metrics(rows) for name, rows in result_by_retriever.items()}
    write_json(args.output_root / f"{run_id}.metadata.json", metadata)
    print(json.dumps(metadata["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
