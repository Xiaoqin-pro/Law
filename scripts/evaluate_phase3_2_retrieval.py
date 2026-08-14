"""Re-evaluate existing retrieval JSONL with reconciled gold definitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.retrieval import retrieval_metrics  # noqa: E402


METHOD_PATHS = {
    "bm25": PROJECT_ROOT / "outputs/retrieval/phase2_full_20260813_010257.bm25.jsonl",
    "dense_cls": PROJECT_ROOT / "outputs/retrieval/phase2_1_cls_v2_full.dense.jsonl",
    "hybrid_cls": PROJECT_ROOT / "outputs/retrieval/phase2_1_cls_v2_full.hybrid.jsonl",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_with_gold(row: Mapping[str, Any], gold_ids: Sequence[int]) -> Dict[str, Any]:
    value = dict(row)
    value["gold_statute_ids"] = [int(item) for item in gold_ids]
    return value


def query_metric_values(row: Mapping[str, Any], gold_ids: Sequence[int]) -> Dict[str, float]:
    gold = {int(value) for value in gold_ids}
    ranked = [int(result["statute_id"]) for result in row["results"]]
    values: Dict[str, float] = {}
    for cutoff in (5, 10):
        retrieved = set(ranked[:cutoff])
        values[f"hit_at_{cutoff}"] = float(bool(gold.intersection(retrieved)))
        values[f"recall_at_{cutoff}"] = (
            float(len(gold.intersection(retrieved))) / float(len(gold)) if gold else 0.0
        )
    rank = next((index + 1 for index, statute_id in enumerate(ranked) if statute_id in gold), None)
    values["mrr"] = 1.0 / rank if rank else 0.0
    return values


def bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int,
    samples: int,
) -> Tuple[float, float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("Paired bootstrap requires equally sized non-empty vectors")
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    observed = float(np.mean(left_array - right_array))
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    chunk_size = 1000
    position = 0
    while position < samples:
        count = min(chunk_size, samples - position)
        indices = rng.integers(0, len(left_array), size=(count, len(left_array)))
        differences[position : position + count] = np.mean(
            left_array[indices] - right_array[indices], axis=1
        )
        position += count
    low, high = np.percentile(differences, [2.5, 97.5])
    return observed, float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation", type=Path, default=PROJECT_ROOT / "reports/data/lecoqa_gold_reconciliation.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/phase3_2")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reconciliation = json.loads(args.reconciliation.read_text(encoding="utf-8"))
    reconciliation_rows = {int(row["query_id"]): row for row in reconciliation["rows"]}
    retrieval_rows = {
        method: {int(row["query_id"]): row for row in load_jsonl(path)}
        for method, path in METHOD_PATHS.items()
    }
    query_ids = sorted(set(reconciliation_rows))
    if any(set(rows) != set(query_ids) for rows in retrieval_rows.values()):
        raise ValueError("Retrieval files and reconciliation do not cover the same query IDs")

    subset_specs = {
        "raw_all": lambda row: (True, row["raw_match_ids"]),
        "raw_clean": lambda row: (bool(row["raw_position_audit"]["raw_clean"]), row["raw_match_ids"]),
        "high_confidence_reconciled": lambda row: (
            bool(row["raw_position_audit"]["all_related_high_confidence"])
            and bool(row["raw_position_audit"]["all_related_resolved"]),
            row["reconciled_gold_ids"],
        ),
        "resolved_agreement": lambda row: (
            row["reconciliation_status"] in {"clean", "order_only_conflict"}
            and bool(row["raw_position_audit"]["all_related_resolved"]),
            row["reconciled_gold_ids"],
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: List[Dict[str, Any]] = []
    subset_values: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for subset_name, selector in subset_specs.items():
        selected: Dict[int, List[int]] = {}
        for query_id in query_ids:
            include, gold_ids = selector(reconciliation_rows[query_id])
            if include and gold_ids:
                selected[query_id] = [int(value) for value in gold_ids]
        subset_values[subset_name] = {}
        for method, rows_by_query in retrieval_rows.items():
            selected_rows = [row_with_gold(rows_by_query[query_id], selected[query_id]) for query_id in selected]
            metrics = retrieval_metrics(selected_rows)
            metric_rows.append({
                "gold_definition": subset_name,
                "retriever": method,
                "query_count": len(selected_rows),
                **metrics,
            })
            subset_values[subset_name][method] = {
                metric: [query_metric_values(rows_by_query[query_id], selected[query_id])[metric] for query_id in selected]
                for metric in ("hit_at_5", "hit_at_10", "recall_at_5", "recall_at_10", "mrr")
            }

    retrieval_csv = args.output_dir / "retrieval_final.csv"
    with retrieval_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    pair_specs = [("dense_cls", "hybrid_cls"), ("dense_cls", "bm25"), ("hybrid_cls", "bm25")]
    paired_rows: List[Dict[str, Any]] = []
    for subset_name, methods in subset_values.items():
        for left_method, right_method in pair_specs:
            for metric in ("hit_at_5", "hit_at_10", "recall_at_5", "recall_at_10", "mrr"):
                left = methods[left_method][metric]
                right = methods[right_method][metric]
                observed, low, high = bootstrap_difference(
                    left,
                    right,
                    seed=args.seed,
                    samples=args.bootstrap_samples,
                )
                paired_rows.append({
                    "gold_definition": subset_name,
                    "left_method": left_method,
                    "right_method": right_method,
                    "metric": metric,
                    "query_count": len(left),
                    "observed_difference_left_minus_right": round(observed, 6),
                    "bootstrap_ci_low": round(low, 6),
                    "bootstrap_ci_high": round(high, 6),
                    "bootstrap_samples": args.bootstrap_samples,
                    "seed": args.seed,
                })

    paired_csv = args.output_dir / "retrieval_paired_bootstrap.csv"
    with paired_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    summary = {
        "evaluation_version": "phase3_2_v1_gold_definitions",
        "reconciliation_artifact": str(args.reconciliation),
        "retrieval_inputs": {method: str(path) for method, path in METHOD_PATHS.items()},
        "retrieval_input_sha256": {method: sha256_file(path) for method, path in METHOD_PATHS.items()},
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed},
        "subset_definitions": {
            "raw_all": "raw LeCoQA match_id for all 309 test queries",
            "raw_clean": "raw_position_audit.raw_clean == true",
            "high_confidence_reconciled": "all related statutes resolved by Level A or B only",
            "resolved_agreement": "clean or order_only_conflict after non-positional resolution; Level C may be present",
        },
        "retrieval_metrics": metric_rows,
        "paired_bootstrap": paired_rows,
    }
    (args.output_dir / "retrieval_final.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"retrieval_csv": str(retrieval_csv), "paired_csv": str(paired_csv), "rows": len(metric_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
