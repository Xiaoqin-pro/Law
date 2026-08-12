"""Normalize LeCoQA into stable JSONL files and produce data statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.data import (  # noqa: E402
    choose_samples,
    load_raw_dataset,
    make_statistics,
    validate_normalized,
    write_json,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "LeCoQA")
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--sample-count", type=int, default=10)
    args = parser.parse_args()

    all_qa, train_qa, test_qa, corpus = load_raw_dataset(args.raw_root)
    validate_normalized(all_qa, corpus, split_name="all")
    validate_normalized(train_qa, corpus, split_name="train")
    validate_normalized(test_qa, corpus, split_name="test")

    args.processed_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.processed_root / "queries.jsonl", all_qa)
    write_jsonl(args.processed_root / "train.jsonl", train_qa)
    write_jsonl(args.processed_root / "test.jsonl", test_qa)
    write_jsonl(args.processed_root / "corpus.jsonl", corpus)
    write_json(args.processed_root / "data_stats.json", make_statistics(all_qa, train_qa, test_qa, corpus))
    write_json(args.processed_root / "sample_check.json", choose_samples(all_qa, args.sample_count))

    print(f"qa_total={len(all_qa)} train={len(train_qa)} test={len(test_qa)} corpus={len(corpus)}")
    print(f"processed data written to {args.processed_root}")
    print(f"human-check samples written to {args.processed_root / 'sample_check.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
