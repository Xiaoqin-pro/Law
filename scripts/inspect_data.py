"""Print a compact data audit and deterministic human-check samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citelaw.data import choose_samples, load_raw_dataset, make_statistics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "LeCoQA")
    parser.add_argument("--sample-count", type=int, default=10)
    args = parser.parse_args()

    all_qa, train_qa, test_qa, corpus = load_raw_dataset(args.raw_root)
    print(json.dumps(make_statistics(all_qa, train_qa, test_qa, corpus), ensure_ascii=False, indent=2))
    print("\n=== human-check samples ===")
    for sample in choose_samples(all_qa, args.sample_count):
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
