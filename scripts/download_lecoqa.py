"""Download the raw LeCoQA files without changing their upstream contents."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data_sources.json"


def download(url: str, destination: Path, *, force: bool = False) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return f"kept {destination}"
    request = urllib.request.Request(url, headers={"User-Agent": "citelaw-data-prep/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    return f"downloaded {destination}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "LeCoQA")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    for item in config["files"].values():
        destination = args.raw_root / item["relative_path"]
        print(download(item["url"], destination, force=args.force))
    print(f"raw dataset ready at {args.raw_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
