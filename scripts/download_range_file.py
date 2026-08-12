"""Download a large HTTP file in verified, resumable byte ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import shutil
import subprocess
from pathlib import Path
from typing import Tuple


def download_part(
    url: str,
    parts_dir: Path,
    index: int,
    start: int,
    end: int,
    proxy: str,
    retries: int,
) -> Tuple[int, int]:
    expected = end - start + 1
    part_path = parts_dir / f"part-{index:04d}-{start}-{end}.bin"
    if part_path.exists() and part_path.stat().st_size == expected:
        return index, expected
    temporary = part_path.with_suffix(part_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    command = [
        "curl.exe", "--proxy", proxy, "-L", "--fail", "--silent", "--show-error",
        "--retry", str(retries), "--retry-all-errors", "--retry-delay", "2",
        "--connect-timeout", "30", "--max-time", "300", "--range", f"{start}-{end}",
        "--output", str(temporary), url,
    ]
    completed = subprocess.run(command, check=False)
    actual = temporary.stat().st_size if temporary.exists() else 0
    if completed.returncode != 0 or actual != expected:
        raise RuntimeError(f"range {start}-{end} failed: exit={completed.returncode}, expected={expected}, actual={actual}")
    temporary.replace(part_path)
    return index, expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    if args.size <= 0 or args.chunk_size <= 0:
        raise SystemExit("--size and --chunk-size must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output.with_name(args.output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = []
    for index in range(math.ceil(args.size / args.chunk_size)):
        start = index * args.chunk_size
        end = min(args.size - 1, start + args.chunk_size - 1)
        ranges.append((index, start, end))
    completed_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_part, args.url, parts_dir, *item, args.proxy, args.retries) for item in ranges]
        for future in concurrent.futures.as_completed(futures):
            index, size = future.result()
            completed_bytes += size
            print(f"part {index + 1}/{len(ranges)} complete; bytes={completed_bytes}/{args.size}", flush=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".merge.tmp")
    with temporary_output.open("wb") as destination:
        for index, start, end in ranges:
            part_path = parts_dir / f"part-{index:04d}-{start}-{end}.bin"
            with part_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    if temporary_output.stat().st_size != args.size:
        raise RuntimeError(f"merged file has size {temporary_output.stat().st_size}, expected {args.size}")
    temporary_output.replace(args.output)
    print(f"completed {args.output} ({args.size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
