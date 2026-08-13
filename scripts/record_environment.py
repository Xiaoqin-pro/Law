"""Record the runtime environment used by the correction experiments."""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def main() -> int:
    package_names = ["torch", "transformers", "bitsandbytes", "numpy", "faiss", "jieba", "FlagEmbedding", "sentence_transformers"]
    packages = {name: {"available": importlib.util.find_spec(name) is not None, "version": package_version(name)} for name in package_names}
    payload = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
    }
    try:
        import torch

        payload["torch_cuda"] = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        payload["torch_cuda_error"] = str(exc)
    output_dir = PROJECT_ROOT / "environment"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase3_system.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
    (output_dir / "phase3_pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
