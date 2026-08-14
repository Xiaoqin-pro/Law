"""Download the independent quantized semantic-judge model artifact."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "unsloth/Phi-3.5-mini-instruct-bnb-4bit"
LOCAL_DIR = ROOT / "outputs/phase3_5/semantic_model_assisted/models/phi-3.5-mini-instruct-bnb-4bit"


def main() -> None:
    LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        max_workers=1,
    )
    print(f"model_id={MODEL_ID}")
    print(f"local_path={resolved}")


if __name__ == "__main__":
    main()
