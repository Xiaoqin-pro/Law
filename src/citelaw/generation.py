"""Phase 3 generation helpers with a strict, resumable output contract."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_processed_queries(path: Path) -> Dict[int, Dict[str, Any]]:
    return {int(row["query_id"]): row for row in load_jsonl(path)}


def build_evidence(
    retrieval_row: Mapping[str, Any],
    corpus_by_id: Mapping[int, Mapping[str, Any]],
    *,
    top_k: int = 10,
) -> str:
    blocks: List[str] = []
    for result in retrieval_row.get("results", [])[:top_k]:
        statute_id = int(result["statute_id"])
        statute = corpus_by_id.get(statute_id)
        if statute is None:
            continue
        blocks.append(
            f"[法条ID {statute_id}] {statute['statute_name']}\n{statute['statute_text']}"
        )
    return "\n\n".join(blocks)


def render_prompt(
    method: str,
    query: Mapping[str, Any],
    *,
    prompt_templates: Mapping[str, str],
    evidence: str = "",
) -> str:
    if method == "direct":
        return prompt_templates["direct"].format(question=query["question"])
    if method in {"bm25", "dense", "hybrid"}:
        return prompt_templates["rag"].format(question=query["question"], evidence=evidence)
    raise ValueError(f"Unknown generation method: {method}")


class LocalQwenGenerator:
    """Load one Qwen model and generate one record at a time.

    4-bit loading is requested explicitly. If bitsandbytes is unavailable or
    the model cannot fit, the exception is intentionally propagated so the
    experiment is marked failed instead of silently changing the setup.
    """

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Optional[Path] = None,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        device_map: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
            device_map=device_map,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
        )
        self.model.eval()
        self.model_revision = getattr(getattr(self.model, "config", None), "_commit_hash", None) or "unknown"

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            encoded_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            encoded_text = prompt
        inputs = self.tokenizer(
            encoded_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def write_generation_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_generation_records(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    records: Dict[int, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        records[int(row["query_id"])] = row
    return records


def classify_runtime_error(exc: BaseException) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "cuda" in text and "memory" in text:
        return "cuda_oom"
    if "bitsandbytes" in text:
        return "missing_bitsandbytes"
    return type(exc).__name__
