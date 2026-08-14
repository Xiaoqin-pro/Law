"""Phase 3 generation helpers with a strict, resumable output contract."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


STATUTE_MARKER_RE = re.compile(r"\[\u6cd5\u6761ID\s+(\d+)\]")


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
            f"[\u6cd5\u6761ID {statute_id}] {statute['statute_name']}\n{statute['statute_text']}"
        )
    return "\n\n".join(blocks)


def build_evidence_blocks(
    retrieval_row: Mapping[str, Any],
    corpus_by_id: Mapping[int, Mapping[str, Any]],
    *,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Return complete statute blocks in retrieval rank order."""

    blocks: List[Dict[str, Any]] = []
    for result in retrieval_row.get("results", [])[:top_k]:
        statute_id = int(result["statute_id"])
        statute = corpus_by_id.get(statute_id)
        if statute is None:
            continue
        blocks.append({
            "statute_id": statute_id,
            "text": f"[\u6cd5\u6761ID {statute_id}] {statute['statute_name']}\n{statute['statute_text']}",
        })
    return blocks


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


def _prompt_token_count(tokenizer: Any, prompt: str) -> int:
    encoded_text = apply_chat_template_text(tokenizer, prompt)
    encoded = tokenizer(encoded_text, add_special_tokens=False, truncation=False)
    return len(encoded["input_ids"])


def pack_evidence_to_budget(
    query: Mapping[str, Any],
    retrieval_row: Mapping[str, Any],
    corpus_by_id: Mapping[int, Mapping[str, Any]],
    *,
    tokenizer: Any,
    prompt_templates: Mapping[str, str],
    top_k: int = 10,
    max_input_tokens: int = 4096,
) -> Dict[str, Any]:
    """Add complete ranked evidence blocks until the prompt budget is full.

    The first block that would exceed the budget and every later block are
    omitted. This keeps the evidence rank order and prevents partial statutes.
    """

    selected: List[Dict[str, Any]] = []
    for block in build_evidence_blocks(retrieval_row, corpus_by_id, top_k=top_k):
        candidate = selected + [block]
        evidence = "\n\n".join(item["text"] for item in candidate)
        prompt = render_prompt("bm25", query, prompt_templates=prompt_templates, evidence=evidence)
        if _prompt_token_count(tokenizer, prompt) > int(max_input_tokens):
            break
        selected.append(block)
    evidence = "\n\n".join(item["text"] for item in selected)
    prompt = render_prompt("bm25", query, prompt_templates=prompt_templates, evidence=evidence)
    return {
        "evidence": evidence,
        "included_statute_ids": [int(item["statute_id"]) for item in selected],
        "input_token_count_before_generation": _prompt_token_count(tokenizer, prompt),
        "evidence_packing_version": "phase3_2_ranked_complete_blocks_v1",
    }


def apply_chat_template_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompt


def audit_prompt_context(
    tokenizer: Any,
    prompt: str,
    *,
    max_input_tokens: int = 4096,
) -> Dict[str, Any]:
    """Measure which retrieved statute blocks survive the model input limit."""

    encoded_text = apply_chat_template_text(tokenizer, prompt)
    full = tokenizer(
        encoded_text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    full_ids = full["input_ids"]
    offsets = full.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("The generation tokenizer must provide fast-tokenizer offsets for context audit")
    limit = int(max_input_tokens)
    visible_token_count = min(len(full_ids), limit)
    markers = list(STATUTE_MARKER_RE.finditer(encoded_text))
    visible: List[int] = []
    fully_visible: List[int] = []
    partially_visible: List[int] = []
    for index, marker in enumerate(markers):
        statute_id = int(marker.group(1))
        block_start = marker.start()
        block_end = markers[index + 1].start() if index + 1 < len(markers) else len(encoded_text)
        block_token_indices = [
            token_index
            for token_index, (start, end) in enumerate(offsets)
            if end > block_start and start < block_end
        ]
        if not block_token_indices:
            continue
        visible_indices = [token_index for token_index in block_token_indices if token_index < visible_token_count]
        if visible_indices:
            visible.append(statute_id)
            if len(visible_indices) == len(block_token_indices):
                fully_visible.append(statute_id)
            else:
                partially_visible.append(statute_id)
    return {
        "input_token_count_before_truncation": len(full_ids),
        "was_truncated": len(full_ids) > limit,
        "visible_statute_ids": visible,
        "fully_visible_statute_ids": fully_visible,
        "partially_visible_statute_ids": partially_visible,
        "context_audit_version": "v1_offsets_chat_template",
    }


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
        encoded_text = apply_chat_template_text(self.tokenizer, prompt)
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
