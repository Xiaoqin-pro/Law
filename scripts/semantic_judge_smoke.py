"""Run a non-semantic smoke test for the blinded Judge input package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "outputs/phase3_5/semantic_model_assisted/semantic_judge_inputs_blinded.jsonl"
IDENTITY = ROOT / "outputs/phase3_5/semantic_model_assisted/semantic_judge_identity_map.jsonl"
REPORT = ROOT / "reports/phase3_5/semantic_judge/semantic_judge_smoke_report.json"

FORBIDDEN = {
    "model", "method", "method_label", "original_stratum", "sampling_weight",
    "selection_probability", "gold_visible", "auto_gold_citation_match",
    "auto_citation_outside_visible", "auto_missing_citation",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flatten_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(flatten_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(flatten_keys(child))
        return keys
    return set()


def main() -> None:
    blinded = read_jsonl(INPUTS)
    identity = read_jsonl(IDENTITY)
    selected: List[Dict[str, Any]] = []
    seen_pairs: set[tuple[str, bool]] = set()
    for case, mapping in zip(blinded, identity):
        stratum = str(mapping["original_stratum"])
        pair = (stratum, bool(case.get("parsed_citations")))
        if stratum in {"S1_gold_visible_and_gold_cited", "S2_gold_visible_but_not_gold_cited", "S4_gold_not_visible_and_not_gold_cited"} and pair not in seen_pairs:
            selected.append({"answer_id": case["answer_id"], "stratum": stratum, "has_conventional_citation": pair[1]})
            seen_pairs.add(pair)
    if len(selected) < 10:
        selected_ids = {item["answer_id"] for item in selected}
        for case in blinded:
            if len(selected) >= 10:
                break
            if case["answer_id"] not in selected_ids:
                selected.append({
                    "answer_id": case["answer_id"],
                    "stratum": next(item["original_stratum"] for item in identity if item["answer_id"] == case["answer_id"]),
                    "has_conventional_citation": bool(case.get("parsed_citations")),
                })
                selected_ids.add(case["answer_id"])
    selected = selected[:10]
    observed_keys = flatten_keys(selected and next(case for case in blinded if case["answer_id"] == selected[0]["answer_id"]))
    leakage = sorted(observed_keys & FORBIDDEN)
    checks = {
        "ten_cases_selected": len(selected) == 10,
        "all_selected_cases_have_questions_and_answers": all(
            bool(case.get("question")) and bool(case.get("generated_answer"))
            for case in blinded
            if case["answer_id"] in {item["answer_id"] for item in selected}
        ),
        "model_identity_and_automatic_category_blinded": not leakage,
        "identity_map_separate": all(set(case) >= {"answer_id", "model", "method", "original_stratum"} for case in identity),
    }
    payload = {
        "analysis_version": "phase3_5B1_semantic_judge_smoke_v1",
        "input_smoke_status": "PASS" if all(checks.values()) else "FAIL",
        "judge_execution_status": "BLOCKED_MISSING_INDEPENDENT_MODEL",
        "checks": checks,
        "selected_cases": selected,
        "forbidden_key_leakage": leakage,
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "gpu_memory_gb": 8,
            "gpu_memory_total_mib_verified_by_nvidia_smi": 8188,
            "system_memory_gb": 16,
        },
        "available_local_instruction_models": [
            "Qwen2.5-7B-Instruct (already evaluated baseline; excluded as Judge)",
            "Qwen3-4B-Instruct-2507 (already evaluated baseline; excluded as Judge)",
        ],
        "required_next_authority": "Provide or explicitly authorize an independent instruction model/API. Do not rent paid GPU without approval.",
        "note": "No semantic labels were generated. This is an input/schema smoke test, not a semantic evaluation and not human validation.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
