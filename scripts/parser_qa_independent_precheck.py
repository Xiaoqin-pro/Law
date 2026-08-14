"""Independent, non-acceptance precheck for Parser QA.

This uses a small standalone regex/number normalizer and raw answer text.  It
does not call ``extract_citations`` and does not write any manual workbook
fields.  Its output is a convenience aid for a human reviewer, not a human
annotation or a replacement for the Parser QA acceptance decision.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
PHASE_DIR = ROOT / "reports/phase3_5"
CORPUS_PATH = ROOT / "data/processed/corpus.jsonl"
ARTICLE_CHARS = r"0-9零〇一二两三四五六七八九十百千万亿"
CITATION_RE = re.compile(
    rf"(?:《(?P<bracket_law>[^》]{{1,80}})》|(?P<plain_law>.+?))\s*第\s*(?P<article>[{ARTICLE_CHARS}]+)\s*条(?:\s*之\s*(?P<suffix>[{ARTICLE_CHARS}]+))?"
)

DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}


def node_executable() -> str:
    return r"C:\Users\111\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"


def read_sheet(path: Path) -> List[Dict[str, Any]]:
    result = subprocess.run(
        [node_executable(), str(ROOT / "scripts/workbook_to_json.mjs"), str(path), "parser_qa"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)["rows"]


def norm_law(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).strip("《》")


def canonical_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def chinese_number(value: str) -> Optional[int]:
    value = value.strip()
    if value.isdigit():
        return int(value)
    if not value or any(char not in DIGITS and char not in UNITS for char in value):
        return None
    if not any(char in UNITS for char in value):
        return int("".join(str(DIGITS[char]) for char in value))
    total = 0
    section = 0
    number = 0
    for char in value:
        if char in DIGITS:
            number = DIGITS[char]
        else:
            unit = UNITS[char]
            if unit >= 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
    return total + section + number


def article_key(main: str, suffix: str = "") -> Optional[str]:
    main_number = chinese_number(main)
    if main_number is None:
        return None
    if not suffix:
        return str(main_number)
    suffix_number = chinese_number(suffix)
    return None if suffix_number is None else f"{main_number}之{suffix_number}"


def parse_raw_citation(raw: str) -> Optional[Tuple[str, str]]:
    match = CITATION_RE.search(raw)
    if not match:
        return None
    law = norm_law(match.group("bracket_law") or match.group("plain_law") or "")
    article = article_key(match.group("article"), match.group("suffix") or "")
    return (law, article) if law and article else None


def split_corpus_name(name: str) -> Optional[Tuple[str, str]]:
    # Match the law name up to the final article expression.  Corpus names
    # can contain brackets, quotation marks, and explanatory punctuation.
    match = re.search(rf"(?P<law>.+?)第(?P<article>[{ARTICLE_CHARS}]+)条(?:之(?P<suffix>[{ARTICLE_CHARS}]+))?$", name)
    if not match:
        return None
    article = article_key(match.group("article"), match.group("suffix") or "")
    return (norm_law(match.group("law")), article) if article else None


def law_matches(observed: str, expected: str) -> bool:
    observed = norm_law(observed)
    expected = norm_law(expected)
    return observed == expected or observed.endswith(expected) or expected.endswith(observed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PHASE_DIR / "parser_qa.xlsx")
    parser.add_argument("--output-dir", type=Path, default=PHASE_DIR)
    args = parser.parse_args()
    corpus = {}
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            corpus[int(row["statute_id"])] = row
    corpus_keys = []
    for statute_id, row in corpus.items():
        split = split_corpus_name(str(row["statute_name"]))
        if split:
            corpus_keys.append((statute_id, split[0], split[1]))

    rows = read_sheet(args.input)
    prechecked: List[Dict[str, Any]] = []
    for row in rows:
        raw_answer = str(row.get("raw_answer") or "")
        raw_citation = str(row.get("raw_citation") or "")
        parser_status = str(row.get("parser_status") or "")
        parsed = parse_raw_citation(raw_citation) if raw_citation else None
        extraction = "yes" if raw_citation and canonical_text(raw_citation) in canonical_text(raw_answer) else "no" if raw_citation else "not_applicable"
        law_check = "not_applicable"
        article_check = "not_applicable"
        mapping_check = "not_applicable"
        if parsed:
            observed_law, observed_article = parsed
            law_check = "yes" if law_matches(observed_law, str(row.get("normalized_law_name") or "")) else "no"
            article_check = "yes" if observed_article == str(row.get("article_number") or "") else "no"
            resolved_id = row.get("resolved_statute_id")
            if parser_status == "resolved_unique" and resolved_id not in (None, ""):
                exact_candidates = [sid for sid, law, article in corpus_keys if article == observed_article and norm_law(observed_law) == norm_law(law)]
                candidates = exact_candidates or [sid for sid, law, article in corpus_keys if article == observed_article and law_matches(observed_law, law)]
                mapping_check = "yes" if len(candidates) == 1 and int(float(resolved_id)) == candidates[0] else "no"
        no_citation_has_citation = bool(CITATION_RE.search(raw_answer))
        prechecked.append({
            "audit_unit_id": row.get("audit_unit_id"),
            "unit_type": row.get("unit_type"),
            "query_id": row.get("query_id"),
            "raw_citation": raw_citation,
            "parser_status": parser_status,
            "independent_extraction_precheck": extraction,
            "independent_law_name_precheck": law_check,
            "independent_article_number_precheck": article_check,
            "independent_mapping_precheck": mapping_check,
            "no_citation_control_detected_citation": no_citation_has_citation if not raw_citation else "not_applicable",
            "human_annotation_required": True,
        })

    resolved = [row for row in prechecked if row["unit_type"] == "resolved_citation"]
    controls = [row for row in prechecked if row["unit_type"] == "no_conventional_citation_answer"]
    summary = {
        "analysis_version": "phase3_5B1_parser_independent_precheck_v1",
        "status": "PRECHECK_ONLY_NOT_ACCEPTANCE",
        "method": "standalone raw-text regex and number normalization; extract_citations was not called",
        "rows": len(prechecked),
        "resolved_subset": len(resolved),
        "resolved_extraction_yes": sum(row["independent_extraction_precheck"] == "yes" for row in resolved),
        "resolved_law_name_yes": sum(row["independent_law_name_precheck"] == "yes" for row in resolved),
        "resolved_article_number_yes": sum(row["independent_article_number_precheck"] == "yes" for row in resolved),
        "resolved_mapping_yes": sum(row["independent_mapping_precheck"] == "yes" for row in resolved),
        "no_citation_controls": len(controls),
        "no_citation_controls_with_independent_citation": sum(bool(row["no_citation_control_detected_citation"]) for row in controls),
        "warning": "These values are machine prechecks only. They must not be copied into human annotation fields or reported as human Parser QA.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "parser_qa_independent_precheck.json").write_text(json.dumps({"summary": summary, "rows": prechecked}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Parser QA independent precheck",
        "",
        "**PRECHECK ONLY — NOT HUMAN VALIDATION AND NOT ACCEPTANCE.**",
        "",
        summary["method"] + ".",
        "",
        f"- Resolved subset: {summary['resolved_subset']}",
        f"- Independent extraction checks yes: {summary['resolved_extraction_yes']}",
        f"- Independent law-name checks yes: {summary['resolved_law_name_yes']}",
        f"- Independent article-number checks yes: {summary['resolved_article_number_yes']}",
        f"- Independent mapping checks yes: {summary['resolved_mapping_yes']}",
        f"- No-citation controls with an independently detected citation: {summary['no_citation_controls_with_independent_citation']}/{summary['no_citation_controls']}",
        "",
        summary["warning"],
    ]
    (args.output_dir / "parser_qa_independent_precheck.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
