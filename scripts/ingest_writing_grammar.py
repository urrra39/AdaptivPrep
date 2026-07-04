"""Ingest SAT Writing (Grammar) questions from Writing Questions.pdf into grammar_bank.json.

The PDF is an export from the College Board SAT Suite Question Bank. It contains
20 Reading & Writing items (Transitions, Rhetorical Synthesis, Standard English
Conventions). Correct answers are cross-checked against the official SAT Suite
Question Bank (sat-questions.onrender.com / College Board rationales); nothing
is invented.

Usage (from project root):
    python scripts/ingest_writing_grammar.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = Path(r"C:\Users\Fayzulloh\Downloads\Writing Questions.pdf")
OUT_PATH = PROJECT_ROOT / "data" / "grammar_bank.json"
SKILLS_PATH = PROJECT_ROOT / "data" / "skills.json"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.json"

# Verified against SAT Suite Question Bank (official rationales).
VERIFIED_ANSWERS: dict[str, str] = {
    "2b08f514": "C",
    "afec1a70": "A",
    "4d2736f0": "D",
    "d3b7d7a3": "B",
    "39ccb463": "A",
    "e3edc138": "D",
    "00221c00": "B",
    "16631d34": "B",
    "1d79a59d": "B",
    "42e6cc83": "B",
    "83898524": "D",
    "fba5d8d1": "D",
    "dc645172": "D",
    "886dc9f9": "B",
    "59a246dc": "D",
    "e060dd6b": "A",
    "6e071432": "C",
    "6ea8c23f": "D",
    "aab74a3b": "D",
    "512f0ac9": "A",
}

SAT_QB_URL = (
    "https://sat-questions.onrender.com/question/"
    "module:english-group:all-skill:all-difficulty:all-active:all/{qid}"
)


def _letter_to_index(letter: str) -> int:
    return ord(letter.upper()) - ord("A")


def _clean(text: str) -> str:
    text = text.replace("\ufffd", "'")
    text = re.sub(r"\s+", " ", text.strip())
    return text


def _fetch_answer_online(qid: str) -> str | None:
    url = SAT_QB_URL.format(qid=qid)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            html = resp.read().decode("utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    m = re.search(
        r"Correct Answer:\s*</span>\s*<span>\s*([A-D])\s*</span>",
        html,
        re.I,
    )
    if m:
        return m.group(1).upper()
    m = re.search(r"Correct Answer:\s*([A-D])", html, re.I)
    return m.group(1).upper() if m else None


def _resolve_answer(qid: str) -> tuple[str, str]:
    """Return (letter, source)."""
    online = _fetch_answer_online(qid)
    if online:
        verified = VERIFIED_ANSWERS.get(qid)
        if verified and verified != online:
            raise ValueError(
                f"{qid}: online answer {online} != verified {verified}"
            )
        return online, "sat_suite_qb"
    if qid in VERIFIED_ANSWERS:
        return VERIFIED_ANSWERS[qid], "verified_key"
    raise ValueError(f"No answer for {qid}")


def _skill_label(domain_skill: str) -> str:
    ds = domain_skill.lower()
    if "rhetorical synthesis" in ds:
        return "sat_rhetorical_synthesis"
    if "transitions" in ds:
        return "sat_transitions"
    if "standard english" in ds or "boundaries" in ds:
        return "sat_standard_english"
    return "sat_writing"


def _parse_pdf(pdf_path: Path) -> list[dict]:
    with pdfplumber.open(str(pdf_path)) as pdf:
        full = "\n".join((p.extract_text() or "") for p in pdf.pages)

    blocks = re.split(r"Question ID:\s*", full)[1:]
    raw_items = []
    for block in blocks:
        qid = block[:8].strip()
        if not re.fullmatch(r"[a-f0-9]{8}", qid):
            continue
        qm = re.search(r"\nQuestion\s*\n", block)
        am = re.search(r"\nAnswer\s*\n", block)
        if not qm or not am or am.start() <= qm.end():
            continue
        meta_blob = block[: qm.start()]
        stem = block[qm.end() : am.start()].strip()
        opts_blob = block[am.end() :]
        domain_skill = _clean(
            meta_blob.split("Assessment Test Domain Skill Difficulty", 1)[-1]
        )
        options = []
        for om in re.finditer(r"^([A-D])\.\s*(.+?)(?=^\s*[A-D]\.\s|\Z)", opts_blob, re.M | re.S):
            options.append(_clean(om.group(2)))
        if len(options) != 4:
            continue
        raw_items.append(
            {
                "id": qid,
                "domain_skill": domain_skill,
                "question_text": _clean(stem),
                "options": options,
            }
        )
    return raw_items


def ingest(pdf_path: Path) -> dict:
    raw = _parse_pdf(pdf_path)
    errors = []
    questions = []
    answer_sources: dict[str, str] = {}

    for item in raw:
        qid = item["id"]
        try:
            letter, source = _resolve_answer(qid)
            answer_sources[qid] = source
        except ValueError as exc:
            errors.append(str(exc))
            continue
        idx = _letter_to_index(letter)
        skill_id = _skill_label(item["domain_skill"])
        questions.append(
            {
                "id": f"sat_{qid}",
                "external_id": qid,
                "skill_id": skill_id,
                "question_text": item["question_text"],
                "options": item["options"],
                "correct_answer": idx,
                "correct_letter": letter,
                "difficulty": "hard",
                "source": f"SAT Suite QB ({item['domain_skill']})",
                "bank": "grammar",
            }
        )
        time.sleep(0.15)

    skills = sorted({q["skill_id"] for q in questions})
    return {
        "meta": {
            "source": "SAT Suite Question Bank (Writing Questions.pdf)",
            "pdf": pdf_path.name,
            "question_count": len(questions),
            "pdf_question_count": len(raw),
            "answer_sources": answer_sources,
            "parse_errors": errors,
            "note": (
                "Writing Questions.pdf contains 20 official SAT Writing items, "
                "not 200. All available items from the book are included."
            ),
        },
        "skills": skills,
        "questions": questions,
    }


def _strip_legacy_grammar() -> None:
    if not QUESTIONS_PATH.exists():
        return
    with open(QUESTIONS_PATH, encoding="utf-8") as fh:
        legacy = json.load(fh)
    kept = [
        q
        for q in legacy
        if not (
            q.get("skill_id", "").startswith("grammar_")
            or q.get("bank") == "grammar"
        )
    ]
    with open(QUESTIONS_PATH, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, ensure_ascii=False, indent=2)


def _update_skills(bank: dict) -> None:
    with open(SKILLS_PATH, encoding="utf-8") as fh:
        skills = json.load(fh)
    skills = [s for s in skills if s.get("category") != "Grammar"]
    labels = {
        "sat_transitions": "SAT Writing: Transitions",
        "sat_rhetorical_synthesis": "SAT Writing: Rhetorical Synthesis",
        "sat_standard_english": "SAT Writing: Standard English Conventions",
        "sat_writing": "SAT Writing",
    }
    for sid in bank["skills"]:
        skills.append(
            {
                "id": sid,
                "name": labels.get(sid, sid),
                "category": "Grammar",
            }
        )
    with open(SKILLS_PATH, "w", encoding="utf-8") as fh:
        json.dump(skills, fh, ensure_ascii=False, indent=2)


def main() -> int:
    if not PDF_PATH.exists():
        print(f"Missing PDF: {PDF_PATH}", file=sys.stderr)
        return 1
    bank = ingest(PDF_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(bank, fh, ensure_ascii=False, indent=2)
    _strip_legacy_grammar()
    _update_skills(bank)
    print(
        f"Wrote {bank['meta']['question_count']} grammar questions -> {OUT_PATH}"
    )
    if bank["meta"]["parse_errors"]:
        print("ERRORS:", bank["meta"]["parse_errors"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
