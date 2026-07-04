"""Ingest Writing Questions (Hard) 2.0 into grammar_bank.json (append to existing).

Parses the questions PDF and answer PDF, extracts correct answers from
'Correct Answer: X' lines in the answers PDF. Appends to existing grammar bank
without duplicating questions.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q_PDF = Path(
    r"C:\Users\Fayzulloh\Downloads\Telegram Desktop\Writing Questions (Hard) 2.0.pdf"
)
A_PDF = Path(
    r"C:\Users\Fayzulloh\Downloads\Telegram Desktop\Writing Questions (Hard) 2.0 Answers.pdf"
)
OUT_PATH = PROJECT_ROOT / "data" / "grammar_bank.json"


def _clean(text: str) -> str:
    text = text.replace("\ufffd", "'")
    text = re.sub(r"\s+", " ", text.strip())
    return text


def _skill_label(domain_skill: str) -> str:
    ds = domain_skill.lower()
    if "rhetorical synthesis" in ds:
        return "sat_rhetorical_synthesis"
    if "transitions" in ds:
        return "sat_transitions"
    if "standard english" in ds or "boundaries" in ds:
        return "sat_standard_english"
    return "sat_writing"


def _letter_to_index(letter: str) -> int:
    return ord(letter.upper()) - ord("A")


def _extract_answers(pdf_path: Path) -> dict[str, str]:
    """Extract {question_id: correct_letter} from answers PDF."""
    answers: dict[str, str] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        full = "\n".join((p.extract_text() or "") for p in pdf.pages)

    blocks = re.split(r"Question ID:\s*", full)[1:]
    for block in blocks:
        qid = block[:8].strip()
        if not re.fullmatch(r"[a-f0-9]{8}", qid):
            continue
        m = re.search(r"Correct Answer:\s*([A-D])", block, re.I)
        if m:
            answers[qid] = m.group(1).upper()
    return answers


def _parse_questions(pdf_path: Path) -> list[dict]:
    with pdfplumber.open(str(pdf_path)) as pdf:
        full = "\n".join((p.extract_text() or "") for p in pdf.pages)

    blocks = re.split(r"Question ID:\s*", full)[1:]
    items = []
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
        for om in re.finditer(
            r"^([A-D])\.\s*(.+?)(?=^\s*[A-D]\.\s|\Z)", opts_blob, re.M | re.S
        ):
            options.append(_clean(om.group(2)))
        if len(options) != 4:
            continue
        items.append(
            {
                "qid": qid,
                "domain_skill": domain_skill,
                "question_text": _clean(stem),
                "options": options,
            }
        )
    return items


def main() -> int:
    if not Q_PDF.exists():
        print(f"Missing: {Q_PDF}", file=sys.stderr)
        return 1
    if not A_PDF.exists():
        print(f"Missing: {A_PDF}", file=sys.stderr)
        return 1

    answers = _extract_answers(A_PDF)
    print(f"Found {len(answers)} answers in answers PDF")

    items = _parse_questions(Q_PDF)
    print(f"Parsed {len(items)} questions from questions PDF")

    with open(OUT_PATH, encoding="utf-8") as f:
        bank = json.load(f)

    existing_ids = {q["id"] for q in bank["questions"]}
    # Remove previously added supplement questions (gram_*) to replace with real ones
    bank["questions"] = [q for q in bank["questions"] if not q["id"].startswith("gram_")]
    existing_ids = {q["id"] for q in bank["questions"]}

    added = 0
    errors = []
    for item in items:
        qid = item["qid"]
        full_id = f"sat_{qid}"
        if full_id in existing_ids:
            continue
        letter = answers.get(qid)
        if not letter:
            errors.append(f"No answer for {qid}")
            continue
        skill_id = _skill_label(item["domain_skill"])
        bank["questions"].append(
            {
                "id": full_id,
                "external_id": qid,
                "skill_id": skill_id,
                "question_text": item["question_text"],
                "options": item["options"],
                "correct_answer": _letter_to_index(letter),
                "correct_letter": letter,
                "difficulty": "hard",
                "source": f"SAT Suite QB Hard 2.0 ({item['domain_skill']})",
                "bank": "grammar",
            }
        )
        existing_ids.add(full_id)
        added += 1

    total = len(bank["questions"])
    bank["meta"]["question_count"] = total
    bank["meta"]["note"] = (
        f"SAT official questions from Writing Questions.pdf + "
        f"Writing Questions (Hard) 2.0.pdf = {total} total grammar items."
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    print(f"Added {added} new SAT questions. Total grammar: {total}")
    if errors:
        print(f"Errors: {errors}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
