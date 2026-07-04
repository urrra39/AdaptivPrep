"""Extract 10 extra grammar questions from SAToplam Writing Book to reach 50 total.

The SAToplam book uses a two-column layout, so extracted text is interleaved.
We parse questions from the Boundaries section (easiest to extract).
Answer keys come from the end-of-book tables.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = Path(
    r"C:\Users\Fayzulloh\Downloads\Telegram Desktop\SAToplam Writing Book.pdf"
)
OUT_PATH = PROJECT_ROOT / "data" / "grammar_bank.json"

NEED = 10  # how many more questions we need


def _clean(text: str) -> str:
    text = text.replace("\ufffd", "'")
    text = re.sub(r"\s+", " ", text.strip())
    return text


def _letter_to_index(letter: str) -> int:
    return ord(letter.upper()) - ord("A")


def _extract_answer_keys(full_text: str) -> dict[str, dict[int, str]]:
    """Return {section_name: {q_number: letter}}."""
    sections: dict[str, dict[int, str]] = {}
    parts = re.split(r"Answers:\s*", full_text)
    for part in parts[1:]:
        section_name = part.split("\n", 1)[0].strip()
        answers: dict[int, str] = {}
        for m in re.finditer(r"(\d+)\s+([A-D])", part):
            num = int(m.group(1))
            letter = m.group(2)
            answers[num] = letter
        sections[section_name] = answers
    return sections


def _parse_page_questions(page_text: str) -> list[dict]:
    """Extract questions from a single page's raw text.

    Returns list of {number, passage_snippet, options}.
    """
    results = []

    chunks = re.split(
        r"\.{10,}",  # separator dots
        page_text,
    )

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        option_matches = list(re.finditer(r"([A-D])\)\s*(.+?)(?=\s*[A-D]\)|\Z)", chunk, re.S))
        if len(option_matches) != 4:
            continue

        options = []
        for om in option_matches:
            opt_text = _clean(om.group(2))
            options.append(opt_text)

        q_nums = re.findall(r"(?:^|\n)\s*(\d{1,3})\s*\n", chunk)
        if not q_nums:
            continue
        q_num = int(q_nums[0])

        before_options = chunk[: option_matches[0].start()]
        passage_text = _clean(before_options)

        results.append({
            "number": q_num,
            "passage_snippet": passage_text,
            "options": options,
        })

    return results


def main() -> int:
    if not PDF_PATH.exists():
        print(f"Missing: {PDF_PATH}", file=sys.stderr)
        return 1

    with pdfplumber.open(str(PDF_PATH)) as pdf:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)

        # Parse questions from Boundaries section (pages ~7-96)
        all_questions: list[dict] = []
        for page_idx in range(7, 96):
            if page_idx >= len(pdf.pages):
                break
            page_text = pdf.pages[page_idx].extract_text() or ""
            parsed = _parse_page_questions(page_text)
            all_questions.extend(parsed)

    answer_keys = _extract_answer_keys(full_text)
    boundaries_answers = answer_keys.get("Boundaries", {})
    print(f"Parsed {len(all_questions)} question candidates from Boundaries")
    print(f"Answer keys for Boundaries: {len(boundaries_answers)}")

    with open(OUT_PATH, encoding="utf-8") as f:
        bank = json.load(f)

    existing_count = len(bank["questions"])
    print(f"Current grammar bank size: {existing_count}")

    if existing_count >= 50:
        print("Already have 50+ questions, nothing to add.")
        return 0

    needed = 50 - existing_count
    print(f"Need {needed} more questions")

    added = 0
    seen_nums: set[int] = set()
    for q in all_questions:
        if added >= needed:
            break
        num = q["number"]
        if num in seen_nums:
            continue
        seen_nums.add(num)

        answer_letter = boundaries_answers.get(num)
        if not answer_letter:
            continue

        if len(q["options"]) != 4:
            continue

        qid = f"satoplam_boundaries_{num}"
        bank["questions"].append({
            "id": qid,
            "external_id": f"satoplam_b{num}",
            "skill_id": "sat_standard_english",
            "question_text": q["passage_snippet"],
            "options": q["options"],
            "correct_answer": _letter_to_index(answer_letter),
            "correct_letter": answer_letter,
            "difficulty": "medium",
            "source": f"SAToplam Writing Book - Boundaries Q{num}",
            "bank": "grammar",
        })
        added += 1

    total = len(bank["questions"])
    bank["meta"]["question_count"] = total
    bank["meta"]["note"] = (
        f"SAT questions from Writing Questions.pdf, "
        f"Writing Questions (Hard) 2.0.pdf, "
        f"and SAToplam Writing Book = {total} total grammar items."
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    print(f"Added {added}. Total grammar: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
