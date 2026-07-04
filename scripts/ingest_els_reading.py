"""One-shot ELS reading ingestion from the local PDF into data/questions.json.

Parses Intermediate Passages 1-4 from *English Through Reading* (ELS, 2004),
cross-references Ex.2 multiple-choice answers against the answer key (p.409),
and appends structured Reading items.  Safe to re-run: existing question ids
are skipped.

Usage (from project root):
    python scripts/ingest_els_reading.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = Path(
    r"C:\Users\Fayzulloh\Downloads\2024-08-23-18-51-27_26cfb7d4d73697d4de1136baa132d47c.pdf"
)
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.json"
SKILLS_PATH = PROJECT_ROOT / "data" / "skills.json"

# Ex.2 answer key (0-based indices) verified against PDF p.409.
ANSWER_KEY: dict[str, list[int]] = {
    "reading_recruiting_agents": [2, 3, 1],  # C, D, B
    "reading_lost_memories": [1, 4, 3],  # B, E, D
    "reading_palm_trees": [0, 2, 4],  # A, C, E
    "reading_overreacting_joke": [4, 3, 0],  # E, D, A
}

PASSAGES: list[dict] = [
    {
        "skill_id": "reading_recruiting_agents",
        "title": "THE BEST RECRUITING AGENTS",
        "title_pattern": r"THE BEST RECRUITING AGENTS",
        "source": "ELS, English Through Reading (2004), Intermediate Passage 1, Ex.2; answer key p.409",
    },
    {
        "skill_id": "reading_lost_memories",
        "title": "TO BRING BACK LOST MEMORIES",
        "title_pattern": r"TO BRING BACK LOST MEMORIES",
        "source": "ELS, English Through Reading (2004), Intermediate Passage 2, Ex.2; answer key p.409",
    },
    {
        "skill_id": "reading_palm_trees",
        "title": "PALM TREES",
        "title_pattern": r"PALM TREES",
        "source": "ELS, English Through Reading (2004), Intermediate Passage 3, Ex.2; answer key p.409",
    },
    {
        "skill_id": "reading_overreacting_joke",
        "title": "OVERREACTING TO A JOKE",
        "title_pattern": r"OVERREACTING TO A JOKE",
        "source": "ELS, English Through Reading (2004), Intermediate Passage 4, Ex.2; answer key p.409",
    },
]

LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_passage(full_text: str, title_pattern: str) -> str:
    """Return passage body between title and EXERCISE 1."""
    match = re.search(
        rf"(?:\d+\.\s*)?{title_pattern}\s*(.*?)\s*EXERCISE 1:",
        full_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Passage body not found for {title_pattern!r}")
    body = match.group(1)
    body = re.sub(r"\n+", " ", body)
    return _normalize(body)


def _extract_mc_questions(full_text: str) -> list[dict]:
    """Parse EXERCISE 2 stems and A-E options from passage page text."""
    block_match = re.search(
        r"EXERCISE 2: Choose the correct answer according to the passage\.(.*?)"
        r"(?:EXERCISE 3:|EXERCISE 2:|$)",
        full_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not block_match:
        raise ValueError("EXERCISE 2 block not found")
    block = block_match.group(1)
    parts = re.split(r"\n(?=\d+\.\s)", block)
    questions = []
    for part in parts:
        part = part.strip()
        if not part or not re.match(r"\d+\.", part):
            continue
        stem_match = re.match(
            r"(\d+)\.\s*(.*?)\n\s*A\)",
            part,
            flags=re.DOTALL,
        )
        if not stem_match:
            continue
        qnum = int(stem_match.group(1))
        stem = _normalize(stem_match.group(2))
        options = []
        for letter in "ABCDE":
            opt_match = re.search(
                rf"{letter}\)\s*(.*?)(?=\n\s*[A-E]\)|\n\s*EXERCISE|\Z)",
                part,
                flags=re.DOTALL,
            )
            if opt_match:
                options.append(_normalize(opt_match.group(1)))
        if len(options) != 5:
            raise ValueError(f"Question {qnum}: expected 5 options, got {len(options)}")
        questions.append({"number": qnum, "stem": stem, "options": options})
    return questions


def _page_texts(pdf: pdfplumber.PDF) -> list[str]:
    return [(p.extract_text() or "") for p in pdf.pages]


def _passage_pages(texts: list[str], title_pattern: str) -> str:
    """Concatenate pages containing the passage through its EXERCISE 2 block."""
    candidates = [
        i
        for i, t in enumerate(texts)
        if re.search(title_pattern, t, re.I) and "EXERCISE 1:" in t
    ]
    if not candidates:
        raise ValueError(f"No passage page found for {title_pattern!r}")
    start_idx = candidates[0]
    chunks = [texts[start_idx]]
    for j in range(start_idx + 1, min(start_idx + 4, len(texts))):
        chunks.append(texts[j])
        if "EXERCISE 2:" in texts[j] and re.search(r"\d+\.\s", texts[j]):
            break
    return "\n".join(chunks)


def build_items(pdf_path: Path) -> list[dict]:
    """Extract structured question dicts ready for questions.json."""
    with pdfplumber.open(str(pdf_path)) as pdf:
        texts = _page_texts(pdf)

    items: list[dict] = []
    for spec in PASSAGES:
        blob = _passage_pages(texts, spec["title_pattern"])
        passage = _extract_passage(blob, spec["title_pattern"])
        mc = _extract_mc_questions(blob)
        answers = ANSWER_KEY[spec["skill_id"]]
        if len(mc) != len(answers):
            raise ValueError(
                f"{spec['skill_id']}: {len(mc)} questions vs {len(answers)} key entries"
            )
        for q, ans_idx in zip(mc, answers):
            items.append(
                {
                    "id": f"{spec['skill_id']}_q{q['number']}",
                    "skill_id": spec["skill_id"],
                    "question_text": q["stem"],
                    "options": q["options"],
                    "correct_answer": ans_idx,
                    "difficulty": "medium" if q["number"] == 1 else "hard",
                    "passage_text": passage,
                    "source": f"{spec['source']} Q{q['number']}",
                }
            )
    return items


def merge_into_bank(items: list[dict]) -> tuple[int, int]:
    """Append new items; return (added, skipped)."""
    bank = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    skills = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
    skill_ids = {s["id"] for s in skills}
    existing_ids = {q["id"] for q in bank}
    added = skipped = 0
    for item in items:
        if item["id"] in existing_ids:
            skipped += 1
            continue
        if item["skill_id"] not in skill_ids:
            raise ValueError(f"Unknown skill_id {item['skill_id']!r}; update skills.json first.")
        bank.append(item)
        existing_ids.add(item["id"])
        added += 1
    if added:
        QUESTIONS_PATH.write_text(
            json.dumps(bank, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return added, skipped


def main() -> int:
    if not PDF_PATH.is_file():
        print(f"PDF not found: {PDF_PATH}", file=sys.stderr)
        return 1
    items = build_items(PDF_PATH)
    added, skipped = merge_into_bank(items)
    print(f"Ingestion complete: {added} added, {skipped} skipped ({len(items)} parsed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
