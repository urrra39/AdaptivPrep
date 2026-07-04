"""Ingest Vocabook.pdf Fighting Time MCQs into vocabulary_bank.json.

Parses College Panda (Sets 1–16) and Ivy Global (Sets 1–4) fill-in-the-blank
exercises with A–D options, cross-referenced against answer keys on pages
412–416. Target: 200 authentic vocabulary-in-context items (160 + 40).

Usage (from project root):
    python scripts/ingest_vocabook.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = Path(r"C:\Users\Fayzulloh\Downloads\Vocabook.pdf")
OUT_PATH = PROJECT_ROOT / "data" / "vocabulary_bank.json"
SKILLS_PATH = PROJECT_ROOT / "data" / "skills.json"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.json"

TARGET_COUNT = 200
COLLEGE_PANDA_SETS = 16
IVY_SETS_FOR_TOPUP = 4  # 160 + 40 = 200

MCQ_RE = re.compile(
    r"(?P<num>\d+)\.\s+"
    r"(?P<stem>.+?______.+?)"
    r"\nA\.\s*(?P<a>.+?)\nB\.\s*(?P<b>.+?)\nC\.\s*(?P<c>.+?)\nD\.\s*(?P<d>.+?)"
    r"(?=\n\d+\.|\n@|\Z)",
    re.S,
)
SET_RE = re.compile(r"^Set\s+(\d+)\b", re.M)
FIGHT_RE = re.compile(r"Fight(?:ing)?\s+Time", re.I)
SECTION_MARKERS = {
    "college_panda": re.compile(r"College Panda\s+400\s+Words", re.I),
    "ivy_global": re.compile(r"Ivy Global Words\s+500", re.I),
    "advanced": re.compile(r"Advanced Package Words", re.I),
    "satashkent": re.compile(r"SATashkent Words", re.I),
    "answers": re.compile(r"^Answers\s*$", re.M),
}


def _clean(text: str) -> str:
    text = text.replace("\ufffd", "'")
    text = re.sub(r"@satashkent\s+\d+\s*$", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text.strip())
    return text


def _page_texts(pdf: pdfplumber.PDF) -> list[str]:
    return [(p.extract_text() or "") for p in pdf.pages]


def _detect_section(text: str, current: str) -> str:
    for name, pat in SECTION_MARKERS.items():
        if pat.search(text):
            return name
    return current


def _parse_answer_grid(text: str) -> dict[int, dict[int, str]]:
    """Parse '# SET 1 SET 2 ...' grids into {set_num: {q_num: letter}}."""
    out: dict[int, dict[int, str]] = {}
    lines = text.splitlines()
    headers: list[int] = []
    for line in lines:
        line = line.strip()
        if line.startswith("# SET"):
            headers = [int(x) for x in re.findall(r"SET\s+(\d+)", line)]
            continue
        if not headers:
            continue
        m = re.match(r"^(\d+)\s+((?:[A-D]\s*)+)$", line)
        if not m:
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit() and all(
                re.fullmatch(r"[A-D]", p) for p in parts[1:]
            ):
                qnum = int(parts[0])
                letters = parts[1:]
            else:
                continue
        else:
            qnum = int(m.group(1))
            letters = m.group(2).split()
        for set_num, letter in zip(headers, letters):
            out.setdefault(set_num, {})[qnum] = letter.upper()
    return out


def _extract_answer_keys(texts: list[str]) -> tuple[dict, dict]:
    start = next(
        i for i, t in enumerate(texts) if SECTION_MARKERS["answers"].search(t)
    )
    blob = "\n".join(texts[start:])
    cp_blob = blob.split("Ivy Global Words 500")[0]
    ivy_blob = blob.split("Ivy Global Words 500")[1].split("Advanced Package")[0]
    return _parse_answer_grid(cp_blob), _parse_answer_grid(ivy_blob)


def _parse_mcqs(text: str) -> list[dict]:
    items = []
    for m in MCQ_RE.finditer(text):
        stem = _clean(m.group("stem"))
        if "Learning Time" in stem or "Reading Time" in stem:
            continue
        opts = [_clean(m.group(k)) for k in ("a", "b", "c", "d")]
        items.append({"num": int(m.group("num")), "stem": stem, "options": opts})
    return items


def _collect_fighting_sets(texts: list[str]) -> tuple[list[list[dict]], list[list[dict]]]:
    """Return (college_panda_sets, ivy_global_sets) each a list of 10-MCQ sets."""
    all_mcqs: list[dict] = []
    ivy_started = False
    answers_page = next(
        i for i, t in enumerate(texts) if SECTION_MARKERS["answers"].search(t)
    )

    for i, text in enumerate(texts):
        if i >= answers_page:
            break
        if i < 8:  # skip front matter / table of contents
            continue
        if "Ivy Global Words 500" in text and "Learning Time" in text:
            ivy_started = True
        if not FIGHT_RE.search(text):
            continue
        all_mcqs.extend(_parse_mcqs(text))

    cp_flat = all_mcqs[: COLLEGE_PANDA_SETS * 10]
    ivy_flat = all_mcqs[
        COLLEGE_PANDA_SETS * 10 : COLLEGE_PANDA_SETS * 10 + IVY_SETS_FOR_TOPUP * 10
    ]

    def _chunk(flat: list[dict], n_sets: int) -> list[list[dict]]:
        out = []
        for s in range(n_sets):
            chunk = flat[s * 10 : (s + 1) * 10]
            if len(chunk) == 10:
                out.append(chunk)
        return out

    return _chunk(cp_flat, COLLEGE_PANDA_SETS), _chunk(ivy_flat, IVY_SETS_FOR_TOPUP)


def _letter_to_index(letter: str) -> int:
    return ord(letter.upper()) - ord("A")


def ingest(pdf_path: Path) -> dict:
    with pdfplumber.open(str(pdf_path)) as pdf:
        texts = _page_texts(pdf)
    cp_keys, ivy_keys = _extract_answer_keys(texts)
    cp_sets, ivy_sets = _collect_fighting_sets(texts)

    errors = []
    questions = []

    def _add(section: str, set_num: int, q: dict, q_idx: int, keys: dict) -> None:
        key = keys.get(set_num, {}).get(q_idx)
        if not key:
            errors.append(f"No answer key for {section} set {set_num} q{q_idx}")
            return
        skill_id = f"vocab_{section}_set_{set_num:02d}"
        questions.append(
            {
                "id": f"vocab_{section}_s{set_num:02d}_q{q_idx:02d}",
                "skill_id": skill_id,
                "question_text": q["stem"],
                "options": q["options"],
                "correct_answer": _letter_to_index(key),
                "correct_letter": key,
                "difficulty": "medium",
                "source": f"Vocabook {section} Set {set_num} Fighting Time Q{q_idx}",
                "bank": "vocabulary",
                "set": set_num,
                "set_question": q_idx,
            }
        )

    for set_num, mcqs in enumerate(cp_sets, start=1):
        for q_idx, q in enumerate(mcqs, start=1):
            _add("college_panda", set_num, q, q_idx, cp_keys)

    for set_num, mcqs in enumerate(ivy_sets, start=1):
        for q_idx, q in enumerate(mcqs, start=1):
            _add("ivy_global", set_num, q, q_idx, ivy_keys)

    questions = questions[:TARGET_COUNT]
    skills = sorted({q["skill_id"] for q in questions})

    return {
        "meta": {
            "source": "Vocabook by @satashkent (College Panda + Ivy Global)",
            "pdf": pdf_path.name,
            "question_count": len(questions),
            "target_count": TARGET_COUNT,
            "college_panda_sets": len(cp_sets),
            "ivy_global_sets": len(ivy_sets),
            "parse_errors": errors,
        },
        "skills": skills,
        "questions": questions,
    }


def _strip_legacy_vocabulary() -> None:
    if not QUESTIONS_PATH.exists():
        return
    with open(QUESTIONS_PATH, encoding="utf-8") as fh:
        legacy = json.load(fh)
    kept = [
        q
        for q in legacy
        if not (
            q.get("skill_id", "").startswith("vocab_")
            or q.get("bank") == "vocabulary"
        )
    ]
    with open(QUESTIONS_PATH, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, ensure_ascii=False, indent=2)


def _update_skills(bank: dict) -> None:
    with open(SKILLS_PATH, encoding="utf-8") as fh:
        skills = json.load(fh)
    skills = [
        s
        for s in skills
        if s.get("category") not in ("Vocabulary", "Academic Vocabulary")
    ]
    for sid in bank["skills"]:
        if sid.startswith("vocab_college_panda"):
            n = sid.rsplit("_", 1)[-1]
            name = f"Vocab: College Panda Set {int(n)}"
        elif sid.startswith("vocab_ivy_global"):
            n = sid.rsplit("_", 1)[-1]
            name = f"Vocab: Ivy Global Set {int(n)}"
        else:
            name = sid
        skills.append({"id": sid, "name": name, "category": "Vocabulary"})
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
    _strip_legacy_vocabulary()
    _update_skills(bank)
    print(
        f"Wrote {bank['meta']['question_count']} vocabulary questions -> {OUT_PATH}"
    )
    if bank["meta"]["question_count"] < TARGET_COUNT:
        print(
            f"WARNING: expected {TARGET_COUNT}, got {bank['meta']['question_count']}",
            file=sys.stderr,
        )
    if bank["meta"]["parse_errors"]:
        print("ERRORS (first 10):", bank["meta"]["parse_errors"][:10], file=sys.stderr)
    return 0 if bank["meta"]["question_count"] >= TARGET_COUNT * 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
