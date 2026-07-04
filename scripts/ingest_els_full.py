"""Full ELS *English Through Reading* (2004) ingestion into data/reading_bank.json.

Extracts every passage in the PDF with authentic Exercise 1 (vocabulary match),
Exercise 2 (reading comprehension MCQ), and Exercise 3 (word-bank completion).
All answers are cross-referenced against the answer key at the back of the PDF;
nothing is invented.

Usage (from project root):
    python scripts/ingest_els_full.py
"""
from __future__ import annotations

import difflib
import json
import random
import re
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = Path(
    r"C:\Users\Fayzulloh\Downloads\2024-08-23-18-51-27_26cfb7d4d73697d4de1136baa132d47c.pdf"
)
OUT_PATH = PROJECT_ROOT / "data" / "reading_bank.json"
SKILLS_PATH = PROJECT_ROOT / "data" / "skills.json"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.json"

EX1_START = re.compile(r"EXERCISE 1:\s*Find words", re.I)
EX2_START = re.compile(r"EXERCISE 2:\s*Choose the correct answer", re.I)
EX3_START = re.compile(r"EXERCISE 3:", re.I)
TITLE_SKIP = re.compile(
    r"^(INTERMEDIATE|UPPER|ADVANCED|PASSAGES|CONTENTS|ELS\b|\d+\s*ELS)",
    re.I,
)


def _norm_title(title: str) -> str:
    t = re.sub(r"^\d+\.\s*", "", title.strip())
    t = re.sub(r"^[iİ]\s+", "", t)
    t = re.sub(r"^[A-Z]\s+(?=WHERE|THE|A\s|AN\s|ALPINE|FROM|SWIMMING|CLASSIFYING|ALEXANDRE|DR\.|HELEN|TRAINING|OWNER|SPARTACUS|BOGEY|ETERNAL|FROM\s|OVERREACTING)", "", t)
    if re.match(r"^[lL]-HO", t):
        t = "I" + t[1:]
    t = re.sub(r"\s+", " ", t).upper()
    if t == "L0VE":
        t = "LOVE"
    return t


def _strip_parens(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", title).strip()


def _ocr_title(title: str) -> str:
    t = _strip_parens(title)
    t = re.sub(r"[^A-Z0-9 ]", "", t.upper())
    return re.sub(r"\b0\b", "O", t.replace("0", "O"))


def _title_from_key_line(line: str) -> str | None:
    line = line.strip()
    if not line or re.match(r"Ex\.?\s*1:", line, re.I) or line.startswith("ELS"):
        return None
    m = re.match(r"^.+?\.\s*(.+)$", line)
    if not m:
        m = re.match(r"^.+?\.(.+[A-Za-z].*)$", line)
    if not m:
        return None
    title = _normalize(m.group(1))
    if not title or title.startswith("Ex."):
        return None
    return _norm_title(title)


def _slug(title: str, level: str, seq: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]
    lvl = level.lower().replace("-", "_")[:20]
    return f"els_{lvl}_{seq:03d}_{base}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _page_texts(pdf: pdfplumber.PDF) -> list[str]:
    return [(p.extract_text() or "") for p in pdf.pages]


def _find_passage_starts(texts: list[str]) -> list[tuple[int, str]]:
    """Return (page_index, title) for each passage opening page."""
    starts = []
    for i, t in enumerate(texts):
        if not EX1_START.search(t):
            continue
        first = t.split("\n")[0].strip()
        if not first or EX1_START.match(first):
            continue
        if TITLE_SKIP.match(first):
            continue
        starts.append((i, first))
    return starts


def _passage_blob(texts: list[str], start: int, end: int) -> str:
    return "\n".join(texts[start:end])


def _extract_passage_body(blob: str, title: str) -> str:
    pat = re.escape(title.split("\n")[0][:30])
    m = re.search(rf"(?:\d+\.\s*)?{pat}.*?\n(.*?)\nEXERCISE 1:", blob, re.I | re.S)
    if not m:
        m = re.search(r"^(.*?)\nEXERCISE 1:", blob, re.S)
    body = m.group(1) if m else ""
    body = re.sub(r"\n+", " ", body)
    return _normalize(body)


def _parse_ex1_items(blob: str) -> list[tuple[str, str]]:
    m = re.search(r"EXERCISE 1:.*?COLUMN A(.*?)EXERCISE 2:", blob, re.I | re.S)
    if not m:
        m = re.search(
            r"EXERCISE 1:.*?COLUMN A(.*?)(?:EXERCISE 2:|EXERCISE 3:|\n\d+\.\s.+?\n\s*A\))",
            blob,
            re.I | re.S,
        )
    if not m:
        m = re.search(
            r"EXERCISE 1:(.*?)(?:EXERCISE 2:|EXERCISE 3:|\n\d+\.\s.+?\n\s*A\))",
            blob,
            re.I | re.S,
        )
    if not m:
        return []
    block = m.group(1)
    # Drop COLUMN B header if present in captured block.
    block = re.sub(r"^\s*COLUMN B\s*", "", block, flags=re.I)
    items = []
    for letter, defn in re.findall(
        r"([a-z])\)\s*(.*?)(?=\n[a-z]\)|\n[A-Z]|\Z)", block, re.S | re.I
    ):
        defn = _normalize(defn)
        if defn and len(defn) > 3:
            items.append((letter.lower(), defn))
    return items


def _parse_ex2_items(blob: str) -> list[dict]:
    m = re.search(
        r"EXERCISE 2: Choose the correct answer according to the passage\.(.*?)"
        r"(?:EXERCISE 3:|$)",
        blob,
        re.I | re.S,
    )
    if not m:
        m = re.search(
            r"EXERCISE 2:.*?(?:according to the passage\.)(.*?)(?:EXERCISE 3:|$)",
            blob,
            re.I | re.S,
        )
    if not m:
        # Some passages omit the Ex.2 header and go straight to numbered MCQs.
        m = re.search(
            r"(?:EXERCISE 1:.*?)(?:\n\d+\.\s.+?\n\s*A\))(.*?)(?:EXERCISE 3:|$)",
            blob,
            re.I | re.S,
        )
    if not m:
        return []
    block = m.group(1)
    out = []
    for part in re.split(r"\n(?=\d+\.\s)", block):
        part = part.strip()
        if not re.match(r"\d+\.", part):
            continue
        stem_m = re.match(r"(\d+)\.\s*(.*?)(?:\n\s*A\)|\s+A\))", part, re.S)
        if not stem_m:
            continue
        qnum = int(stem_m.group(1))
        stem = _normalize(stem_m.group(2))
        opts = []
        for letter in "ABCDE":
            om = re.search(
                rf"{letter}\)\s*(.*?)(?=\n\s*[A-E]\)|\n\s*EXERCISE|\Z)",
                part,
                re.S,
            )
            if om:
                opts.append(_normalize(om.group(1)))
        if len(opts) == 5:
            out.append({"number": qnum, "stem": stem, "options": opts})
    return out


def _parse_ex3_items(blob: str) -> list[tuple[int, str]]:
    m = re.search(
        r"EXERCISE 3[\.:\",]+\s*(.*?)(?:\n\s*\d*\s*ELS\b|\Z)",
        blob,
        re.I | re.S,
    )
    if not m:
        m = re.search(r"EXERCISE 3[\.:\",]+\s*(.*)", blob, re.I | re.S)
    if not m:
        return []
    block = m.group(1)
    # Drop instruction line when present.
    block = re.sub(
        r"^Complete the sentences by selecting words from Column B.*?\.?\s*",
        "",
        block,
        flags=re.I | re.S,
    )
    items = []
    for num, sent in re.findall(r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)", block, re.S):
        sent = _normalize(sent)
        if sent and len(sent) > 5:
            items.append((int(num), sent))
    return items


def _parse_key_block(lines: list[str], title: str) -> dict:
    ex1, ex2, ex3 = {}, {}, {}
    for ln in lines[1:]:
        low = ln.lower()
        if re.match(r"Ex\.?\s*1:", ln, re.I):
            body = re.sub(r"^Ex\.?\s*1:\s*", "", ln, flags=re.I)
            for letter, word in re.findall(
                r"([a-z])\)\s*(.+?)(?=\s+[a-z]\)|$)", body, re.I
            ):
                ex1[letter.lower()] = _normalize(word)
        elif re.match(r"Ex\.?\s*2:", ln, re.I):
            body = re.sub(r"^Ex\.?\s*2:\s*", "", ln, flags=re.I)
            body = body.replace("İ", "I").replace("ı", "i")
            body = re.sub(r"^[Iİ]\.([A-E])", r"1.\1", body, flags=re.I)
            body = re.sub(r"^([A-E])\b", r"1.\1", body, flags=re.I)
            for num, letter in re.findall(r"(\d+)\.?\s*([A-E])", body, re.I):
                ex2[int(num)] = ord(letter.upper()) - ord("A")
        elif re.match(r"\.?\s*Ex\.?\s*\.?\s*3:", ln, re.I):
            body = re.sub(r"^\.?\s*Ex\.?\s*\.?\s*3:\s*", "", ln, flags=re.I)
            for num, word in re.findall(r"(\d+)\.?\s*([A-Za-z][A-Za-z\-']*)", body):
                ex3[int(num)] = word.lower()
    return {"ex1": ex1, "ex2": ex2, "ex3": ex3}


def _parse_answer_key(texts: list[str]) -> dict[str, dict]:
    """Map normalized title -> {ex1, ex2, ex3} answer dicts."""
    start = next(i for i, t in enumerate(texts) if t.strip().startswith("ANSWER KEY"))
    lines = []
    for t in texts[start:]:
        lines.extend(t.split("\n"))

    answers: dict[str, dict] = {}
    i = 0
    while i < len(lines) - 1:
        ln = lines[i].strip()
        nxt = lines[i + 1].strip()
        if ln in ("INTERMEDIATE PASSAGES", "UPPER-INTERMEDIATE PASSAGES", "ADVANCED PASSAGES"):
            i += 1
            continue
        if re.match(r"Ex\.?\s*1:", nxt, re.I):
            title = _title_from_key_line(ln)
            if title:
                block = [ln]
                i += 1
                while i < len(lines):
                    nxt_ln = lines[i].strip()
                    nxt2 = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    if (
                        i + 1 < len(lines)
                        and _title_from_key_line(nxt_ln)
                        and re.match(r"Ex\.?\s*1:", nxt2, re.I)
                    ):
                        break
                    block.append(lines[i])
                    i += 1
                parsed = _parse_key_block(block, title)
                if parsed["ex1"] or parsed["ex2"] or parsed["ex3"]:
                    answers[title] = parsed
                continue
        i += 1
    return answers


def _titles_match(passage_title: str, key_title: str) -> bool:
    if passage_title == key_title:
        return True
    p = _strip_parens(passage_title)
    k = _strip_parens(key_title)
    if p == k:
        return True
    if _ocr_title(p) == _ocr_title(k):
        return True
    if difflib.SequenceMatcher(None, p, k).ratio() >= 0.82:
        return True
    pw = p.split()
    kw = k.split()
    if pw and kw and sorted(pw) == sorted(kw):
        return True
    return False


def _match_answer_key(title_norm: str, answer_key: dict) -> dict | None:
    if title_norm in answer_key:
        return answer_key[title_norm]
    stripped = _norm_title(_strip_parens(title_norm))
    if stripped in answer_key:
        return answer_key[stripped]

    best_key = None
    best_ratio = 0.0
    for k, v in answer_key.items():
        if _titles_match(title_norm, k):
            return v
        ratio = difflib.SequenceMatcher(None, _ocr_title(title_norm), _ocr_title(k)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = k
    if best_key and best_ratio >= 0.78:
        return answer_key[best_key]
    return None


def _mc_options(correct: str, word_bank: list[str], rng: random.Random) -> tuple[list[str], int]:
    """Build 5-option MCQ from the passage word bank (Ex.1 / Ex.3 style)."""
    pool = list(dict.fromkeys(w.lower() for w in word_bank if w))
    correct = correct.lower()
    if correct not in pool:
        pool.append(correct)
    distractors = [w for w in pool if w != correct]
    rng.shuffle(distractors)
    opts = distractors[:4] + [correct]
    rng.shuffle(opts)
    return opts, opts.index(correct)


def _build_questions(
    passage_id: str,
    passage_text: str,
    title: str,
    level: str,
    ex1_items: list,
    ex2_items: list,
    ex3_items: list,
    ans: dict,
    rng: random.Random,
) -> list[dict]:
    questions = []
    word_bank = list(ans.get("ex1", {}).values())
    src_base = f"ELS English Through Reading (2004), {level}, {title}"

    for letter, defn in ex1_items:
        correct_word = ans.get("ex1", {}).get(letter)
        if not correct_word:
            continue
        opts, idx = _mc_options(correct_word, word_bank, rng)
        questions.append(
            {
                "id": f"{passage_id}_ex1_{letter}",
                "skill_id": passage_id,
                "passage_id": passage_id,
                "exercise": 1,
                "sub_id": letter,
                "question_text": f"[Exercise 1] {defn}",
                "options": opts,
                "correct_answer": idx,
                "difficulty": "medium",
                "passage_text": passage_text,
                "source": f"{src_base}, Ex.1 {letter}",
            }
        )

    for item in ex2_items:
        num = item["number"]
        ans_idx = ans.get("ex2", {}).get(num)
        if ans_idx is None:
            continue
        questions.append(
            {
                "id": f"{passage_id}_ex2_q{num}",
                "skill_id": passage_id,
                "passage_id": passage_id,
                "exercise": 2,
                "sub_id": str(num),
                "question_text": f"[Exercise 2] {item['stem']}",
                "options": item["options"],
                "correct_answer": ans_idx,
                "difficulty": "medium" if num == 1 else "hard",
                "passage_text": passage_text,
                "source": f"{src_base}, Ex.2 Q{num}",
            }
        )

    for num, sent in ex3_items:
        correct_word = ans.get("ex3", {}).get(num)
        if not correct_word:
            continue
        opts, idx = _mc_options(correct_word, word_bank, rng)
        questions.append(
            {
                "id": f"{passage_id}_ex3_q{num}",
                "skill_id": passage_id,
                "passage_id": passage_id,
                "exercise": 3,
                "sub_id": str(num),
                "question_text": f"[Exercise 3] {sent} ______",
                "options": opts,
                "correct_answer": idx,
                "difficulty": "hard",
                "passage_text": passage_text,
                "source": f"{src_base}, Ex.3 Q{num}",
            }
        )
    return questions


def _infer_level(title_norm: str, seq: int) -> str:
    if seq <= 51:
        return "Intermediate"
    if seq <= 120:
        return "Upper-Intermediate"
    return "Advanced"


def ingest(pdf_path: Path) -> dict:
    with pdfplumber.open(str(pdf_path)) as pdf:
        texts = _page_texts(pdf)
    key_page = next(
        i for i, t in enumerate(texts) if t.strip().startswith("ANSWER KEY")
    )
    starts = _find_passage_starts(texts[:key_page])
    answer_key = _parse_answer_key(texts)
    rng = random.Random(42)
    passages = []
    errors = []

    for seq, (start_idx, raw_title) in enumerate(starts, start=1):
        end_idx = starts[seq][0] if seq < len(starts) else key_page
        blob = _passage_blob(texts, start_idx, end_idx)
        title_norm = _norm_title(raw_title)
        level = _infer_level(title_norm, seq)
        passage_id = _slug(raw_title, level, seq)

        ans = _match_answer_key(title_norm, answer_key)
        if not ans:
            errors.append(f"No answer key for: {title_norm}")
            continue

        passage_text = _extract_passage_body(blob, raw_title)
        ex1 = _parse_ex1_items(blob)
        ex2 = _parse_ex2_items(blob)
        ex3 = _parse_ex3_items(blob)
        questions = _build_questions(
            passage_id, passage_text, raw_title, level, ex1, ex2, ex3, ans, rng
        )
        if not questions:
            errors.append(f"No questions parsed: {title_norm}")
            continue
        passages.append(
            {
                "id": passage_id,
                "seq": seq,
                "level": level,
                "title": _normalize(_norm_title(raw_title)),
                "passage_text": passage_text,
                "questions": questions,
            }
        )

    return {
        "meta": {
            "source": "ELS English Through Reading (2004)",
            "pdf": str(pdf_path.name),
            "passage_starts_in_pdf": len(starts),
            "passage_count": len(passages),
            "question_count": sum(len(p["questions"]) for p in passages),
            "parse_errors": errors,
        },
        "passages": passages,
    }


def _strip_legacy_reading() -> None:
    """Remove hand-ingested reading rows from questions.json and skills.json."""
    skills = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
    skills = [s for s in skills if s.get("category") != "Reading"]
    SKILLS_PATH.write_text(json.dumps(skills, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    bank = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    bank = [q for q in bank if not q["skill_id"].startswith("reading_")]
    QUESTIONS_PATH.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    if not PDF_PATH.is_file():
        print(f"PDF not found: {PDF_PATH}", file=sys.stderr)
        return 1
    data = ingest(PDF_PATH)
    OUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _strip_legacy_reading()
    m = data["meta"]
    print(
        f"Done: {m['passage_count']} passages, {m['question_count']} questions -> {OUT_PATH.name}"
    )
    if m["parse_errors"]:
        print(f"Warnings: {len(m['parse_errors'])} (see meta.parse_errors in JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
