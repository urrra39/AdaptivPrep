"""Loaders for static content banks (skills, questions, reading/grammar/vocabulary).

Each bank is kept separate on disk:
  - reading_bank.json   — ELS passages (Ex 1–3)
  - grammar_bank.json   — SAT Writing (Writing Questions.pdf)
  - vocabulary_bank.json — Vocabook Fighting Time MCQs

``questions.json`` retains no grammar/vocabulary items after ingestion.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _PROJECT_ROOT / "data"
SKILLS_PATH = DATA_DIR / "skills.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
READING_BANK_PATH = DATA_DIR / "reading_bank.json"
GRAMMAR_BANK_PATH = DATA_DIR / "grammar_bank.json"
VOCABULARY_BANK_PATH = DATA_DIR / "vocabulary_bank.json"

_EXercise_PROMPT_RE = re.compile(r"^\[Exercise \d+\]\s*", re.I)

# Junk fingerprints that make a question unusable (options must be real English words).
_OPTION_JUNK_PATTERNS = [
    re.compile(r"[\^<>»«]"),                          # stray typographic marks
    re.compile(r"::+"),                                # double-colon garbage
    re.compile(r"[a-z]{2}i{3,}", re.I),                # attiiua-like triple-i runs
    re.compile(r"[bcdfghjklmnpqrstvwxz]{5,}", re.I),   # 5+ consonants in a row
    re.compile(r"\bmmjfmmm\b", re.I),
]

# Source/page leakage suffixes that appear inside ELS Exercise-1 prompts.
_SOURCE_LEAK_PATTERNS = [
    re.compile(r"\s*\d{1,3}\s*[•·-]?\s*ELS\b.*$", re.I),        # "52 • ELS ::mmjfmmm 11»"
    re.compile(r"\s*\bELS\b\s*.*$", re.I),                       # bare "ELS ..." tail
    re.compile(r"\s*::+.*$"),                                    # trailing "::junk"
    re.compile(r"\s*[»«].*$"),                                   # trailing »« junk
    re.compile(r"\s*\^\s*.*$"),                                  # trailing "^ in turn"
]


def _has_option_junk(text: str) -> bool:
    if not text:
        return False
    for pat in _OPTION_JUNK_PATTERNS:
        if pat.search(text):
            return True
    return False


def clean_display_prompt(text: str) -> str:
    """Strip book/page leakage (ELS, page nums, stray marks) from a prompt before display."""
    if not text:
        return text
    cleaned = text
    for pat in _SOURCE_LEAK_PATTERNS:
        cleaned = pat.sub("", cleaned)
    # Remove OCR replacement chars (U+FFFD) and stray bullet/interpunct marks anywhere.
    cleaned = re.sub(r"[\ufffd\u2022\u00b7]+", " ", cleaned)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)             # ". ." / ".." -> "."
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)       # space before punctuation
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def clean_passage_text(text: str) -> str:
    """Return passage text with only whitespace normalization.

    The old regex-based blank-marker stripping was too aggressive and removed
    legitimate "I" pronouns and "1" numerals from 21+ passages. Now we only
    collapse redundant whitespace.
    """
    if not text:
        return text
    cleaned = re.sub(r"\s{2,}", " ", text)
    return cleaned.strip()


def passage_display_text(passage_id: str) -> str:
    p = _reading_passages_by_id().get(passage_id)
    if not p:
        return ""
    return clean_passage_text(p.get("passage_text") or "")


_INNER_CAPS_RE = re.compile(r"[a-z][A-Z][a-z]*[A-Z]")   # "uEgLhtSs" fused OCR garble
_DIGIT_FUSED_RE = re.compile(r"\b\d+[a-z]{2,}\b", re.I)   # "174thoughts" page-num fused into a word


def is_garbled_prompt(text: str) -> bool:
    """True when the Exercise-1 definition body is unreadable even after cleaning."""
    body = _EXercise_PROMPT_RE.sub("", clean_display_prompt(text or ""))
    if len(body) < 3:
        return True
    if "))" in body:
        return True
    if "\ufffd" in body:
        return True
    if _INNER_CAPS_RE.search(body):
        return True
    if _DIGIT_FUSED_RE.search(body):
        return True
    words = body.split()
    if words and max(len(w) for w in words) > 35:
        return True
    return False


def _usable_reading_question(q: dict) -> bool:
    """Reject Ex-1 items where the prompt is unreadable or any option carries OCR junk."""
    if q.get("exercise") == 1:
        if is_garbled_prompt(q.get("question_text", "")):
            return False
        for opt in q.get("options", []):
            if _has_option_junk(opt or ""):
                return False
    return True


@lru_cache(maxsize=1)
def load_skills() -> list:
    """Return grammar/vocabulary skills (reading passages are dynamic)."""
    with open(SKILLS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_questions() -> list:
    """Legacy/misc questions only (grammar/vocab live in dedicated banks)."""
    if not QUESTIONS_PATH.exists():
        return []
    with open(QUESTIONS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_reading_bank() -> dict:
    with open(READING_BANK_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_grammar_bank() -> dict:
    with open(GRAMMAR_BANK_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_vocabulary_bank() -> dict:
    with open(VOCABULARY_BANK_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _reading_passages_by_id() -> dict:
    return {p["id"]: p for p in load_reading_bank()["passages"]}


@lru_cache(maxsize=1)
def _reading_questions_by_id() -> dict:
    out = {}
    for p in load_reading_bank()["passages"]:
        for q in p["questions"]:
            out[q["id"]] = q
    return out


@lru_cache(maxsize=1)
def _grammar_questions_by_id() -> dict:
    return {q["id"]: q for q in load_grammar_bank()["questions"]}


@lru_cache(maxsize=1)
def _vocabulary_questions_by_id() -> dict:
    return {q["id"]: q for q in load_vocabulary_bank()["questions"]}


def reading_passage_ids() -> list:
    return list(_reading_passages_by_id().keys())


def grammar_question_ids() -> list:
    return list(_grammar_questions_by_id().keys())


def vocabulary_question_ids() -> list:
    return list(_vocabulary_questions_by_id().keys())


def get_reading_passage(passage_id: str) -> dict | None:
    return _reading_passages_by_id().get(passage_id)


def questions_for_passage_id(passage_id: str) -> list:
    p = _reading_passages_by_id().get(passage_id)
    if not p:
        return []
    return [q for q in p["questions"] if _usable_reading_question(q)]


def all_questions_for_passage_id(passage_id: str) -> list:
    p = _reading_passages_by_id().get(passage_id)
    if not p:
        return []
    return list(p["questions"])


def reading_passage_count() -> int:
    return len(_reading_passages_by_id())


def reading_bank_available() -> bool:
    """True when the ELS corpus file exists and has at least one passage."""
    if not READING_BANK_PATH.exists():
        return False
    try:
        return reading_passage_count() > 0
    except (OSError, json.JSONDecodeError, KeyError):
        return False


def grammar_question_count() -> int:
    return len(_grammar_questions_by_id())


def vocabulary_question_count() -> int:
    return len(_vocabulary_questions_by_id())


@lru_cache(maxsize=1)
def skills_by_id() -> dict:
    return {s["id"]: s for s in load_skills()}


@lru_cache(maxsize=1)
def questions_by_id() -> dict:
    base = {q["id"]: q for q in load_questions()}
    base.update(_reading_questions_by_id())
    base.update(_grammar_questions_by_id())
    base.update(_vocabulary_questions_by_id())
    return base


def skill_ids() -> list:
    """Grammar/vocabulary skills tracked for bandit/mastery (not reading passages)."""
    return [s["id"] for s in load_skills()]


def get_skill(skill_id: str) -> dict:
    if skill_id in skills_by_id():
        return skills_by_id()[skill_id]
    p = _reading_passages_by_id().get(skill_id)
    if p:
        return {"id": skill_id, "name": f"Reading: {p['title']}", "category": "Reading"}
    raise KeyError(skill_id)


def get_question(question_id: str) -> dict:
    return questions_by_id()[question_id]


def questions_for_skill(skill_id: str) -> list:
    if skill_id in _reading_passages_by_id():
        return questions_for_passage_id(skill_id)
    g = [q for q in load_grammar_bank()["questions"] if q["skill_id"] == skill_id]
    if g:
        return g
    v = [q for q in load_vocabulary_bank()["questions"] if q["skill_id"] == skill_id]
    if v:
        return v
    return [q for q in load_questions() if q.get("skill_id") == skill_id]


def skill_name(skill_id: str) -> str:
    try:
        return get_skill(skill_id)["name"]
    except KeyError:
        return skill_id


def display_skill_name(skill_id: str) -> str:
    """User-facing skill label — never exposes book/source names."""
    try:
        cat = get_skill(skill_id)["category"]
    except KeyError:
        return skill_id
    return {"Reading": "READING", "Grammar": "Grammatika", "Vocabulary": "Lug'at"}.get(
        cat, cat
    )


def display_category(skill_id: str) -> str:
    try:
        return get_skill(skill_id)["category"]
    except KeyError:
        return "Other"


def clear_caches() -> None:
    """Test helper: drop LRU caches after bank files change on disk."""
    for fn in (
        load_skills,
        load_questions,
        load_reading_bank,
        load_grammar_bank,
        load_vocabulary_bank,
        _reading_passages_by_id,
        _reading_questions_by_id,
        _grammar_questions_by_id,
        _vocabulary_questions_by_id,
        skills_by_id,
        questions_by_id,
    ):
        fn.cache_clear()
