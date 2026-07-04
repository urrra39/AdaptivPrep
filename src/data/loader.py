"""Loaders for static content banks (skills, questions, reading/grammar/vocabulary).

Each bank is kept separate on disk:
  - reading_bank.json   — ELS passages (Ex 1–3)
  - grammar_bank.json   — SAT Writing (Writing Questions.pdf)
  - vocabulary_bank.json — Vocabook Fighting Time MCQs

``questions.json`` retains no grammar/vocabulary items after ingestion.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _PROJECT_ROOT / "data"
SKILLS_PATH = DATA_DIR / "skills.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
READING_BANK_PATH = DATA_DIR / "reading_bank.json"
GRAMMAR_BANK_PATH = DATA_DIR / "grammar_bank.json"
VOCABULARY_BANK_PATH = DATA_DIR / "vocabulary_bank.json"


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
    return list(p["questions"]) if p else []


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
