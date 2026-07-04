"""Integrity checks for the ELS reading_bank.json corpus."""
from collections import Counter

from src.data import loader


def test_reading_bank_available():
    bank = loader.load_reading_bank()
    assert loader.reading_bank_available()
    assert loader.reading_passage_count() >= 40
    assert bank["meta"]["passage_count"] == 199
    assert bank["meta"]["question_count"] >= 3000
    assert bank["meta"]["parse_errors"] == []


def test_every_passage_has_all_three_exercises():
    bank = loader.load_reading_bank()
    for p in bank["passages"]:
        ex = set(q["exercise"] for q in p["questions"])
        assert ex == {1, 2, 3}, f"{p['id']} missing { {1,2,3} - ex }"


def test_reading_questions_have_passage_text():
    p = loader.load_reading_bank()["passages"][0]
    for q in p["questions"]:
        assert q.get("passage_text"), q["id"]
        assert q.get("exercise") in (1, 2, 3)
