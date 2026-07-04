"""Integrity checks for grammar_bank.json (SAT Writing Questions.pdf)."""
from src.data import loader


def test_grammar_bank_loaded():
    bank = loader.load_grammar_bank()
    assert bank["meta"]["question_count"] == 20
    assert bank["meta"]["parse_errors"] == []


def test_grammar_questions_have_four_options_and_valid_answer():
    for q in loader.load_grammar_bank()["questions"]:
        assert len(q["options"]) == 4
        assert 0 <= q["correct_answer"] <= 3
        assert q.get("bank") == "grammar"
        assert "passage_id" not in q
        letter = chr(ord("A") + q["correct_answer"])
        assert q.get("correct_letter") == letter


def test_grammar_caption_not_reading_exercise():
    from src.app.quiz_app import quiz_caption_details

    q = loader.load_grammar_bank()["questions"][0]
    _title, detail = quiz_caption_details(q)
    assert "Exercise" not in detail
    assert detail == "Grammatika"


def test_grammar_bank_separate_from_vocabulary():
    g_ids = set(loader.grammar_question_ids())
    v_ids = set(loader.vocabulary_question_ids())
    assert not g_ids & v_ids
