"""Integrity checks for vocabulary_bank.json (Vocabook.pdf)."""
from src.data import loader


def test_vocabulary_bank_loaded():
    bank = loader.load_vocabulary_bank()
    assert bank["meta"]["question_count"] == 200
    assert bank["meta"]["parse_errors"] == []


def test_vocabulary_questions_have_blanks_and_four_options():
    for q in loader.load_vocabulary_bank()["questions"]:
        assert "______" in q["question_text"]
        assert len(q["options"]) == 4
        assert 0 <= q["correct_answer"] <= 3
        assert q.get("bank") == "vocabulary"


def test_vocabulary_caption_not_reading_exercise():
    from src.app.quiz_app import quiz_caption_details

    q = loader.load_vocabulary_bank()["questions"][0]
    _title, detail = quiz_caption_details(q)
    assert "Exercise" not in detail
    assert detail == "Lug'at"


def test_vocabulary_bank_separate_from_reading_and_grammar():
    v_ids = set(loader.vocabulary_question_ids())
    r_ids = set(loader.questions_by_id()) & v_ids
    for qid in v_ids:
        q = loader.get_question(qid)
        assert "passage_id" not in q
        assert q.get("bank") == "vocabulary"
    g_ids = set(loader.grammar_question_ids())
    assert not v_ids & g_ids
