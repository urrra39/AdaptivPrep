"""Passage text cleanup for display."""
from src.data import loader


def test_clean_passage_text_normalizes_whitespace():
    raw = "Hello   world.  This  is   a   test."
    cleaned = loader.clean_passage_text(raw)
    assert cleaned == "Hello world. This is a test."


def test_clean_passage_text_preserves_pronouns_and_numbers():
    raw = (
        "when I a joke is retold Though he 1 may not admit goes to 1 extremes "
        "as a I truly humorous a really I keen appreciative 1 of humour"
    )
    cleaned = loader.clean_passage_text(raw)
    assert " I " in cleaned, "Should preserve pronoun 'I'"
    assert " 1 " in cleaned, "Should preserve numeral '1'"
