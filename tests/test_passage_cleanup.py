"""Passage text cleanup for display."""
from src.data import loader


def test_clean_passage_text_removes_blank_markers():
    raw = (
        "when I a joke is retold Though he 1 may not admit goes to 1 extremes "
        "as a I truly humorous a really I keen appreciative 1 of humour"
    )
    cleaned = loader.clean_passage_text(raw)
    assert " I a " not in cleaned
    assert " 1 " not in cleaned
    assert "when a joke" in cleaned
