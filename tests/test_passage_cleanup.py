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


def test_clean_display_prompt_strips_els_source_leak():
    raw = "[Exercise 1] in imagination; in memory (phrase) 82 \u2022 ELS"
    cleaned = loader.clean_display_prompt(raw)
    assert "ELS" not in cleaned
    assert "82" not in cleaned
    assert cleaned == "[Exercise 1] in imagination; in memory (phrase)"


def test_clean_display_prompt_removes_stray_bullets_and_replacement_chars():
    raw = "[Exercise 3] \u2022 1. We now know \ufffd there are four types. ______"
    cleaned = loader.clean_display_prompt(raw)
    assert "\u2022" not in cleaned
    assert "\ufffd" not in cleaned
    assert "1. We now know" in cleaned


def test_is_garbled_prompt_flags_fused_ocr():
    # page number fused into "thoughts" with interleaved ELS caps
    raw = "[Exercise 1] examining one's own 174t h\ufffdo uEgLhtSs, ideas and feelings"
    assert loader.is_garbled_prompt(raw) is True


def test_is_garbled_prompt_accepts_clean_definition():
    assert loader.is_garbled_prompt("[Exercise 1] rubbish; waste material") is False
