"""Guarantee no source-name or OCR junk ever reaches a user-visible field.

Scans EVERY usable question's prompt + options across all three banks after the
display sanitizer runs. If future ingestion reintroduces junk, this test fails.
"""
import json
import re
from pathlib import Path

import pytest

from src.data import loader

DATA = Path(__file__).resolve().parents[1] / "data"

# The material name that leaked before ("ELS"), plus raw OCR junk fingerprints.
FORBIDDEN = re.compile(
    r"\bELS\b"                       # book/source name
    r"|[\ufffd\u2022\u00b7\^»«]"     # replacement char, bullets, stray marks
    r"|::+"                          # double-colon garbage
    r"|[a-z][A-Z][a-z]*[A-Z]"        # fused mixed-case OCR ("uEgLhtSs")
)


def _reading_prompts_and_options():
    bank = json.loads((DATA / "reading_bank.json").read_text(encoding="utf-8"))
    for p in bank["passages"]:
        for q in p.get("questions", []):
            if not loader._usable_reading_question(q):
                continue
            yield q.get("id", "?"), "prompt", q.get("question_text", "")
            for opt in q.get("options", []):
                yield q.get("id", "?"), "option", opt


def _bank_prompts_and_options(filename):
    bank = json.loads((DATA / filename).read_text(encoding="utf-8"))
    for q in bank["questions"]:
        yield q.get("id", "?"), "prompt", q.get("question_text", "")
        for opt in q.get("options", []):
            yield q.get("id", "?"), "option", opt


def _all_fields():
    yield from _reading_prompts_and_options()
    yield from _bank_prompts_and_options("grammar_bank.json")
    yield from _bank_prompts_and_options("vocabulary_bank.json")


def test_no_source_leaks_after_sanitize():
    offenders = []
    for qid, field, raw in _all_fields():
        cleaned = loader.clean_display_prompt(raw or "")
        if FORBIDDEN.search(cleaned):
            offenders.append(f"{qid}.{field}: {cleaned!r}")
    assert not offenders, "Source/junk leak reached a visible field:\n" + "\n".join(offenders[:20])
