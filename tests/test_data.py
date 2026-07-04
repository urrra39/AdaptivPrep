"""Integrity tests for content banks (skills + grammar/vocabulary/legacy questions)."""
import collections

from src.data import loader

SKILLS = loader.load_skills()
LEGACY = loader.load_questions()
GRAMMAR = loader.load_grammar_bank()["questions"]
VOCAB = loader.load_vocabulary_bank()["questions"]


class TestSkills:
    def test_minimum_count(self):
        assert len(SKILLS) >= 15

    def test_required_fields(self):
        for s in SKILLS:
            assert set(s) >= {"id", "name", "category"}
            assert s["id"] and s["name"] and s["category"]

    def test_ids_unique(self):
        ids = [s["id"] for s in SKILLS]
        assert len(ids) == len(set(ids))


class TestDedicatedBanks:
    def test_grammar_and_vocabulary_counts(self):
        assert len(GRAMMAR) == 50
        assert len(VOCAB) == 200

    def test_bank_ids_unique_and_separate(self):
        g_ids = {q["id"] for q in GRAMMAR}
        v_ids = {q["id"] for q in VOCAB}
        assert len(g_ids) == len(GRAMMAR)
        assert len(v_ids) == len(VOCAB)
        assert not g_ids & v_ids

    def test_every_bank_question_valid(self):
        skill_ids = {s["id"] for s in SKILLS}
        for q in GRAMMAR + VOCAB:
            assert set(q) >= {
                "id",
                "skill_id",
                "question_text",
                "options",
                "correct_answer",
                "difficulty",
            }, q["id"]
            assert q["skill_id"] in skill_ids, q["id"]
            assert len(q["options"]) == 4, q["id"]
            assert 0 <= q["correct_answer"] < 4, q["id"]


class TestLegacyQuestions:
    def test_legacy_items_still_valid_if_present(self):
        skill_ids = {s["id"] for s in SKILLS}
        for q in LEGACY:
            assert q["skill_id"] in skill_ids, q["id"]
            assert len(q["options"]) in (4, 5), q["id"]
            assert 0 <= q["correct_answer"] < len(q["options"]), q["id"]

    def test_bank_skills_have_questions(self):
        per_skill = collections.Counter(q["skill_id"] for q in GRAMMAR + VOCAB)
        for sid in loader.load_grammar_bank()["skills"] + loader.load_vocabulary_bank()["skills"]:
            assert per_skill[sid] >= 1, sid
