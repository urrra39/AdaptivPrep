"""Quota-engine and anti-repetition invariants (pure logic, no Streamlit UI)."""
import random
from collections import Counter

from src.app import quiz_app
from src.data import loader
from src.models.bkt import BKTParams

PHASE_ORDER = quiz_app.PHASE_ORDER
READING_N = quiz_app.READING_PASSAGES_PER_SESSION


def _bucket_of(question: dict) -> str:
    return quiz_app.quota_bucket(loader.get_skill(question["skill_id"])["category"])


def _session_for_drain(rng: random.Random) -> dict:
    pool = loader.reading_passage_ids()
    eligible = [
        pid
        for pid in pool
        if len(loader.all_questions_for_passage_id(pid)) >= quiz_app.READING_QUESTIONS_PER_PASSAGE
    ]
    source = eligible if len(eligible) >= READING_N else pool
    pids = rng.sample(source, min(READING_N, len(source)))
    reading_ids: list[str] = []
    for pid in pids:
        reading_ids.extend(quiz_app.pick_reading_question_ids(pid))
    rq = len(reading_ids)
    g_pool = loader.grammar_question_ids()
    g_n = min(quiz_app.QUOTAS["Grammar"], len(g_pool))
    g_ids = rng.sample(g_pool, g_n)
    v_pool = loader.vocabulary_question_ids()
    v_n = min(quiz_app.QUOTAS["Vocabulary"], len(v_pool))
    v_ids = rng.sample(v_pool, v_n)
    return {
        "reading_passage_ids": pids,
        "reading_question_ids": reading_ids,
        "reading_order": reading_ids.copy(),
        "session_reading_quota": rq,
        "grammar_question_ids": g_ids,
        "grammar_order": g_ids.copy(),
        "session_grammar_quota": g_n,
        "vocabulary_question_ids": v_ids,
        "vocabulary_order": v_ids.copy(),
        "session_vocabulary_quota": v_n,
    }


def drain() -> list:
    mastery = {sid: BKTParams().p_init for sid in loader.skill_ids()}
    rng = random.Random(99)
    session = _session_for_drain(rng)
    seen, used, served = set(), {b: 0 for b in PHASE_ORDER}, []
    while True:
        q = quiz_app.select_next_question(mastery, rng, seen, used, session)
        if q is None:
            return served
        served.append(q)
        seen.add(q["id"])
        used[_bucket_of(q)] += 1


def test_no_repeats_and_session_length_is_min_quota_supply():
    served = drain()
    ids = [q["id"] for q in served]
    assert len(ids) == len(set(ids))
    per_bucket = Counter(_bucket_of(q) for q in served)
    rng = random.Random(99)
    session = _session_for_drain(rng)
    supply = Counter(
        quiz_app.quota_bucket(s["category"])
        for s in loader.load_skills()
        for _ in loader.questions_for_skill(s["id"])
    )
    supply["Reading"] = session["session_reading_quota"]
    supply["Grammar"] = session["session_grammar_quota"]
    supply["Vocabulary"] = session["session_vocabulary_quota"]
    for bucket in PHASE_ORDER:
        cap = (
            session["session_reading_quota"]
            if bucket == "Reading"
            else session.get(f"session_{bucket.lower()}_quota", quiz_app.QUOTAS[bucket])
        )
        if bucket == "Reading":
            assert per_bucket[bucket] == cap, bucket
        else:
            assert per_bucket[bucket] == min(cap, supply[bucket]), bucket


def test_reading_bank_has_enough_passages():
    assert loader.reading_passage_count() >= READING_N


def test_phases_served_in_reading_grammar_vocabulary_order():
    served = drain()
    phases = [_bucket_of(q) for q in served]
    for i in range(len(phases) - 1):
        assert PHASE_ORDER.index(phases[i]) <= PHASE_ORDER.index(phases[i + 1])


def test_pick_reading_ten_per_passage():
    pid = loader.reading_passage_ids()[0]
    ids = quiz_app.pick_reading_question_ids(pid)
    assert len(ids) == quiz_app.READING_QUESTIONS_PER_PASSAGE


def test_reading_caption_shows_paragraph_and_exercise():
    from src.app.quiz_app import quiz_caption_details

    pids = loader.reading_passage_ids()[:3]
    q = loader.questions_for_passage_id(pids[0])[0]
    title, detail = quiz_caption_details(q, pids)
    assert title == "READING"
    assert "Paragraph 1/3" in detail
    assert "Exercise 1" in detail


def test_reading_exercise_order_within_passage():
    served = drain()
    reading = [q for q in served if _bucket_of(q) == "Reading"]
    by_passage: dict = {}
    for q in reading:
        by_passage.setdefault(q["passage_id"], []).append(q)
    for pid, qs in by_passage.items():
        exercises = [q["exercise"] for q in qs]
        assert exercises == sorted(exercises)


def test_quota_ceiling_binds_when_supply_exceeds_it(monkeypatch):
    monkeypatch.setattr(
        quiz_app, "QUOTAS", {"Reading": 0, "Grammar": 3, "Vocabulary": 4}
    )
    monkeypatch.setattr(quiz_app, "READING_PASSAGES_PER_SESSION", 2)

    def _tiny_session(rng):
        pids = loader.reading_passage_ids()[:2]
        reading_ids = []
        for pid in pids:
            reading_ids.extend(quiz_app.pick_reading_question_ids(pid, n=2))
        g_ids = loader.grammar_question_ids()[:3]
        v_ids = loader.vocabulary_question_ids()[:4]
        return {
            "reading_passage_ids": pids,
            "reading_question_ids": reading_ids,
            "reading_order": reading_ids.copy(),
            "session_reading_quota": len(reading_ids),
            "grammar_question_ids": g_ids,
            "grammar_order": g_ids.copy(),
            "session_grammar_quota": 3,
            "vocabulary_question_ids": v_ids,
            "vocabulary_order": v_ids.copy(),
            "session_vocabulary_quota": 4,
        }

    mastery = {sid: BKTParams().p_init for sid in loader.skill_ids()}
    rng = random.Random(1)
    session = _tiny_session(rng)
    seen, used, served = set(), {b: 0 for b in PHASE_ORDER}, []
    while True:
        q = quiz_app.select_next_question(mastery, rng, seen, used, session)
        if q is None:
            break
        served.append(q)
        seen.add(q["id"])
        used[_bucket_of(q)] += 1
    r_q = session["session_reading_quota"]
    counts = Counter(_bucket_of(q) for q in served)
    assert counts["Grammar"] == 3
    assert counts["Vocabulary"] == 4
    assert counts["Reading"] == r_q
