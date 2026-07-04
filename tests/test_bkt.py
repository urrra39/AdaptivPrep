"""Unit tests for the Bayesian Knowledge Tracing model.

The hand-computed values below pin the exact update equations with the
default parameters (p_init=0.30, p_transit=0.15, p_guess=0.25, p_slip=0.10):

    posterior(0.3, correct)   = 0.3*0.9 / (0.3*0.9 + 0.7*0.25)  = 0.27/0.445
    posterior(0.3, incorrect) = 0.3*0.1 / (0.3*0.1 + 0.7*0.75)  = 0.03/0.555
    update(p, obs)            = posterior + (1-posterior)*0.15
"""
import random

import pytest

from src.data import loader, schema
from src.models.bkt import BKTModel, BKTParams, get_mastery

POST_CORRECT_03 = 0.27 / 0.445            # 0.6067415730...
POST_WRONG_03 = 0.03 / 0.555              # 0.0540540540...
UPDATE_CORRECT_03 = POST_CORRECT_03 + (1 - POST_CORRECT_03) * 0.15
UPDATE_WRONG_03 = POST_WRONG_03 + (1 - POST_WRONG_03) * 0.15


# --------------------------------------------------------------------- #
# Parameter validation                                                   #
# --------------------------------------------------------------------- #
class TestBKTParams:
    def test_defaults_are_valid(self):
        p = BKTParams()
        assert (p.p_init, p.p_transit, p.p_guess, p.p_slip) == (0.30, 0.15, 0.25, 0.10)

    @pytest.mark.parametrize("field", ["p_init", "p_transit", "p_guess", "p_slip"])
    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.7])
    def test_out_of_range_rejected(self, field, bad):
        with pytest.raises(ValueError):
            BKTParams(**{field: bad})

    def test_degenerate_guess_plus_slip_rejected(self):
        with pytest.raises(ValueError):
            BKTParams(p_guess=0.6, p_slip=0.5)


# --------------------------------------------------------------------- #
# Update equations                                                       #
# --------------------------------------------------------------------- #
class TestUpdateEquations:
    model = BKTModel()

    def test_posterior_hand_computed(self):
        assert self.model.posterior(0.3, True) == pytest.approx(POST_CORRECT_03)
        assert self.model.posterior(0.3, False) == pytest.approx(POST_WRONG_03)

    def test_update_hand_computed(self):
        assert self.model.update(0.3, True) == pytest.approx(UPDATE_CORRECT_03)
        assert self.model.update(0.3, False) == pytest.approx(UPDATE_WRONG_03)

    def test_evidence_moves_in_right_direction(self):
        # The pure Bayes step must raise mastery on correct, lower it on wrong.
        for p in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]:
            assert self.model.posterior(p, True) > p
            assert self.model.posterior(p, False) < p

    def test_mastery_stays_in_unit_interval(self):
        rng = random.Random(42)
        p = BKTParams().p_init
        for _ in range(500):
            p = self.model.update(p, rng.random() < 0.5)
            assert 0.0 <= p <= 1.0

    def test_long_correct_streak_converges_high(self):
        assert self.model.mastery_from_sequence([True] * 25) > 0.99

    def test_long_wrong_streak_stays_low(self):
        # All-wrong converges to the transit-driven fixed point (~0.17 with
        # defaults), far below certainty and below the prior's update path.
        assert self.model.mastery_from_sequence([False] * 25) < 0.2

    def test_trajectory_matches_sequence_replay(self):
        obs = [True, False, True, True, False, True]
        traj = self.model.trajectory(obs)
        assert len(traj) == len(obs)
        assert traj[-1] == pytest.approx(self.model.mastery_from_sequence(obs))
        assert traj[0] == pytest.approx(UPDATE_CORRECT_03)

    def test_predict_correct_bounds_and_values(self):
        m = self.model
        # p(1-s) + (1-p)g: bounded by [g, 1-s], with pinned endpoints/middle.
        assert m.predict_correct(0.0) == pytest.approx(0.25)
        assert m.predict_correct(1.0) == pytest.approx(0.90)
        assert m.predict_correct(0.3) == pytest.approx(0.445)
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert 0.25 <= m.predict_correct(p) <= 0.90


# --------------------------------------------------------------------- #
# Per-skill parameter overrides                                          #
# --------------------------------------------------------------------- #
class TestPerSkillParams:
    def test_override_and_fallback(self):
        special = BKTParams(p_init=0.5, p_transit=0.2, p_guess=0.2, p_slip=0.05)
        model = BKTModel(per_skill_params={"grammar_articles": special})
        assert model.params_for("grammar_articles") is special
        assert model.params_for("vocab_health") is model.default_params
        assert model.params_for(None) is model.default_params


# --------------------------------------------------------------------- #
# get_mastery integration (temp SQLite DB)                                #
# --------------------------------------------------------------------- #
class TestGetMastery:
    @pytest.fixture()
    def db(self, tmp_path):
        path = tmp_path / "test.db"
        schema.init_db(path)
        return path

    def test_new_user_sits_at_prior_for_all_skills(self, db):
        user = schema.get_or_create_user("newbie", db_path=db)
        mastery = get_mastery(user, db_path=db)
        assert set(mastery) == set(loader.skill_ids())
        assert all(v == pytest.approx(BKTParams().p_init) for v in mastery.values())

    def test_history_moves_mastery_and_matches_manual_replay(self, db):
        user = schema.get_or_create_user("worker", db_path=db)
        strong, weak, untouched = (
            "vocab_college_panda_set_01",
            "sat_transitions",
            "vocab_college_panda_set_02",
        )
        for correct_seq, skill in [((True, True, True), strong),
                                   ((False, False, False), weak)]:
            for i, correct in enumerate(correct_seq):
                schema.record_response(user, f"{skill}_q{i+1}", skill, correct, db_path=db)

        mastery = get_mastery(user, db_path=db)
        model = BKTModel()
        prior = BKTParams().p_init
        assert mastery[strong] == pytest.approx(model.mastery_from_sequence([True] * 3))
        assert mastery[weak] == pytest.approx(model.mastery_from_sequence([False] * 3))
        assert mastery[strong] > prior > mastery[weak]
        assert mastery[untouched] == pytest.approx(prior)

    def test_unknown_skill_rows_are_ignored(self, db):
        user = schema.get_or_create_user("ghost", db_path=db)
        schema.record_response(user, "old_q1", "skill_removed_in_v2", True, db_path=db)
        mastery = get_mastery(user, db_path=db)
        assert "skill_removed_in_v2" not in mastery
        assert set(mastery) == set(loader.skill_ids())
