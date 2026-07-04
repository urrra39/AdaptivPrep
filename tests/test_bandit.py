"""Unit tests for the epsilon-greedy recommendation policy.

All stochastic tests inject a seeded ``random.Random`` so they are exactly
reproducible; frequency bounds are wide enough to be robust yet tight
enough to catch a broken mixture rate.
"""
import random

import pytest

from src.models.bandit import EpsilonGreedyBandit, ThompsonSamplingBandit

ARMS = ["s1", "s2", "s3", "s4", "s5"]


def make(epsilon: float, seed: int = 7) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(ARMS, epsilon=epsilon, rng=random.Random(seed))


class TestConstruction:
    def test_rejects_empty_arms(self):
        with pytest.raises(ValueError):
            EpsilonGreedyBandit([])

    def test_rejects_duplicate_arms(self):
        with pytest.raises(ValueError):
            EpsilonGreedyBandit(["a", "b", "a"])

    @pytest.mark.parametrize("eps", [-0.1, 1.1, 5.0])
    def test_rejects_out_of_range_epsilon(self, eps):
        with pytest.raises(ValueError):
            EpsilonGreedyBandit(ARMS, epsilon=eps)

    @pytest.mark.parametrize("eps", [0.0, 0.15, 1.0])
    def test_accepts_boundary_epsilon(self, eps):
        assert EpsilonGreedyBandit(ARMS, epsilon=eps).epsilon == eps


class TestRewards:
    def test_reward_is_one_minus_mastery(self):
        b = make(0.0)
        rewards = b.rewards_from_mastery(
            {"s1": 0.9, "s2": 0.3, "s3": 0.0, "s4": 1.0, "s5": 0.5}
        )
        assert rewards["s1"] == pytest.approx(0.1)
        assert rewards["s3"] == pytest.approx(1.0)
        assert rewards["s4"] == pytest.approx(0.0)

    def test_mastery_clamped_to_unit_interval(self):
        b = make(0.0)
        assert b.reward({"s1": 1.4}, "s1") == 0.0
        assert b.reward({"s1": -0.4}, "s1") == 1.0

    def test_missing_skill_treated_as_zero_mastery(self):
        assert make(0.0).reward({}, "s1") == 1.0


class TestSelection:
    def test_pure_exploit_picks_unique_weakest_skill(self):
        b = make(0.0)
        mastery = {"s1": 0.9, "s2": 0.2, "s3": 0.8, "s4": 0.95, "s5": 0.6}
        assert all(b.select_skill(mastery) == "s2" for _ in range(50))

    def test_exploit_breaks_cold_start_ties_across_all_arms(self):
        b = make(0.0)
        mastery = {sid: 0.3 for sid in ARMS}  # cold start: every arm tied
        seen = {b.select_skill(mastery) for _ in range(300)}
        assert seen == set(ARMS)

    def test_exploit_ties_restricted_to_the_top_set(self):
        b = make(0.0)
        mastery = {"s1": 0.1, "s2": 0.1, "s3": 0.9, "s4": 0.9, "s5": 0.9}
        seen = {b.select_skill(mastery) for _ in range(300)}
        assert seen == {"s1", "s2"}

    def test_pure_explore_reaches_every_arm(self):
        b = make(1.0)
        mastery = {"s1": 0.0, "s2": 1.0, "s3": 1.0, "s4": 1.0, "s5": 1.0}
        seen = {b.select_skill(mastery) for _ in range(300)}
        assert seen == set(ARMS)  # even fully mastered arms still explored

    def test_mixture_rate_close_to_expected(self):
        # eps=0.4 with a unique weakest arm:
        # P(weakest) = (1 - eps) + eps/K = 0.6 + 0.4/5 = 0.68
        b = make(0.4, seed=123)
        mastery = {"s1": 0.1, "s2": 0.8, "s3": 0.8, "s4": 0.8, "s5": 0.8}
        n = 4000
        hit_rate = sum(b.select_skill(mastery) == "s1" for _ in range(n)) / n
        assert 0.63 <= hit_rate <= 0.73

    def test_rank_skills_weakest_first(self):
        b = make(0.0)
        mastery = {"s1": 0.9, "s2": 0.2, "s3": 0.8, "s4": 0.95, "s5": 0.6}
        assert b.rank_skills(mastery) == ["s2", "s5", "s3", "s1", "s4"]


class TestThompsonSampling:
    def make_ts(self, seed: int = 7) -> ThompsonSamplingBandit:
        return ThompsonSamplingBandit(ARMS, rng=random.Random(seed))

    def test_shares_base_validation(self):
        with pytest.raises(ValueError):
            ThompsonSamplingBandit([])
        with pytest.raises(ValueError):
            ThompsonSamplingBandit(["a", "b", "a"])

    def test_always_returns_a_valid_arm(self):
        b = self.make_ts()
        mastery = {"s1": 0.3, "s2": 0.7, "s3": 1.0, "s4": 0.0, "s5": 0.5}
        assert all(b.select_skill(mastery) in set(ARMS) for _ in range(200))

    def test_certainly_unlearned_arm_always_chosen(self):
        # mastery 0 -> sampled unlearned w.p. 1; mastery 1 -> w.p. 0.
        b = self.make_ts()
        mastery = {"s1": 1.0, "s2": 1.0, "s3": 0.0, "s4": 1.0, "s5": 1.0}
        assert all(b.select_skill(mastery) == "s3" for _ in range(200))

    def test_probability_matching_rate_two_arms(self):
        # a: mastery .2 (unlearned w.p. .8), b: .8 (unlearned w.p. .2).
        # P(pick b) = P(only b unlearned) + P(both)/2 = .04 + .16/2 = .12
        # (learned-learned rounds fall back to greedy -> a).
        b = ThompsonSamplingBandit(["a", "b"], rng=random.Random(11))
        mastery = {"a": 0.2, "b": 0.8}
        n = 4000
        freq_b = sum(b.select_skill(mastery) == "b" for _ in range(n)) / n
        assert 0.09 <= freq_b <= 0.15

    def test_exploration_scales_with_uncertainty(self):
        # Probe rate must order by uncertainty: s1 (60%) > s2 (90%) > 0;
        # fully-mastered arms are never picked (fallback greedy prefers s1).
        b = self.make_ts(seed=3)
        mastery = {"s1": 0.6, "s2": 0.9, "s3": 1.0, "s4": 1.0, "s5": 1.0}
        picks = [b.select_skill(mastery) for _ in range(3000)]
        assert picks.count("s1") > picks.count("s2") > 0
        assert set(picks) == {"s1", "s2"}

    def test_all_mastered_world_falls_back_to_weakest(self):
        b = self.make_ts()
        mastery = {"s1": 0.99, "s2": 1.0, "s3": 1.0, "s4": 1.0, "s5": 1.0}
        assert all(b.select_skill(mastery) == "s1" for _ in range(300))

    def test_inherits_reward_model_and_ranking(self):
        b = self.make_ts()
        mastery = {"s1": 0.9, "s2": 0.2, "s3": 0.8, "s4": 0.95, "s5": 0.6}
        assert b.rank_skills(mastery) == ["s2", "s5", "s3", "s1", "s4"]
