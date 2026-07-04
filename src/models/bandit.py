"""Recommendation Layer - bandit policies over skills.

Before every question the tutor must decide which skill the learner should
practise next.  We frame this as a multi-armed bandit: each skill is an arm,
and pulling an arm (asking one question from that skill) yields learning
value proportional to how far the learner is from mastering it:

    reward(skill) = 1 - P(mastery of skill)        # BKT posterior, live

**The exploration-exploitation dilemma.**  A purely greedy policy would
always drill the single weakest skill.  That fails in two ways: (i) the
mastery estimates it trusts are themselves noisy - lucky guesses and
careless slips can make a known skill look weak or vice versa - and
estimates for skills that are never revisited go stale; (ii) pedagogically,
learners need varied, spaced practice, not thirty consecutive questions on
one topic.  A purely random policy (the Phase 2 behaviour) has the opposite
failure: it wastes most questions on already-mastered material.

Two policies are implemented behind the same ``select_skill`` interface:

* :class:`EpsilonGreedyBandit` - the simplest principled compromise: with
  probability 1-epsilon exploit, with probability epsilon explore uniformly.
* :class:`ThompsonSamplingBandit` (Phase 10) - posterior sampling, where
  exploration is allocated by *uncertainty* rather than a flat constant.

Why bandits and not full deep RL: at cold start per-user data is sparse;
bandit policies are statistically stable from the first interaction,
whereas value-based RL needs many episodes before its policy is
trustworthy.  UCB is the remaining natural upgrade and slots in behind the
same interface.
"""
from __future__ import annotations

import random
from typing import List, Mapping, Optional, Sequence


class SkillBandit:
    """Shared machinery: a fixed arm set and the mastery-derived reward model.

    Policies here are *state-informed* rather than classical: arm values are
    not running averages of sampled rewards but are recomputed from the live
    BKT mastery state on every call, so every policy always acts on the
    freshest available estimate of the learner.  Subclasses implement only
    the selection rule (:meth:`select_skill`).
    """

    def __init__(
        self,
        skill_ids: Sequence[str],
        rng: Optional[random.Random] = None,
    ) -> None:
        self.skill_ids: List[str] = list(skill_ids)
        if not self.skill_ids:
            raise ValueError("bandit needs at least one skill arm")
        if len(set(self.skill_ids)) != len(self.skill_ids):
            raise ValueError("duplicate skill ids among bandit arms")
        # Injectable RNG keeps unit tests and simulations reproducible.
        self.rng = rng if rng is not None else random.Random()

    # ------------------------------------------------------------------ #
    # Reward model                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clamp(p: float) -> float:
        return min(max(float(p), 0.0), 1.0)

    def reward(self, mastery: Mapping[str, float], skill_id: str) -> float:
        """Expected learning value of one more question on ``skill_id``.

        Skills absent from the mapping count as mastery 0.0: an unknown
        skill is the largest learning opportunity, which safely surfaces
        arms added to the taxonomy after a user's history began.
        """
        return 1.0 - self._clamp(mastery.get(skill_id, 0.0))

    def rewards_from_mastery(self, mastery: Mapping[str, float]) -> dict:
        """Reward for every arm under the current mastery state."""
        return {sid: self.reward(mastery, sid) for sid in self.skill_ids}

    def rank_skills(self, mastery: Mapping[str, float]) -> List[str]:
        """All arms sorted weakest-first (consumed by the analytics dashboard)."""
        rewards = self.rewards_from_mastery(mastery)
        return sorted(self.skill_ids, key=lambda sid: rewards[sid], reverse=True)

    def _greedy_pick(self, mastery: Mapping[str, float]) -> str:
        """Highest-reward arm, ties broken uniformly at random.

        Random tie-breaking matters at cold start: every skill sits at the
        same BKT prior, and a deterministic argmax would forever favour
        whichever arm happens to be listed first.
        """
        rewards = self.rewards_from_mastery(mastery)
        best = max(rewards.values())
        return self.rng.choice([sid for sid, r in rewards.items() if r == best])

    def select_skill(self, mastery: Mapping[str, float]) -> str:
        raise NotImplementedError("subclasses implement the selection rule")


class EpsilonGreedyBandit(SkillBandit):
    """Epsilon-greedy policy: exploit w.p. 1-epsilon, explore uniformly w.p. epsilon."""

    def __init__(
        self,
        skill_ids: Sequence[str],
        epsilon: float = 0.15,
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__(skill_ids, rng=rng)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon!r}")
        self.epsilon = float(epsilon)

    def select_skill(self, mastery: Mapping[str, float]) -> str:
        """Choose the next skill to practise: explore w.p. epsilon, else exploit."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.skill_ids)
        return self._greedy_pick(mastery)


class ThompsonSamplingBandit(SkillBandit):
    """Thompson sampling over the BKT belief state (posterior sampling).

    Classical Thompson sampling maintains a Beta posterior per arm learned
    from reward draws.  This tutor already carries a full Bayesian belief:
    the BKT estimate P(L_k) *is* the posterior that arm k has nothing left
    to teach.  So each round samples a hypothesis of the latent world,

        L_hat_k ~ Bernoulli(P(L_k))    independently per skill,

    and acts greedily in that sampled world: practise a uniformly random
    skill among those hypothesised unlearned; if the sampled world says
    everything is learned, fall back to the lowest-mastery arm.

    Exploration therefore emerges from posterior uncertainty itself - a
    skill believed 95% mastered is still probed in ~5% of rounds, and the
    probe rate decays exactly as confidence grows.  Unlike epsilon-greedy
    there is no exploration constant to tune, and exploration is directed
    at *uncertain* arms instead of being spread uniformly (probability
    matching; Thompson, 1933; empirical case in Chapelle & Li, 2011).
    """

    def select_skill(self, mastery: Mapping[str, float]) -> str:
        sampled_unlearned = [
            sid for sid in self.skill_ids
            if self.rng.random() >= self._clamp(mastery.get(sid, 0.0))
        ]
        if sampled_unlearned:
            return self.rng.choice(sampled_unlearned)
        # Sampled world fully mastered -> probe the weakest estimate anyway.
        return self._greedy_pick(mastery)
