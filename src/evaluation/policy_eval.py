"""Policy evaluation - does adaptive selection beat random practice?

**The simulation framework.**  Real A/B data does not exist at cold start,
so the recommendation policy is benchmarked on *synthetic students* whose
ground truth follows the exact generative model BKT assumes (Corbett &
Anderson, 1995): each skill is a latent binary state L in {learned, not
learned}; an unlearned skill is answered correctly only by guessing (prob
``g``), a learned one is missed only by slipping (prob ``1-s``); and every
practice opportunity converts an unlearned skill with prob ``p_transit``.
Per-student parameters are randomly perturbed around the tutor's defaults,
so the tutor operates under *model misspecification* - it never sees the
student's true parameters, only noisy binary answers.  This keeps the
benchmark honest: a policy can only win through better *allocation* of
questions, not through oracle knowledge.

**Why the bandit should win - and by how much.**  Under this generative
model every question on an *unlearned* skill has the same conversion prob
``p_transit``, while a question on an *already-learned* skill teaches
nothing.  A uniform-random policy therefore wastes an increasing share of
questions as mastery grows (in the limit, all of them); a policy that
concentrates on low-mastery skills wastes almost none.  The measurable gap
is exactly the fraction of wasted opportunities - which is why the report
tracks ``wasted%`` alongside mastery itself.

**Four policies bracket the answer.**
  * ``random``   - uniform over skills: the floor (what Phase 2 shipped).
  * ``bandit``   - epsilon-greedy on the tutor's *estimated* mastery: the
    deployed policy, seeing only what a real tutor sees.
  * ``thompson`` - posterior sampling on the same estimates (Phase 10):
    exploration allocated by uncertainty instead of a flat epsilon.
  * ``oracle``   - uniform over truly-unlearned skills, reading the
    student's hidden state: the ceiling no observation-based policy can
    beat.  A policy's value is where it lands inside [random, oracle].

**Statistics reported.**  Mean fraction of skills truly learned after N
questions across M independent students, with a 95% normal-approximation
CI (mean +/- 1.96*sd/sqrt(M) - justified by CLT at M=300), Cohen's d for
bandit-vs-random (standardised effect size, pooled sd), and relative
improvement.  All RNG is seeded: every number in the report is exactly
reproducible.

Running ``python -m src.evaluation.policy_eval`` regenerates
``docs/evaluation_results.md`` (both this section and the kt_eval one).
"""
from __future__ import annotations

import math
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.data import loader
from src.models.bandit import EpsilonGreedyBandit, ThompsonSamplingBandit
from src.models.bkt import BKTModel, BKTParams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "docs" / "evaluation_results.md"

# Benchmark configuration - deliberately fixed so the committed report is
# reproducible bit-for-bit from a clean checkout.
N_STUDENTS = 300
N_QUESTIONS = 120
CHECKPOINTS = (30, 60, 90, 120)
EPSILON = 0.15
SEED = 2026
POLICIES = ("random", "bandit", "thompson", "oracle")
# Deterministic per-policy seed offsets.  str hash() is salted per process
# (PYTHONHASHSEED), so it must never feed an RNG behind published numbers.
_POLICY_OFFSET = {"random": 11, "bandit": 23, "thompson": 37, "oracle": 41}


def _perturbed_params(rng: random.Random) -> BKTParams:
    """Draw one student's true parameters around the tutor's defaults.

    Uniform jitter (clipped to valid ranges) models cohort heterogeneity:
    stronger/weaker priors, faster/slower learners, sloppier/steadier
    answerers.  The tutor always assumes the *default* parameters, so every
    simulated interaction carries realistic estimation error.
    """
    d = BKTParams()
    return BKTParams(
        p_init=min(max(d.p_init + rng.uniform(-0.15, 0.15), 0.05), 0.6),
        p_transit=min(max(d.p_transit + rng.uniform(-0.07, 0.07), 0.03), 0.4),
        p_guess=min(max(d.p_guess + rng.uniform(-0.08, 0.08), 0.10), 0.35),
        p_slip=min(max(d.p_slip + rng.uniform(-0.05, 0.05), 0.02), 0.25),
    )


class SyntheticStudent:
    """A learner whose hidden state follows the BKT generative process.

    The order of operations inside :meth:`answer` matters and follows the
    BKT convention: the answer is emitted from the *current* latent state,
    and the learning transition applies *after* the opportunity - practice
    teaches, but the lesson lands after you have committed to an answer.
    """

    def __init__(
        self,
        skill_ids: Sequence[str],
        rng: random.Random,
        params: Optional[Dict[str, BKTParams]] = None,
    ) -> None:
        self.rng = rng
        self.params: Dict[str, BKTParams] = params or {
            sid: _perturbed_params(rng) for sid in skill_ids
        }
        # Initial knowledge: each skill independently learned w.p. p_init.
        self.learned: Dict[str, bool] = {
            sid: rng.random() < p.p_init for sid, p in self.params.items()
        }

    def answer(self, skill_id: str) -> bool:
        """Emit one observable answer and apply the learning transition."""
        p = self.params[skill_id]
        if self.learned[skill_id]:
            correct = self.rng.random() < (1.0 - p.p_slip)
        else:
            correct = self.rng.random() < p.p_guess
            if self.rng.random() < p.p_transit:  # practice converts the skill
                self.learned[skill_id] = True
        return correct

    def mastered_fraction(self) -> float:
        """Ground-truth outcome metric: share of skills truly learned."""
        return sum(self.learned.values()) / len(self.learned)


def _simulate_one(
    policy: str, student: SyntheticStudent, skill_ids: List[str], rng: random.Random
) -> Dict:
    """Run one student through N_QUESTIONS under ``policy``.

    Returns the mastered fraction at each checkpoint plus the count of
    wasted questions (asked on a skill that was already learned - the
    quantity adaptive selection exists to minimise).
    """
    model = BKTModel()
    estimate = {sid: model.params_for(sid).p_init for sid in skill_ids}
    if policy == "bandit":
        selector = EpsilonGreedyBandit(skill_ids, epsilon=EPSILON, rng=rng)
    elif policy == "thompson":
        selector = ThompsonSamplingBandit(skill_ids, rng=rng)
    elif policy in ("random", "oracle"):
        selector = None
    else:
        raise ValueError(f"unknown policy {policy!r}")
    trajectory, wasted = {}, 0

    for t in range(1, N_QUESTIONS + 1):
        if policy == "random":
            sid = rng.choice(skill_ids)
        elif policy == "oracle":  # upper bound: reads the hidden state
            unlearned = [s for s in skill_ids if not student.learned[s]]
            sid = rng.choice(unlearned) if unlearned else rng.choice(skill_ids)
        else:  # bandit / thompson act on the tutor's estimate only
            sid = selector.select_skill(estimate)

        if student.learned[sid]:
            wasted += 1
        correct = student.answer(sid)
        # The tutor sees only the binary answer - same update as production.
        estimate[sid] = model.update(estimate[sid], correct, sid)
        if t in CHECKPOINTS:
            trajectory[t] = student.mastered_fraction()

    return {"trajectory": trajectory, "wasted": wasted}


def run_benchmark(seed: int = SEED) -> Dict[str, Dict]:
    """Benchmark all three policies on identical student populations.

    Each policy sees the *same* M students (same per-student seeds, hence
    identical true parameters and initial knowledge) - a paired design that
    removes between-student variance from the policy comparison, exactly
    like a matched-pairs experiment.
    """
    skill_ids = loader.skill_ids()
    results: Dict[str, Dict] = {}
    for policy in POLICIES:
        finals, wasted_counts = [], []
        per_checkpoint: Dict[int, List[float]] = {c: [] for c in CHECKPOINTS}
        for i in range(N_STUDENTS):
            # Two independent streams: the student's own randomness must be
            # identical across policies (paired design), while the policy's
            # randomness must not be shared between arms.
            student = SyntheticStudent(skill_ids, random.Random(seed * 100_003 + i))
            policy_rng = random.Random(seed * 200_003 + i * 7 + _POLICY_OFFSET[policy])
            out = _simulate_one(policy, student, skill_ids, policy_rng)
            finals.append(out["trajectory"][CHECKPOINTS[-1]])
            wasted_counts.append(out["wasted"])
            for c in CHECKPOINTS:
                per_checkpoint[c].append(out["trajectory"][c])
        mean = statistics.fmean(finals)
        sd = statistics.stdev(finals)
        results[policy] = {
            "checkpoint_means": {c: statistics.fmean(v) for c, v in per_checkpoint.items()},
            "final_mean": mean,
            "final_sd": sd,
            "ci95": 1.96 * sd / math.sqrt(N_STUDENTS),
            "wasted_pct": 100.0 * statistics.fmean(wasted_counts) / N_QUESTIONS,
            "finals": finals,
        }
    return results


def cohens_d(a: List[float], b: List[float]) -> float:
    """Standardised mean difference with pooled variance (Cohen, 1988)."""
    var_a, var_b = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt((var_a + var_b) / 2.0)
    return (statistics.fmean(a) - statistics.fmean(b)) / pooled


def markdown_section(results: Dict[str, Dict]) -> str:
    """Render the policy benchmark as the report's markdown section."""
    lines = [
        "## Recommendation policy: bandit vs random (synthetic students)",
        "",
        f"Paired simulation of **{N_STUDENTS} synthetic students x "
        f"{N_QUESTIONS} questions x {len(loader.skill_ids())} skills** "
        f"(seed {SEED}; identical student populations per policy). Students "
        "follow the BKT generative process with per-student perturbed "
        "parameters; the tutor observes only binary answers. `oracle` reads "
        "the hidden state and is the ceiling for any observation-based policy.",
        "",
        "| Policy | " + " | ".join(f"N={c}" for c in CHECKPOINTS)
        + " | final +/- 95% CI | wasted questions |",
        "|---|" + "---|" * (len(CHECKPOINTS) + 2),
    ]
    for policy in POLICIES:
        r = results[policy]
        cells = " | ".join(f"{r['checkpoint_means'][c]:.3f}" for c in CHECKPOINTS)
        lines.append(
            f"| {policy} | {cells} | {r['final_mean']:.3f} +/- {r['ci95']:.3f} "
            f"| {r['wasted_pct']:.1f}% |"
        )
    d = cohens_d(results["bandit"]["finals"], results["random"]["finals"])
    rel = 100.0 * (results["bandit"]["final_mean"] - results["random"]["final_mean"]) / \
        results["random"]["final_mean"]
    gap = results["oracle"]["final_mean"] - results["random"]["final_mean"]
    captured = (results["bandit"]["final_mean"] - results["random"]["final_mean"]) / gap \
        if gap > 0 else float("nan")
    d_tb = cohens_d(results["thompson"]["finals"], results["bandit"]["finals"])
    rel_tb = 100.0 * (
        results["thompson"]["final_mean"] - results["bandit"]["final_mean"]
    ) / results["bandit"]["final_mean"]
    lines += [
        "",
        f"**Epsilon-greedy vs random:** +{rel:.1f}% relative final mastery, "
        f"Cohen's d = {d:.2f}. The bandit captures {100 * captured:.0f}% of "
        "the oracle-random gap while observing only binary answers.",
        "",
        f"**Thompson vs epsilon-greedy:** {rel_tb:+.1f}% relative final "
        f"mastery (Cohen's d = {d_tb:.2f}), with zero exploration "
        "hyperparameters - exploration is allocated by posterior uncertainty "
        "instead of a flat epsilon.",
        "",
        "*Mastery = ground-truth fraction of skills in the learned state. "
        "Wasted = questions spent on already-learned skills - the quantity "
        "adaptive selection exists to minimise.*",
    ]
    return "\n".join(lines)


def main() -> None:
    """Regenerate docs/evaluation_results.md (both report sections)."""
    from src.evaluation import kt_eval  # local import: avoids a cycle

    print("Running knowledge-tracing evaluation...")
    kt_section = kt_eval.markdown_section()
    print(f"Running policy benchmark ({len(POLICIES)} policies x "
          f"{N_STUDENTS} students x {N_QUESTIONS} questions)...")
    policy_section = markdown_section(run_benchmark())

    report = "\n".join([
        "# Evaluation results",
        "",
        "All numbers are exactly reproducible: regenerate this file with",
        "`python -m src.evaluation.policy_eval` (fixed seeds, no wall-clock "
        "inputs).",
        "",
        kt_section,
        "",
        policy_section,
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # open() rather than Path.write_text: its newline kwarg needs Python 3.10+,
    # and the report should stay LF regardless of the generating platform.
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
