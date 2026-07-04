"""Bayesian Knowledge Tracing (BKT) - the Student Modeling Layer.

Standard 4-parameter BKT after Corbett & Anderson (1995).  Each skill is
modelled independently as a two-state hidden Markov model: the learner
either has mastered the skill (latent state L) or has not.  After every
observed answer the mastery estimate P(L) is updated in two steps:

1. **Evidence** (Bayes' rule) - how likely was this answer under mastery
   vs non-mastery, accounting for lucky guesses (``p_guess``) and careless
   slips (``p_slip``):

       P(L | correct)   = P(L)(1-s) / [ P(L)(1-s) + (1-P(L)) g ]
       P(L | incorrect) = P(L) s     / [ P(L) s    + (1-P(L))(1-g) ]

2. **Learning** (transition) - every practice opportunity converts an
   unmastered skill to mastered with probability ``p_transit``:

       P(L') = P(L | obs) + (1 - P(L | obs)) * p_transit

Why BKT first (and not DKT): per-skill BKT has four interpretable
parameters, produces sensible estimates from the very first learner
(no training corpus needed), and its mastery output directly feeds the
bandit's reward signal; DKT (LSTM) needs data from hundreds of learners
and is deferred to Phase 10.

Reference: Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing:
Modeling the acquisition of procedural knowledge. *User Modeling and
User-Adapted Interaction*, 4(4), 253-278.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from src.data import loader, schema

# Guard against division by ~0 with extreme (but legal) parameter values.
_EPS = 1e-9


@dataclass(frozen=True)
class BKTParams:
    """The four classic BKT parameters (each a probability in (0, 1)).

    Defaults are standard literature values; ``p_guess=0.25`` is principled
    for 4-option multiple choice (uniform random guessing).
    """

    p_init: float = 0.30     # P(L0): mastery before any practice
    p_transit: float = 0.15  # P(T): unmastered -> mastered per opportunity
    p_guess: float = 0.25    # P(G): correct answer despite non-mastery
    p_slip: float = 0.10     # P(S): incorrect answer despite mastery

    def __post_init__(self) -> None:
        for name in ("p_init", "p_transit", "p_guess", "p_slip"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value!r}")
        # Degeneracy guard: at g + s >= 1 a correct answer would count as
        # evidence *against* mastery and the model loses identifiability.
        if self.p_guess + self.p_slip >= 1.0:
            raise ValueError(
                "p_guess + p_slip must be < 1, got "
                f"{self.p_guess} + {self.p_slip} = {self.p_guess + self.p_slip}"
            )


class BKTModel:
    """Per-skill BKT with optional per-skill parameter overrides.

    ``per_skill_params`` lets individually fitted parameters be plugged in
    later (Phase 7+); skills without an entry use ``default_params``.
    """

    def __init__(
        self,
        default_params: Optional[BKTParams] = None,
        per_skill_params: Optional[Mapping[str, BKTParams]] = None,
    ) -> None:
        self.default_params = default_params or BKTParams()
        self.per_skill_params = dict(per_skill_params or {})

    def params_for(self, skill_id: Optional[str] = None) -> BKTParams:
        """Parameters used for ``skill_id`` (falls back to the defaults)."""
        if skill_id is not None and skill_id in self.per_skill_params:
            return self.per_skill_params[skill_id]
        return self.default_params

    # ------------------------------------------------------------------ #
    # Core update equations                                              #
    # ------------------------------------------------------------------ #
    def posterior(
        self, p_mastery: float, correct: bool, skill_id: Optional[str] = None
    ) -> float:
        """Evidence step only: P(L | observation), no learning transition."""
        prm = self.params_for(skill_id)
        p = min(max(p_mastery, 0.0), 1.0)
        if correct:
            num = p * (1.0 - prm.p_slip)
            den = num + (1.0 - p) * prm.p_guess
        else:
            num = p * prm.p_slip
            den = num + (1.0 - p) * (1.0 - prm.p_guess)
        return num / max(den, _EPS)

    def update(
        self, p_mastery: float, correct: bool, skill_id: Optional[str] = None
    ) -> float:
        """Full BKT update: Bayesian evidence, then the learning transition."""
        prm = self.params_for(skill_id)
        post = self.posterior(p_mastery, correct, skill_id)
        return post + (1.0 - post) * prm.p_transit

    def predict_correct(
        self, p_mastery: float, skill_id: Optional[str] = None
    ) -> float:
        """P(next answer is correct) given current mastery.

        Marginalises over the latent state: p(1-s) + (1-p)g.  Used by the
        Phase 7 evaluation (AUC of predicted correctness on held-out data).
        """
        prm = self.params_for(skill_id)
        p = min(max(p_mastery, 0.0), 1.0)
        return p * (1.0 - prm.p_slip) + (1.0 - p) * prm.p_guess

    # ------------------------------------------------------------------ #
    # Sequence replay                                                    #
    # ------------------------------------------------------------------ #
    def trajectory(
        self, observations: Iterable[bool], skill_id: Optional[str] = None
    ) -> list:
        """Mastery estimate after each observation, starting from p_init."""
        p = self.params_for(skill_id).p_init
        out = []
        for correct in observations:
            p = self.update(p, bool(correct), skill_id)
            out.append(p)
        return out

    def mastery_from_sequence(
        self, observations: Iterable[bool], skill_id: Optional[str] = None
    ) -> float:
        """Final mastery estimate after replaying a full answer sequence."""
        p = self.params_for(skill_id).p_init
        for correct in observations:
            p = self.update(p, bool(correct), skill_id)
        return p


# ---------------------------------------------------------------------- #
# Public API                                                             #
# ---------------------------------------------------------------------- #
def get_mastery(
    user_id: int,
    db_path=None,
    model: Optional[BKTModel] = None,
) -> dict:
    """Return ``{skill_id: P(mastery)}`` for *every* skill in the taxonomy.

    Replays the learner's response history from SQLite in chronological
    order.  Skills not yet practised sit at their prior ``p_init``, so the
    result always covers the full skill set - exactly what the bandit
    (Phase 4) and the dashboard (Phase 6) need.
    """
    model = model or BKTModel()
    mastery = {sid: model.params_for(sid).p_init for sid in loader.skill_ids()}
    reading_ids = set(loader.reading_passage_ids())
    for row in schema.get_responses(user_id, db_path=db_path):
        sid = row["skill_id"]
        if sid not in mastery:
            if sid in reading_ids:
                mastery[sid] = model.params_for(sid).p_init
            else:
                continue
        mastery[sid] = model.update(mastery[sid], bool(row["correct"]), sid)
    return mastery
