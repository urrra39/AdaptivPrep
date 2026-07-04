"""Knowledge-tracing evaluation - how well does BKT predict the next answer?

**Protocol: prequential (predict-then-update) evaluation.**  For every
response in a learner's chronological log we first record the model's
prediction ``P(correct)`` from its *pre-observation* state, then feed the
observation in.  Each prediction is therefore made strictly before the
label it is scored against, which is the standard leakage-free protocol
for sequential learner models (equivalent to one-step-ahead forecasting;
cf. the evaluation setup in Piech et al., 2015, DKT).  A random
train/test split over individual responses would leak: hiding response t
while training on t+1 lets the model peek at the future of the same
sequence.

**Primary metric: AUC-ROC.**  Accuracy is misleading here because the
label base rate drifts upward as students learn (predicting "correct"
gets cheap).  AUC is base-rate invariant: it equals

    AUC = P(score(random correct answer) > score(random incorrect answer))
          + 1/2 * P(equal scores)

i.e. the probability the model ranks a genuinely-known item above an
unknown one.  We compute it exactly via the Mann-Whitney U statistic with
average ranks for ties - O(n log n), no threshold sweep, no binning error.
Chance level is 0.5 regardless of base rate; published BKT fits typically
land in the 0.65-0.75 band.

**Secondary metrics.**  Log-loss (proper scoring rule - punishes confident
wrong predictions, so it audits *calibration*, which AUC ignores) and the
Brier score (mean squared error of the probability, decomposable into
calibration + refinement), plus raw accuracy at the 0.5 threshold for
completeness.

**Data source.**  Real response logs are used when the SQLite DB holds
enough of them; otherwise the evaluation falls back to synthetic students
drawn from the same misspecified-parameter population as the policy
benchmark (see policy_eval), and the report labels the source honestly.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

from src.data import loader, schema
from src.models.bkt import BKTModel
from src.evaluation.policy_eval import SyntheticStudent

# Below this many real responses, AUC confidence intervals are too wide to
# be meaningful - fall back to the synthetic cohort.
MIN_REAL_RESPONSES = 500

SYNTH_STUDENTS = 150
SYNTH_QUESTIONS = 80
SEED = 4242


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def mann_whitney_auc(labels: List[int], scores: List[float]) -> float:
    """Exact AUC-ROC via the rank-sum identity AUC = (R1 - n1(n1+1)/2) / (n1*n0).

    ``R1`` is the sum of the positives' ranks with *average ranks on ties* -
    the tie correction is what makes this exact for discrete scores, where
    a naive pairwise count would have to split ties by convention anyway
    (a tie contributes 1/2, matching the definition in the module docstring).
    """
    n1 = sum(labels)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC undefined: need both positive and negative labels")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):  # assign average ranks across each tie group
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def log_loss(labels: List[int], scores: List[float], eps: float = 1e-12) -> float:
    """Mean negative log-likelihood (proper scoring rule; audits calibration)."""
    total = 0.0
    for y, p in zip(labels, scores):
        p = min(max(p, eps), 1.0 - eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / len(labels)


def brier_score(labels: List[int], scores: List[float]) -> float:
    """Mean squared probability error (calibration + refinement)."""
    return sum((p - y) ** 2 for y, p in zip(labels, scores)) / len(labels)


# --------------------------------------------------------------------------- #
# Prequential replay                                                          #
# --------------------------------------------------------------------------- #
def prequential_predictions(
    sequences: Iterable[List[Tuple[str, bool]]],
    model: Optional[BKTModel] = None,
) -> Tuple[List[int], List[float]]:
    """Predict-then-update over each (skill_id, correct) sequence.

    The BKT state is per-student and per-skill; predictions are emitted from
    the state *before* the update, so no prediction ever sees its own label.
    """
    model = model or BKTModel()
    labels: List[int] = []
    scores: List[float] = []
    for seq in sequences:
        state: Dict[str, float] = {}
        for skill_id, correct in seq:
            p = state.get(skill_id, model.params_for(skill_id).p_init)
            scores.append(model.predict_correct(p, skill_id))
            labels.append(int(correct))
            state[skill_id] = model.update(p, bool(correct), skill_id)
    return labels, scores


# --------------------------------------------------------------------------- #
# Data sources                                                                #
# --------------------------------------------------------------------------- #
def load_real_sequences(db_path=None) -> List[List[Tuple[str, bool]]]:
    """All users' chronological (skill_id, correct) sequences from SQLite."""
    schema.init_db(db_path)  # idempotent: a fresh checkout has no tables yet
    conn = schema.get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT user_id, skill_id, correct FROM responses ORDER BY user_id, id"
        ).fetchall()
    finally:
        conn.close()
    by_user: Dict[int, List[Tuple[str, bool]]] = {}
    for r in rows:
        by_user.setdefault(r["user_id"], []).append((r["skill_id"], bool(r["correct"])))
    return list(by_user.values())


def generate_synthetic_sequences(
    n_students: int = SYNTH_STUDENTS,
    n_questions: int = SYNTH_QUESTIONS,
    seed: int = SEED,
) -> List[List[Tuple[str, bool]]]:
    """Logs from the misspecified-parameter synthetic cohort, random practice.

    Random skill selection (not the bandit) is used for *data generation* so
    the evaluation set covers all skills at all mastery levels - an adaptive
    log would concentrate observations on weak skills and bias the metric.
    """
    skill_ids = loader.skill_ids()
    sequences = []
    for i in range(n_students):
        rng = random.Random(seed * 31 + i)
        student = SyntheticStudent(skill_ids, rng)
        seq = []
        for _ in range(n_questions):
            sid = rng.choice(skill_ids)
            seq.append((sid, student.answer(sid)))
        sequences.append(seq)
    return sequences


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def evaluate(db_path=None) -> Dict:
    """Full evaluation on real data when available, else synthetic."""
    sequences = load_real_sequences(db_path)
    n_real = sum(len(s) for s in sequences)
    if n_real >= MIN_REAL_RESPONSES:
        source = f"real response log ({len(sequences)} users, {n_real} responses)"
    else:
        source = (
            f"synthetic cohort ({SYNTH_STUDENTS} students x {SYNTH_QUESTIONS} "
            f"questions, seed {SEED}; real log has only {n_real} responses)"
        )
        sequences = generate_synthetic_sequences()
    labels, scores = prequential_predictions(sequences)
    accuracy = sum(
        1 for y, p in zip(labels, scores) if (p >= 0.5) == bool(y)
    ) / len(labels)
    return {
        "source": source,
        "n": len(labels),
        "base_rate": sum(labels) / len(labels),
        "auc": mann_whitney_auc(labels, scores),
        "log_loss": log_loss(labels, scores),
        "brier": brier_score(labels, scores),
        "accuracy": accuracy,
    }


def markdown_section(db_path=None) -> str:
    m = evaluate(db_path)
    return "\n".join([
        "## Knowledge tracing: BKT next-answer prediction",
        "",
        f"Prequential (predict-then-update) evaluation on the {m['source']}. "
        "Every prediction is emitted strictly before its label - no "
        "train/test leakage is possible in this protocol.",
        "",
        "| Metric | Value | Reference point |",
        "|---|---|---|",
        f"| AUC-ROC (Mann-Whitney, tie-corrected) | **{m['auc']:.3f}** "
        "| 0.500 = chance; literature BKT fits ~0.65-0.75 |",
        f"| Log-loss | {m['log_loss']:.3f} | lower is better; proper scoring rule |",
        f"| Brier score | {m['brier']:.3f} | lower is better; 0.25 = predicting 0.5 always |",
        f"| Accuracy @0.5 | {m['accuracy']:.3f} | base rate {m['base_rate']:.3f} |",
        f"| Predictions scored | {m['n']:,} | |",
        "",
        "*AUC is the primary metric: it is invariant to the upward base-rate "
        "drift that inflates accuracy as students learn. Log-loss and Brier "
        "audit calibration, which AUC ignores.*",
    ])


if __name__ == "__main__":
    print(markdown_section())
