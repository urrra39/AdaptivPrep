# Evaluation results

All numbers are exactly reproducible: regenerate this file with
`python -m src.evaluation.policy_eval` (fixed seeds, no wall-clock inputs).

## Knowledge tracing: BKT next-answer prediction

Prequential (predict-then-update) evaluation on the synthetic cohort (150 students x 80 questions, seed 4242; real log has only 0 responses). Every prediction is emitted strictly before its label - no train/test leakage is possible in this protocol.

| Metric | Value | Reference point |
|---|---|---|
| AUC-ROC (Mann-Whitney, tie-corrected) | **0.728** | 0.500 = chance; literature BKT fits ~0.65-0.75 |
| Log-loss | 0.593 | lower is better; proper scoring rule |
| Brier score | 0.205 | lower is better; 0.25 = predicting 0.5 always |
| Accuracy @0.5 | 0.679 | base rate 0.562 |
| Predictions scored | 12,000 | |

*AUC is the primary metric: it is invariant to the upward base-rate drift that inflates accuracy as students learn. Log-loss and Brier audit calibration, which AUC ignores.*

## Recommendation policy: bandit vs random (synthetic students)

Paired simulation of **300 synthetic students x 120 questions x 18 skills** (seed 2026; identical student populations per policy). Students follow the BKT generative process with per-student perturbed parameters; the tutor observes only binary answers. `oracle` reads the hidden state and is the ceiling for any observation-based policy.

| Policy | N=30 | N=60 | N=90 | N=120 | final +/- 95% CI | wasted questions |
|---|---|---|---|---|---|---|
| random | 0.456 | 0.577 | 0.676 | 0.745 | 0.745 +/- 0.012 | 55.6% |
| bandit | 0.479 | 0.642 | 0.784 | 0.876 | 0.876 +/- 0.012 | 38.5% |
| thompson | 0.458 | 0.613 | 0.746 | 0.857 | 0.857 +/- 0.012 | 40.9% |
| oracle | 0.536 | 0.769 | 0.921 | 0.982 | 0.982 +/- 0.006 | 27.5% |

**Epsilon-greedy vs random:** +17.6% relative final mastery, Cohen's d = 1.26. The bandit captures 55% of the oracle-random gap while observing only binary answers.

**Thompson vs epsilon-greedy:** -2.2% relative final mastery (Cohen's d = -0.18), with zero exploration hyperparameters - exploration is allocated by posterior uncertainty instead of a flat epsilon.

*Mastery = ground-truth fraction of skills in the learned state. Wasted = questions spent on already-learned skills - the quantity adaptive selection exists to minimise.*
