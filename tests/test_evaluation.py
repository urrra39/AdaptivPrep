"""Tests for the evaluation sandbox: AUC math pinned by hand, and a smoke
check that the paired simulation reproduces its two headline claims
(bandit > random, oracle >= bandit) at small scale."""
import pytest

from src.evaluation import kt_eval, policy_eval


class TestMannWhitneyAUC:
    def test_perfect_separation_is_one(self):
        assert kt_eval.mann_whitney_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_perfect_inversion_is_zero(self):
        assert kt_eval.mann_whitney_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0

    def test_all_tied_scores_give_half(self):
        assert kt_eval.mann_whitney_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)

    def test_hand_computed_mixed_case(self):
        # positives at 0.8 and 0.4, negatives at 0.6 and 0.2:
        # pairs won 3 of 4 -> AUC = 0.75
        auc = kt_eval.mann_whitney_auc([1, 0, 1, 0], [0.8, 0.6, 0.4, 0.2])
        assert auc == pytest.approx(0.75)

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            kt_eval.mann_whitney_auc([1, 1], [0.5, 0.6])


class TestPrequentialProtocol:
    def test_prediction_precedes_update(self):
        # First prediction for a fresh skill must equal predict_correct(p_init)
        # regardless of the observed label that follows it.
        from src.models.bkt import BKTModel, BKTParams

        model = BKTModel()
        prior_pred = model.predict_correct(BKTParams().p_init)
        labels, scores = kt_eval.prequential_predictions([[("vocab_health", True)]])
        assert scores[0] == pytest.approx(prior_pred)
        assert labels == [1]

    def test_synthetic_auc_beats_chance(self):
        seqs = kt_eval.generate_synthetic_sequences(n_students=40, n_questions=40)
        labels, scores = kt_eval.prequential_predictions(seqs)
        assert kt_eval.mann_whitney_auc(labels, scores) > 0.55


class TestPolicyBenchmark:
    @pytest.fixture(scope="class")
    def results(self):
        # Small-scale run keeps the suite fast; ordering claims already hold.
        import unittest.mock as mock

        with mock.patch.multiple(
            policy_eval, N_STUDENTS=60, N_QUESTIONS=60, CHECKPOINTS=(30, 60)
        ):
            return policy_eval.run_benchmark(seed=7)

    def test_bandit_outperforms_random(self, results):
        assert results["bandit"]["final_mean"] > results["random"]["final_mean"]

    def test_thompson_outperforms_random(self, results):
        assert results["thompson"]["final_mean"] > results["random"]["final_mean"]

    def test_oracle_is_the_ceiling(self, results):
        assert results["oracle"]["final_mean"] >= results["bandit"]["final_mean"]
        assert results["oracle"]["final_mean"] >= results["thompson"]["final_mean"]

    def test_bandit_wastes_fewer_questions(self, results):
        assert results["bandit"]["wasted_pct"] < results["random"]["wasted_pct"]
