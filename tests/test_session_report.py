"""Tests for IELTS band mapping and session report structure."""
from src.analysis import session_report


def test_accuracy_to_band_monotonic():
    assert session_report.accuracy_to_band(0.99) >= session_report.accuracy_to_band(0.50)


def test_zero_accuracy_returns_no_band():
    assert session_report.accuracy_to_band(0.0) is None


def test_low_accuracy_gets_realistic_band():
    assert session_report.accuracy_to_band(0.25) == 3.5
    assert session_report.accuracy_to_band(0.12) == 2.5


def test_very_low_accuracy_still_gets_a_band():
    """Any non-zero accuracy should return a band (not None)."""
    assert session_report.accuracy_to_band(0.07) == 2.0
    assert session_report.accuracy_to_band(0.05) == 1.5
    assert session_report.accuracy_to_band(0.025) == 1.0
    assert session_report.accuracy_to_band(0.01) == 1.0
    assert session_report.accuracy_to_band(0.001) == 1.0


def test_zero_correct_bucket_has_no_band():
    report = session_report.build_session_report(
        bucket_stats={"Reading": {"correct": 0, "total": 3}},
        skill_stats={},
        mastery={},
        mistakes=[],
        elapsed_secs=60,
        total=3,
        correct=0,
        quotas={"Reading": 40, "Grammar": 50, "Vocabulary": 50},
    )
    assert report["bucket_bands"]["Reading"] is None
    assert report["overall_band"] is None


def test_accuracy_uses_quota_not_answered():
    """1 correct out of 5 answered, but quota is 40 → accuracy = 1/40 = 2.5%."""
    report = session_report.build_session_report(
        bucket_stats={"Reading": {"correct": 1, "total": 5}},
        skill_stats={},
        mastery={},
        mistakes=[],
        elapsed_secs=60,
        total=5,
        correct=1,
        quotas={"Reading": 40, "Grammar": 50, "Vocabulary": 50},
    )
    reading_acc = report["bucket_accuracy"]["Reading"]
    assert reading_acc == 1 / 40
    assert report["bucket_bands"]["Reading"] == 1.0
    assert report["overall_band"] == 1.0


def test_one_correct_out_of_full_quota_gets_low_band():
    """1/40 Reading correct → very low but non-None band."""
    report = session_report.build_session_report(
        bucket_stats={"Reading": {"correct": 1, "total": 40}},
        skill_stats={},
        mastery={},
        mistakes=[],
        elapsed_secs=600,
        total=40,
        correct=1,
        quotas={"Reading": 40, "Grammar": 50, "Vocabulary": 50},
    )
    assert report["bucket_bands"]["Reading"] == 1.0
    assert report["overall_band"] == 1.0


def test_build_session_report_shape():
    report = session_report.build_session_report(
        bucket_stats={
            "Reading": {"correct": 8, "total": 10},
            "Grammar": {"correct": 30, "total": 50},
            "Vocabulary": {"correct": 25, "total": 50},
        },
        skill_stats={"sat_transitions": {"correct": 2, "total": 4}},
        mastery={"sat_transitions": 0.4},
        mistakes=[{"question": {}, "choice": 0}],
        elapsed_secs=600,
        total=110,
        correct=63,
        quotas={"Reading": 40, "Grammar": 50, "Vocabulary": 50},
    )
    assert report["total"] == 110
    assert report["total_quota"] == 140
    assert report["wrong"] == 47
    assert report["overall_band"] is not None
    assert len(report["weaknesses"]) == 3
    assert report["weaknesses"][0]["name"] in ("READING", "GRAMMAR", "VOCABULARY")
    assert report["bucket_bands"]["Reading"] is not None


def test_full_correct_gets_high_band():
    report = session_report.build_session_report(
        bucket_stats={
            "Reading": {"correct": 40, "total": 40},
            "Grammar": {"correct": 50, "total": 50},
            "Vocabulary": {"correct": 50, "total": 50},
        },
        skill_stats={},
        mastery={},
        mistakes=[],
        elapsed_secs=1200,
        total=140,
        correct=140,
        quotas={"Reading": 40, "Grammar": 50, "Vocabulary": 50},
    )
    assert report["overall_band"] == 9.0
    assert report["accuracy"] == 1.0


def test_partial_session_realistic_accuracy():
    """User quit after 6 Reading questions (1 correct) → accuracy = 1/140."""
    report = session_report.build_session_report(
        bucket_stats={"Reading": {"correct": 1, "total": 6}},
        skill_stats={},
        mastery={},
        mistakes=[],
        elapsed_secs=200,
        total=6,
        correct=1,
        quotas={"Reading": 40, "Grammar": 50, "Vocabulary": 50},
    )
    assert report["accuracy"] == 1 / 140
    assert report["bucket_accuracy"]["Reading"] == 1 / 40
    assert report["bucket_bands"]["Reading"] == 1.0
    assert report["overall_band"] == 1.0
    assert report["total_quota"] == 140
