"""Tests for IELTS band mapping and session report structure."""
from src.analysis import session_report


def test_accuracy_to_band_monotonic():
    assert session_report.accuracy_to_band(0.99) >= session_report.accuracy_to_band(0.50)


def test_zero_accuracy_returns_no_band():
    assert session_report.accuracy_to_band(0.0) is None


def test_low_accuracy_gets_realistic_band():
    assert session_report.accuracy_to_band(0.25) == 3.5
    assert session_report.accuracy_to_band(0.10) == 2.5


def test_very_low_accuracy_returns_no_band():
    assert session_report.accuracy_to_band(0.05) is None
    assert session_report.accuracy_to_band(0.01) is None


def test_zero_correct_bucket_has_no_band():
    report = session_report.build_session_report(
        bucket_stats={"Reading": {"correct": 0, "total": 3}},
        skill_stats={},
        mastery={},
        mistakes=[],
        elapsed_secs=60,
        total=3,
        correct=0,
    )
    assert report["bucket_bands"]["Reading"] is None
    assert report["overall_band"] is None


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
    )
    assert report["total"] == 110
    assert report["wrong"] == 47
    assert report["overall_band"] is not None
    assert len(report["weaknesses"]) == 3
    assert report["weaknesses"][0]["name"] in ("READING", "Grammatika", "Lug'at")
    assert report["bucket_bands"]["Reading"] is not None
