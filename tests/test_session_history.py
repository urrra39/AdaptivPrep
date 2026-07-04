"""Tests for persistent session history and compare logic."""
import pytest

from src.analysis import session_history
from src.data import schema


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "hist.db"
    schema.init_db(path)
    return path


def test_compare_sessions_shows_improvement_message():
    older = {"overall_band": 4.0, "accuracy": 0.45}
    newer = {"overall_band": 5.5, "accuracy": 0.62}
    cmp_ = session_history.compare_sessions(older, newer)
    assert cmp_["improved"] is True
    assert "Yuksalish bor" in cmp_["message"]
    assert cmp_["band_delta"] == pytest.approx(1.5)
    assert cmp_["band_ratio"] == pytest.approx(5.5 / 4.0)


def test_band_score_pie_old_and_new_tones():
    older = {"overall_band": 4.0}
    newer = {"overall_band": 5.5}
    old_fig = session_history.band_score_pie(older, "ESKI", tone="old")
    good_fig = session_history.band_score_pie(newer, "YANGI", tone="good")
    bad_fig = session_history.band_score_pie(newer, "YANGI", tone="bad")
    none_fig = session_history.band_score_pie({}, "Bo'sh", tone="bad")
    assert old_fig.data[0].marker.colors[0] == "#9e9e9e"
    assert good_fig.data[0].marker.colors[0] == "#2ecc71"
    assert bad_fig.data[0].marker.colors[0] == "#e74c3c"
    assert none_fig.data[0].labels[0] == "Natija yo'q"


def test_band_score_pie_dark_mode():
    row = {"overall_band": 6.5}
    fig = session_history.band_score_pie(row, "Test", tone="good", dark=True)
    assert fig.layout.paper_bgcolor == "#000000"
    assert fig.layout.plot_bgcolor == "#000000"


def test_save_and_list_session_results(db):
    uid = schema.register_user("a@test.com", "password12", "Ali", db_path=db)
    report = {
        "total": 10,
        "correct": 7,
        "wrong": 3,
        "accuracy": 0.7,
        "overall_band": 6.0,
        "bucket_bands": {"Reading": 6.0, "Grammar": 5.5, "Vocabulary": 6.5},
        "bucket_stats": {"Reading": {"correct": 3, "total": 4}},
    }
    rid = schema.save_session_result(uid, report, 600, db_path=db)
    assert rid >= 1
    rows = schema.list_session_results(uid, db_path=db)
    assert len(rows) == 1
    assert rows[0]["correct"] == 7
    assert rows[0]["wrong"] == 3
    assert rows[0]["duration_secs"] == 600
    assert rows[0]["overall_band"] == 6.0
