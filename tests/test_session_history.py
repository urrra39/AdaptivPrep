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
    assert old_fig.data[0].marker.colors[0] == "#C5A059"
    assert good_fig.data[0].marker.colors[0] == "#4E7A57"
    assert bad_fig.data[0].marker.colors[0] == "#A9503C"
    assert none_fig.data[0].labels[0] == "Natija yo'q"


def test_band_score_pie_dark_mode():
    row = {"overall_band": 6.5}
    fig = session_history.band_score_pie(row, "Test", tone="good", dark=True)
    assert fig.layout.paper_bgcolor == "#0B1610"
    assert fig.layout.plot_bgcolor == "#0B1610"


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
def test_band_overview_orders_and_tones():
    rows = [
        {"overall_band": 6.0},  # newest = current
        {"overall_band": 7.5},
        {"overall_band": 4.5},
    ]
    out = session_history.band_overview(rows)
    titles = [t for t, _, _ in out]
    assert titles == ["ENG YUQORI", "O'RTACHA", "ENG PASTI", "HOZIRGI"]
    assert out[0][1]["overall_band"] == 7.5 and out[0][2] == "good"
    assert out[1][1]["overall_band"] == 6.0 and out[1][2] == "old"
    assert out[2][1]["overall_band"] == 4.5 and out[2][2] == "bad"
    assert out[3][1]["overall_band"] == 6.0 and out[3][2] == "old"


def test_band_overview_current_above_and_below_average():
    rows_up = [{"overall_band": 8.0}, {"overall_band": 5.0}]
    assert session_history.band_overview(rows_up)[3][2] == "good"
    rows_down = [{"overall_band": 5.0}, {"overall_band": 8.0}]
    assert session_history.band_overview(rows_down)[3][2] == "bad"


def test_band_overview_equal_results_stay_neutral():
    rows = [{"overall_band": 6.5}, {"overall_band": 6.5}, {"overall_band": 6.5}]
    out = session_history.band_overview(rows)
    assert all(tone == "old" for _, _, tone in out)
    assert all(r["overall_band"] == 6.5 for _, r, _ in out)


def test_band_overview_handles_missing_bands():
    out = session_history.band_overview([{"overall_band": None}])
    assert all(tone == "old" for _, _, tone in out)
    assert all(r["overall_band"] is None for _, r, _ in out)
