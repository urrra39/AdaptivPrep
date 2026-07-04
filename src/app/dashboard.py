"""AdaptivPrep - analytics dashboard (Phase 6).

Three views, each answering one pedagogical question:

1. **Radar chart - "what is my knowledge profile right now?"**  Live BKT
   mastery for all 16 skills on one closed polygon.  A radar is the right
   encoding here because the skill set is a fixed-size cyclic domain and the
   *shape* of the polygon is the signal - a balanced learner traces a wide
   circle, a lopsided one shows dents exactly at the skills the bandit will
   target next.  (For >25 skills a radar degrades into spaghetti; the
   architecture caps the taxonomy well below that.)

2. **Bar chart - "which category needs work?"**  Mean mastery per category
   (Academic Vocabulary / Grammar / Vocabulary in Use).  Categories are the
   level at which a student plans a study week, so aggregation is a mean
   over member skills - each skill is an equally weighted, independently
   modelled BKT unit, so the unweighted mean is the honest summary.

3. **Line chart - "am I actually improving?"**  Two time series over the
   response log: rolling accuracy (raw, observable) and mean BKT mastery
   replayed after every answer (modelled, latent).  Showing both makes the
   model auditable at a glance - mastery should trail accuracy smoothly; if
   the curves diverge wildly the BKT parameters need refitting (Phase 7).

Everything renders from the same SQLite response log the quiz writes -
the dashboard is a pure *reader* (no writes), so it can run side-by-side
with a live quiz session without lock contention.

Run with:  streamlit run src/app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Same sys.path bootstrap as quiz_app.py: `streamlit run` puts this file's
# folder on the path, not the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import loader, schema  # noqa: E402
from src.models.bkt import BKTModel, get_mastery  # noqa: E402

# Colour constants shared across charts so the three views read as one report.
ACCENT = "#2E86AB"
ACCENT_FILL = "rgba(46, 134, 171, 0.30)"
MASTERY_TARGET = 0.95  # BKT's conventional "mastered" threshold, shown as a guide

# Rolling window for observed accuracy: long enough to smooth single-answer
# noise, short enough to react within one study session.
ACCURACY_WINDOW = 10


# --------------------------------------------------------------------------- #
# Data assembly (pure reads; cached per user for snappy reruns)               #
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30)
def load_user_frame(user_id: int) -> pd.DataFrame:
    """The user's full response log as a DataFrame, oldest first.

    ttl=30s: analytics may lag a live quiz by seconds, and in exchange
    Streamlit's rerun-per-widget-interaction never re-reads SQLite.
    """
    rows = schema.get_responses(user_id)
    if not rows:
        return pd.DataFrame(
            columns=["question_id", "skill_id", "correct", "response_time_ms", "timestamp"]
        )
    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["attempt"] = range(1, len(df) + 1)
    return df


def mastery_timeline(df: pd.DataFrame) -> pd.Series:
    """Mean mastery across all skills after each successive answer.

    Replays the log through the same BKTModel the tutor uses, one update per
    row - O(n) total, identical arithmetic to get_mastery's full replay, so
    the curve's endpoint always agrees with the radar chart.
    """
    model = BKTModel()
    state = {sid: model.params_for(sid).p_init for sid in loader.skill_ids()}
    points = []
    for row in df.itertuples():
        if row.skill_id in state:
            state[row.skill_id] = model.update(
                state[row.skill_id], bool(row.correct), row.skill_id
            )
        points.append(sum(state.values()) / len(state))
    return pd.Series(points, index=df["attempt"], name="mean_mastery")


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #
def radar_chart(mastery: dict) -> go.Figure:
    """Closed polygon of per-skill mastery (the learner's knowledge profile)."""
    skills = loader.load_skills()
    labels = [s["name"].split(": ", 1)[-1] for s in skills]  # compact axis labels
    values = [mastery[s["id"]] for s in skills]
    # Close the polygon by repeating the first point - plotly does not do
    # this automatically and an open radar reads as a rendering bug.
    fig = go.Figure(
        go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            fill="toself",
            fillcolor=ACCENT_FILL,
            line={"color": ACCENT, "width": 2},
            name="O'zlashtirish",
            hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        polar={"radialaxis": {"range": [0, 1], "tickformat": ".0%"}},
        showlegend=False,
        margin={"l": 60, "r": 60, "t": 30, "b": 30},
        height=520,
    )
    return fig


def category_bar_chart(mastery: dict) -> go.Figure:
    """Mean mastery per category, sorted weakest-first (the study-plan view)."""
    per_category: dict = {}
    for skill in loader.load_skills():
        per_category.setdefault(skill["category"], []).append(mastery[skill["id"]])
    agg = sorted(
        ((cat, sum(vals) / len(vals)) for cat, vals in per_category.items()),
        key=lambda kv: kv[1],
    )
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in agg],
            y=[c for c, _ in agg],
            orientation="h",
            marker_color=ACCENT,
            text=[f"{v:.0%}" for _, v in agg],
            textposition="outside",
            hovertemplate="%{y}: %{x:.0%}<extra></extra>",
        )
    )
    fig.add_vline(
        x=MASTERY_TARGET,
        line_dash="dash",
        line_color="green",
        annotation_text="maqsad 95%",
    )
    fig.update_layout(
        xaxis={"range": [0, 1.08], "tickformat": ".0%", "title": "O'rtacha o'zlashtirish"},
        margin={"l": 10, "r": 10, "t": 30, "b": 30},
        height=280,
    )
    return fig


def progress_chart(df: pd.DataFrame) -> go.Figure:
    """Observed rolling accuracy vs modelled mean mastery, per attempt.

    min_periods=1 lets the accuracy curve start from the first answer instead
    of showing a 10-attempt gap at the session start.
    """
    accuracy = (
        df["correct"].rolling(ACCURACY_WINDOW, min_periods=1).mean().set_axis(df["attempt"])
    )
    mastery = mastery_timeline(df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=accuracy.index, y=accuracy, mode="lines",
            name=f"Aniqlik (oxirgi {ACCURACY_WINDOW} ta)",
            line={"color": "#F18F01", "width": 2},
            hovertemplate="savol %{x}: %{y:.0%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mastery.index, y=mastery, mode="lines",
            name="O'rtacha o'zlashtirish (BKT)",
            line={"color": ACCENT, "width": 2},
            hovertemplate="savol %{x}: %{y:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"title": "Javob berilgan savollar soni"},
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 10, "r": 10, "t": 30, "b": 30},
        height=380,
    )
    return fig


# --------------------------------------------------------------------------- #
# Page                                                                        #
# --------------------------------------------------------------------------- #
def _select_user() -> tuple:
    """Username selector mirroring the quiz login. Access to the selected
    user's data is gated by the same PIN (see :func:`_authorized_to_view`)."""
    conn = schema.get_connection()
    try:
        users = conn.execute("SELECT id, username FROM users ORDER BY username").fetchall()
    finally:
        conn.close()
    if not users:
        return None, None
    names = [u["username"] for u in users]
    chosen = st.selectbox("Foydalanuvchini tanlang", names)
    return next(u["id"] for u in users if u["username"] == chosen), chosen


def _authorized_to_view(username: str, pin: str, db_path=None) -> tuple:
    """Whether ``pin`` may view ``username``'s analytics.

    Pure (no Streamlit) so the access gate is unit-testable. Reuses the
    schema PIN helpers - no PIN or hashing logic is duplicated here. Returns
    ``(allowed, reason)`` with reason in {ok, no_user, no_pin, bad_pin}.
    """
    row = schema.get_user(username, db_path=db_path)
    if row is None:
        return False, "no_user"
    if row["pin_hash"] is None:
        return False, "no_pin"
    if not schema.verify_pin(int(row["id"]), pin, db_path=db_path):
        return False, "bad_pin"
    return True, "ok"


def main() -> None:
    st.set_page_config(page_title="AdaptivPrep - Tahlil", layout="wide")
    st.title("AdaptivPrep - o'quv tahlili")
    schema.init_db()  # idempotent; makes the dashboard safe to open first

    user_id, username = _select_user()
    if user_id is None:
        st.info("Hozircha foydalanuvchilar yo'q. Avval quiz ilovasida mashq qiling.")
        return

    # The same PIN that protects the quiz login gates this user's analytics.
    if schema.get_user(username)["pin_hash"] is None:
        st.warning(
            "Bu hisob uchun PIN o'rnatilmagan, shuning uchun ma'lumotlarni bu "
            "yerda ko'rib bo'lmaydi. Avval quiz ilovasida shu nom bilan kirib, "
            "PIN yarating."
        )
        return
    pin = st.text_input("PIN kiriting", type="password", max_chars=4, key="dash_pin")
    if not pin:
        st.info("Ushbu foydalanuvchi ma'lumotlarini ko'rish uchun PIN kiriting.")
        return
    if not _authorized_to_view(username, pin)[0]:
        st.error("PIN noto'g'ri. Ma'lumotlar ko'rsatilmaydi.")
        return

    df = load_user_frame(user_id)
    stats = schema.get_user_stats(user_id)
    mastery = get_mastery(user_id)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Javoblar", stats["answered"])
    col2.metric("To'g'ri", stats["correct"])
    col3.metric("Aniqlik", f"{stats['accuracy'] * 100:.0f}%")
    col4.metric(
        "O'zlashtirilgan ko'nikmalar",
        f"{sum(v >= MASTERY_TARGET for v in mastery.values())}/{len(mastery)}",
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Ko'nikmalar profili (radar)")
        st.plotly_chart(radar_chart(mastery), use_container_width=True)
    with right:
        st.subheader("Kategoriyalar bo'yicha")
        st.plotly_chart(category_bar_chart(mastery), use_container_width=True)
        weakest = sorted(mastery, key=mastery.get)[:3]
        st.markdown("**Eng zaif 3 ko'nikma** (bandit navbatdagi nishonlari):")
        for sid in weakest:
            st.markdown(f"- {loader.skill_name(sid)} — {mastery[sid]:.0%}")

    st.subheader("Vaqt bo'yicha o'sish")
    if df.empty:
        st.info(f"{username} hali savollarga javob bermagan.")
    else:
        st.plotly_chart(progress_chart(df), use_container_width=True)


if __name__ == "__main__":
    main()
