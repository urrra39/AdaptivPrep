"""AdaptivPrep - Streamlit quiz interface (v5: 4 ELS passages per session).

Phase order: Reading (4 random ELS passages, Ex 1-2-3 each) -> Grammar (50) ->
Vocabulary (50).  Reading passages are sampled per session so each user sees a
different random subset from the full book corpus.
"""
from __future__ import annotations

import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for root in (here.parents[2], here.parents[1], Path.cwd()):
        if (root / "src" / "data" / "schema.py").is_file():
            return root
    return here.parents[2]


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import session_history, session_report  # noqa: E402
from src.data import auth_email, loader, schema  # noqa: E402
from src.feedback import ai_provider, llm_feedback  # noqa: E402
from src.models.bandit import EpsilonGreedyBandit  # noqa: E402
from src.models.bkt import BKTModel, get_mastery  # noqa: E402

APP_TITLE = "AdaptivPrep - IELTS mashqlari"
APP_VERSION = "v5"
SESSION_SCHEMA_VERSION = 11
_BKT = BKTModel()
_EPSILON = 0.15

# Fixed exam section order — UI and selector both iterate this tuple.
PHASE_ORDER = ("Reading", "Grammar", "Vocabulary")
READING_PASSAGES_PER_SESSION = 4
READING_QUESTIONS_PER_PASSAGE = 10
READING_TOTAL = READING_PASSAGES_PER_SESSION * READING_QUESTIONS_PER_PASSAGE  # 40
QUOTAS = {"Reading": READING_TOTAL, "Grammar": 50, "Vocabulary": 50}


def _bridge_streamlit_secrets() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        try:
            os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass
    for key in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "APP_BASE_URL"):
        if key not in os.environ:
            try:
                os.environ[key] = st.secrets[key]
            except (KeyError, FileNotFoundError):
                pass


def _dev_mode() -> bool:
    return os.environ.get("ADAPTIVPREP_DEV", "").lower() in ("1", "true", "yes")


def _session_smtp_override() -> dict | None:
    cfg = st.session_state.get("smtp_override")
    return cfg if isinstance(cfg, dict) and cfg.get("password") else None


def _render_smtp_setup(default_email: str = "") -> None:
    """Inline Gmail/SMTP setup when env vars are missing."""
    if auth_email.smtp_configured(_session_smtp_override()):
        return
    defaults = auth_email.gmail_defaults_for(default_email)
    with st.expander("Gmail orqali email yuborishni sozlash", expanded=True):
        st.caption(
            "Google Account → Security → 2-Step Verification yoqing, "
            "so'ng App passwords bo'limidan 16 xonali parol oling."
        )
        host = st.text_input(
            "SMTP server",
            value=defaults.get("host", "smtp.gmail.com"),
            key="smtp_setup_host",
        )
        port = st.number_input(
            "Port",
            min_value=1,
            max_value=65535,
            value=int(defaults.get("port", 587)),
            key="smtp_setup_port",
        )
        user = st.text_input(
            "Gmail manzil (yuboruvchi)",
            value=defaults.get("user", default_email),
            key="smtp_setup_user",
        )
        password = st.text_input(
            "Gmail App Password",
            type="password",
            key="smtp_setup_password",
            help="Oddiy Gmail paroli emas — App Password kerak.",
        )
        cols = st.columns(2)
        override = {
            "host": host.strip(),
            "port": int(port),
            "user": user.strip(),
            "password": password,
            "from_addr": user.strip(),
            "use_ssl": int(port) == 465,
        }
        if cols[0].button("Ulanishni tekshirish", use_container_width=True):
            result = auth_email.validate_smtp(override)
            if result.ok:
                st.session_state.smtp_override = override
                st.success(result.message_uz)
            else:
                st.error(result.message_uz)
        if cols[1].button("Sozlamalarni saqlash", use_container_width=True):
            if password:
                st.session_state.smtp_override = override
                st.success("SMTP sozlamalari sessiya uchun saqlandi.")
            else:
                st.error("App Password kiriting.")


def quota_bucket(category: str) -> str:
    return category if category in QUOTAS else "Vocabulary"


def pick_reading_question_ids(passage_id: str, n: int = READING_QUESTIONS_PER_PASSAGE) -> list[str]:
    """Up to ``n`` unique questions per passage, Ex 1 → 2 → 3 order."""
    qs = loader.questions_for_passage_id(passage_id)
    seen = {q["id"] for q in qs}
    if len(qs) < n:
        for q in sorted(
            loader.all_questions_for_passage_id(passage_id),
            key=lambda item: (item["exercise"], str(item.get("sub_id", ""))),
        ):
            if q["id"] in seen:
                continue
            qs.append(q)
            seen.add(q["id"])
            if len(qs) >= n:
                break
    qs.sort(key=lambda q: (q["exercise"], str(q.get("sub_id", ""))))
    out: list[str] = []
    picked: set[str] = set()
    for q in qs:
        if q["id"] in picked:
            continue
        out.append(q["id"])
        picked.add(q["id"])
        if len(out) >= n:
            break
    return out


def quiz_caption_details(
    question: dict,
    reading_passage_ids: list | None = None,
) -> tuple[str, str]:
    """Return (title, detail) for the question caption bar — no source/book names."""
    passage_id = question.get("passage_id")
    if passage_id and reading_passage_ids:
        ex = question.get("exercise", 1)
        try:
            pnum = reading_passage_ids.index(passage_id) + 1
            detail = f"Paragraph {pnum}/{len(reading_passage_ids)} · Exercise {ex}"
        except ValueError:
            detail = f"Exercise {ex}"
        return "READING", detail
    if question.get("bank") == "grammar":
        return "Grammatika", ""
    if question.get("bank") == "vocabulary":
        return "Lug'at", ""
    category = loader.get_skill(question["skill_id"])["category"]
    label = {
        "Reading": "READING",
        "Grammar": "Grammatika",
        "Vocabulary": "Lug'at",
    }.get(category, category)
    return label, ""


def _format_savol_caption(question_num: int, title: str, detail: str) -> str:
    if detail:
        return f"Savol #{question_num} — {title} — {detail}"
    return f"Savol #{question_num} — {title}"


READING_EXERCISE_HEADINGS = {
    1: "Exercise 1 — Passagedagi so'zlarni toping",
    2: "Exercise 2 — To'g'ri javobni tanlang",
    3: "Exercise 3 — So'z banki bilan matnni to'ldiring",
}


def _session_needs_repair() -> bool:
    """Stale Streamlit state from an older app version or pre-bank sessions."""
    if not st.session_state.get("user_id"):
        return False
    if st.session_state.get("session_schema_version") != SESSION_SCHEMA_VERSION:
        return True
    if int(st.session_state.get("session_grammar_quota", 0)) != QUOTAS["Grammar"]:
        return True
    if int(st.session_state.get("session_vocabulary_quota", 0)) != QUOTAS["Vocabulary"]:
        return True
    if st.session_state.get("finished"):
        return False
    r_pass = st.session_state.get("reading_passage_ids") or []
    if len(r_pass) != min(READING_PASSAGES_PER_SESSION, loader.reading_passage_count()):
        return True
    if len(r_pass) > READING_PASSAGES_PER_SESSION:
        return True
    if not st.session_state.get("reading_question_ids"):
        return True
    if int(st.session_state.get("session_reading_quota", 0)) != READING_TOTAL:
        return True
    if not st.session_state.get("grammar_question_ids"):
        return True
    if len(st.session_state.get("grammar_question_ids", [])) != QUOTAS["Grammar"]:
        return True
    if not st.session_state.get("session_queue"):
        return True
    queue = st.session_state.get("session_queue") or []
    expected = READING_TOTAL + QUOTAS["Grammar"] + QUOTAS["Vocabulary"]
    if expected and len(queue) != expected:
        return True


def _phase_has_supply(
    phase: str,
    seen_ids: set,
    reading_passage_ids: list | None = None,
    grammar_question_ids: list | None = None,
    vocabulary_question_ids: list | None = None,
    reading_order: list | None = None,
) -> bool:
    if phase == "Reading":
        if reading_order:
            return any(qid not in seen_ids for qid in reading_order)
        pool = reading_passage_ids or []
        for pid in pool:
            if any(q["id"] not in seen_ids for q in loader.questions_for_passage_id(pid)):
                return True
        return False
    if phase == "Grammar":
        pool = grammar_question_ids or []
        return any(qid not in seen_ids for qid in pool)
    if phase == "Vocabulary":
        pool = vocabulary_question_ids or []
        return any(qid not in seen_ids for qid in pool)
    for skill in loader.load_skills():
        if quota_bucket(skill["category"]) != phase:
            continue
        if any(q["id"] not in seen_ids for q in loader.questions_for_skill(skill["id"])):
            return True
    return False


def _reading_quota(session: dict) -> int:
    return int(session.get("session_reading_quota", 0))


def _grammar_quota(session: dict) -> int:
    return int(session.get("session_grammar_quota", QUOTAS["Grammar"]))


def _vocabulary_quota(session: dict) -> int:
    return int(session.get("session_vocabulary_quota", QUOTAS["Vocabulary"]))


def active_phase(quota_used: dict, seen_ids: set, session: dict | None = None) -> str | None:
    session = session or {}
    caps = {
        "Reading": _reading_quota(session),
        "Grammar": _grammar_quota(session),
        "Vocabulary": _vocabulary_quota(session),
    }
    for phase in PHASE_ORDER:
        if quota_used.get(phase, 0) >= caps[phase]:
            continue
        if _phase_has_supply(
            phase,
            seen_ids,
            session.get("reading_passage_ids"),
            session.get("grammar_question_ids"),
            session.get("vocabulary_question_ids"),
            session.get("reading_order"),
        ):
            return phase
    return None


def eligible_skills(seen_ids: set, quota_used: dict, session: dict | None = None) -> list:
    session = session or {}
    phase = active_phase(quota_used, seen_ids, session)
    if phase is None:
        return []
    if phase == "Reading":
        out = []
        for pid in session.get("reading_passage_ids", []):
            if any(q["id"] not in seen_ids for q in loader.questions_for_passage_id(pid)):
                out.append(pid)
        return out
    if phase == "Grammar":
        return [
            qid
            for qid in session.get("grammar_question_ids", [])
            if qid not in seen_ids
        ]
    if phase == "Vocabulary":
        return [
            qid
            for qid in session.get("vocabulary_question_ids", [])
            if qid not in seen_ids
        ]
    out = []
    for skill in loader.load_skills():
        if quota_bucket(skill["category"]) != phase:
            continue
        if any(q["id"] not in seen_ids for q in loader.questions_for_skill(skill["id"])):
            out.append(skill["id"])
    return out


def _select_reading_question(seen_ids: set, session: dict) -> dict | None:
    """Next reading question following the pre-built 40-question session order."""
    for qid in session.get("reading_order", []):
        if qid not in seen_ids:
            return loader.get_question(qid)
    return None


def _select_bank_question(
    seen_ids: set,
    question_ids: list,
    order: list,
) -> dict | None:
    """Pick next question following pre-shuffled session order."""
    for qid in order:
        if qid in question_ids and qid not in seen_ids:
            return loader.get_question(qid)
    return None


def select_next_question(
    mastery: dict,
    rng: random.Random,
    seen_ids: set,
    quota_used: dict,
    session: dict | None = None,
):
    session = session or {}
    phase = active_phase(quota_used, seen_ids, session)
    if phase == "Reading":
        return _select_reading_question(seen_ids, session)
    if phase == "Grammar":
        return _select_bank_question(
            seen_ids,
            session.get("grammar_question_ids", []),
            session.get("grammar_order", []),
        )
    if phase == "Vocabulary":
        return _select_bank_question(
            seen_ids,
            session.get("vocabulary_question_ids", []),
            session.get("vocabulary_order", []),
        )
    arms = eligible_skills(seen_ids, quota_used, session)
    if not arms:
        return None
    skill_id = EpsilonGreedyBandit(arms, epsilon=_EPSILON, rng=rng).select_skill(mastery)
    pool = [q for q in loader.questions_for_skill(skill_id) if q["id"] not in seen_ids]
    return rng.choice(pool)


def _init_session_stats() -> None:
    st.session_state.bucket_stats = {p: {"correct": 0, "total": 0} for p in PHASE_ORDER}
    st.session_state.skill_stats = {}
    st.session_state.ai_chat = []


def _init_ai_defaults() -> None:
    st.session_state.setdefault("ai_provider", "Anthropic")
    st.session_state.setdefault("ai_model", ai_provider.default_model("Anthropic"))
    st.session_state.setdefault("api_validated", False)
    st.session_state.setdefault("api_status_msg", "")
    st.session_state.setdefault("api_status_expires_at", 0.0)


def _reset_api_validation() -> None:
    st.session_state.api_validated = False
    st.session_state.api_status_msg = ""
    st.session_state.api_status_expires_at = 0.0


def _session_ctx() -> dict:
    return {
        "reading_passage_ids": st.session_state.get("reading_passage_ids", []),
        "reading_question_ids": st.session_state.get("reading_question_ids", []),
        "reading_order": st.session_state.get("reading_order", []),
        "session_reading_quota": st.session_state.get("session_reading_quota", 0),
        "grammar_question_ids": st.session_state.get("grammar_question_ids", []),
        "grammar_order": st.session_state.get("grammar_order", []),
        "session_grammar_quota": st.session_state.get("session_grammar_quota", 0),
        "vocabulary_question_ids": st.session_state.get("vocabulary_question_ids", []),
        "vocabulary_order": st.session_state.get("vocabulary_order", []),
        "session_vocabulary_quota": st.session_state.get("session_vocabulary_quota", 0),
    }


def _init_reading_session(rng: random.Random) -> None:
    """4 passages × 10 questions = 40 reading items (Ex 1→2→3 within each)."""
    pool = loader.reading_passage_ids()
    eligible = [
        pid
        for pid in pool
        if len(pick_reading_question_ids(pid)) >= READING_QUESTIONS_PER_PASSAGE
    ]
    source = list(eligible if len(eligible) >= READING_PASSAGES_PER_SESSION else pool)
    rng.shuffle(source)
    passages: list[str] = []
    reading_ids: list[str] = []
    for pid in source:
        ids = pick_reading_question_ids(pid)
        if len(ids) < READING_QUESTIONS_PER_PASSAGE:
            continue
        passages.append(pid)
        reading_ids.extend(ids[:READING_QUESTIONS_PER_PASSAGE])
        if len(passages) >= READING_PASSAGES_PER_SESSION:
            break
    st.session_state.reading_passage_ids = passages
    st.session_state.reading_question_ids = reading_ids
    st.session_state.reading_order = reading_ids.copy()
    st.session_state.session_reading_quota = len(reading_ids)


def _init_grammar_session(rng: random.Random) -> None:
    pool = loader.grammar_question_ids()
    n = min(QUOTAS["Grammar"], len(pool))
    ids = rng.sample(pool, n)
    st.session_state.grammar_question_ids = ids
    st.session_state.grammar_order = ids.copy()
    rng.shuffle(st.session_state.grammar_order)
    st.session_state.session_grammar_quota = n


def _init_vocabulary_session(rng: random.Random) -> None:
    pool = loader.vocabulary_question_ids()
    n = min(QUOTAS["Vocabulary"], len(pool))
    ids = rng.sample(pool, n)
    st.session_state.vocabulary_question_ids = ids
    st.session_state.vocabulary_order = ids.copy()
    rng.shuffle(st.session_state.vocabulary_order)
    st.session_state.session_vocabulary_quota = n


def _init_ui_prefs() -> None:
    st.session_state.setdefault("theme", "day")
    st.session_state.setdefault("show_results", False)
    st.session_state.setdefault("theme_flash_until", 0.0)


def _render_theme_flash() -> None:
    if time.monotonic() >= st.session_state.get("theme_flash_until", 0.0):
        return
    flash = st.session_state.get("theme_flash", "day")
    cls = "theme-flash-sun" if flash == "day" else "theme-flash-moon"
    icon = "☀️" if flash == "day" else "🌙"
    st.markdown(
        f'<div class="theme-flash-overlay {cls}"><span class="theme-flash-icon">{icon}</span></div>',
        unsafe_allow_html=True,
    )


def _switch_theme(mode: str) -> None:
    if st.session_state.get("theme") != mode:
        st.session_state.theme = mode
        st.session_state.theme_flash = mode
        st.session_state.theme_flash_until = time.monotonic() + 1.3
    st.rerun()


def _apply_theme_css() -> None:
    theme = st.session_state.get("theme", "day")
    is_night = theme == "night"
    sun_anim = "" if is_night else "animation: sunPulse 2s ease-in-out infinite;"
    moon_anim = "animation: moonGlow 2s ease-in-out infinite;" if is_night else ""

    accent_css = """
    .stApp .new-session-marker + div[data-testid="stButton"] > button {
        background: rgba(10, 61, 98, 0.75) !important;
        background-color: rgba(10, 61, 98, 0.75) !important;
        color: #ffffff !important;
        border: 2px solid #0a3d62 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 2.85rem !important;
        box-shadow: 0 0 14px rgba(10, 61, 98, 0.35) !important;
    }
    .stApp .new-session-marker + div[data-testid="stButton"] > button:hover {
        background: rgba(10, 61, 98, 0.92) !important;
        background-color: rgba(10, 61, 98, 0.92) !important;
    }
    .stApp .new-session-marker + div[data-testid="stButton"] > button p {
        color: #ffffff !important;
    }
    .stApp .quiz-nav-marker + div[data-testid="stHorizontalBlock"] .stButton > button {
        background: rgba(10, 61, 98, 0.75) !important;
        background-color: rgba(10, 61, 98, 0.75) !important;
        color: #ffffff !important;
        border: 2px solid #0a3d62 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 2.85rem !important;
        white-space: nowrap !important;
        box-shadow: 0 0 14px rgba(10, 61, 98, 0.35) !important;
    }
    .stApp .quiz-nav-marker + div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: rgba(10, 61, 98, 0.92) !important;
    }
    .stApp .quiz-finish-marker + div[data-testid="stButton"] > button,
    .stApp .quiz-confirm-marker + div[data-testid="stHorizontalBlock"] .stButton > button {
        background: rgba(10, 61, 98, 0.75) !important;
        background-color: rgba(10, 61, 98, 0.75) !important;
        color: #ffffff !important;
        border: 2px solid #0a3d62 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: 0 0 14px rgba(10, 61, 98, 0.35) !important;
    }
    .user-badge {
        display: inline-block;
        background: rgba(46, 134, 171, 0.35) !important;
        color: #ffffff !important;
        border: 1px solid #2E86AB !important;
        padding: 0.12rem 0.65rem;
        border-radius: 0.5rem;
        font-weight: 600;
        box-shadow: 0 0 10px rgba(46, 134, 171, 0.35);
    }
    .stApp button[kind="primary"],
    .stApp [data-testid="stBaseButton-primary"],
    .stApp [data-testid="stFormSubmitButton"] button,
    .stApp [data-testid="stFormSubmitButton"] > button {
        background: rgba(46, 134, 171, 0.45) !important;
        color: #ffffff !important;
        border: 1px solid #2E86AB !important;
        font-weight: 700 !important;
        box-shadow: 0 0 14px rgba(46, 134, 171, 0.45) !important;
    }
    .stApp button[kind="primary"]:hover,
    .stApp [data-testid="stBaseButton-primary"]:hover,
    .stApp [data-testid="stFormSubmitButton"] button:hover,
    .stApp [data-testid="stFormSubmitButton"] > button:hover {
        background: rgba(46, 134, 171, 0.65) !important;
    }
    .stApp [data-testid="stTabs"] button {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding-bottom: 0.4rem !important;
    }
    .stApp [data-testid="stTabs"] button[aria-selected="true"] {
        color: #2E86AB !important;
        border-bottom: 2px solid #2E86AB !important;
        font-weight: 700 !important;
    }
    .stApp [data-testid="stTabs"] button[aria-selected="false"] {
        color: #aaaaaa !important;
        border-bottom: 2px solid transparent !important;
    }
    .stApp [data-baseweb="input"] span,
    .stApp [data-baseweb="input"] [data-baseweb="input-suffix"],
    .stApp span[data-baseweb="input-suffix"],
    .stApp [data-baseweb="input"] [data-baseweb="input-suffix"] > div,
    .stApp span[data-baseweb="input-suffix"] > div,
    .stApp [data-baseweb="input"] > div:last-child,
    .stApp [data-baseweb="input"] > div:last-child > div {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    .stApp [data-baseweb="input"] [data-baseweb="button"],
    .stApp [data-baseweb="input"] button,
    .stApp [data-testid="stTextInput"] [data-baseweb="button"],
    .stApp [data-testid="stTextInput"] button {
        background: rgba(10, 61, 98, 0.88) !important;
        background-color: rgba(10, 61, 98, 0.88) !important;
        border: 1px solid #0a3d62 !important;
        border-radius: 6px !important;
        color: #ffffff !important;
        box-shadow: none !important;
        min-width: 2.4rem !important;
        min-height: 2rem !important;
        margin: 2px !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    .stApp [data-baseweb="input"] [data-baseweb="button"] svg,
    .stApp [data-baseweb="input"] button svg,
    .stApp [data-baseweb="input"] [data-baseweb="button"] svg path,
    .stApp [data-baseweb="input"] button svg path {
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    .stApp [data-testid="stTextInput"] [data-baseweb="base-input"],
    .stApp [data-testid="stTextInput"] [data-baseweb="input"] {
        overflow: hidden !important;
    }
    .summary-section-header {
        margin: 0.5rem 0 0.85rem 0;
    }
    .summary-section-title {
        color: #8ecae6 !important;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .summary-section-caption {
        color: #aaaaaa !important;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-bottom: 0.15rem;
    }
    .summary-card {
        background: rgba(46, 134, 171, 0.22);
        border: 1px solid #2E86AB;
        border-radius: 12px;
        padding: 1rem 0.75rem;
        text-align: center;
        box-shadow: 0 0 18px rgba(46, 134, 171, 0.28);
        margin-bottom: 0.35rem;
    }
    .summary-card-label {
        color: #8ecae6 !important;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .summary-card-value {
        color: #ffffff !important;
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .summary-list-panel {
        background: rgba(46, 134, 171, 0.1);
        border: 1px solid rgba(46, 134, 171, 0.45);
        border-left: 4px solid #2E86AB;
        border-radius: 0 12px 12px 0;
        padding: 0.9rem 1.15rem;
        margin: 0.65rem 0 1.1rem 0;
    }
    .summary-list-title {
        color: #8ecae6 !important;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }
    .summary-list-panel ul {
        margin: 0;
        padding-left: 1.2rem;
    }
    .compare-banner {
        background: rgba(46, 134, 171, 0.18);
        border: 1px solid rgba(46, 134, 171, 0.55);
        border-radius: 12px;
        padding: 0.95rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
        font-weight: 600;
        font-size: 1rem;
        color: #ffffff !important;
        box-shadow: 0 0 16px rgba(46, 134, 171, 0.25);
    }
    .compare-banner-good {
        border-left: 5px solid #2ecc71;
    }
    .compare-banner-bad {
        border-left: 5px solid #e74c3c;
    }
    .stApp [data-testid="InputInstructions"],
    .stApp [data-testid="InputInstructions"] * {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    """

    login_night_css = """
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.65rem !important;
        background: transparent !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stTabs"] button {
        background: rgba(46, 134, 171, 0.28) !important;
        background-color: rgba(46, 134, 171, 0.28) !important;
        color: #8ecae6 !important;
        border: 1px solid #2E86AB !important;
        border-radius: 10px !important;
        border-bottom: 1px solid #2E86AB !important;
        padding: 0.5rem 1.4rem !important;
        font-weight: 700 !important;
        box-shadow: 0 0 12px rgba(46, 134, 171, 0.35) !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stTabs"] button[aria-selected="true"] {
        background: rgba(46, 134, 171, 0.65) !important;
        background-color: rgba(46, 134, 171, 0.65) !important;
        color: #ffffff !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stForm"] {
        background: rgba(46, 134, 171, 0.12) !important;
        border: 1px solid rgba(46, 134, 171, 0.55) !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.1rem 0.6rem 1.1rem !important;
        margin-top: 0.75rem !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stFormSubmitButton"] button,
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stForm"] button,
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stForm"] .stButton > button,
    .stApp [data-testid="column"]:has(.login-panel-marker) .stButton > button {
        background: rgba(46, 134, 171, 0.55) !important;
        background-color: rgba(46, 134, 171, 0.55) !important;
        color: #ffffff !important;
        border: 2px solid #2E86AB !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 2.85rem !important;
        box-shadow: 0 0 18px rgba(46, 134, 171, 0.5) !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stFormSubmitButton"] button p,
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stForm"] button p,
    .stApp [data-testid="column"]:has(.login-panel-marker) .stButton > button p {
        color: #ffffff !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) .stButton {
        margin-top: 0.65rem !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(46, 134, 171, 0.12) !important;
        border-color: rgba(46, 134, 171, 0.55) !important;
        border-radius: 14px !important;
        margin-top: 0.75rem !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) .stButton > button {
        background: rgba(46, 134, 171, 0.55) !important;
        background-color: rgba(46, 134, 171, 0.55) !important;
        color: #ffffff !important;
        border: 2px solid #2E86AB !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 2.85rem !important;
        box-shadow: 0 0 18px rgba(46, 134, 171, 0.5) !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) .stButton > button p {
        color: #ffffff !important;
    }
    """

    login_day_css = """
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.65rem !important;
        background: transparent !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stTabs"] button {
        background: rgba(10, 61, 98, 0.12) !important;
        background-color: rgba(10, 61, 98, 0.12) !important;
        color: #0a3d62 !important;
        border: 1px solid rgba(10, 61, 98, 0.55) !important;
        border-radius: 10px !important;
        border-bottom: 1px solid rgba(10, 61, 98, 0.55) !important;
        padding: 0.5rem 1.4rem !important;
        font-weight: 700 !important;
        box-shadow: 0 0 10px rgba(10, 61, 98, 0.15) !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stTabs"] button[aria-selected="true"] {
        background: rgba(10, 61, 98, 0.9) !important;
        background-color: rgba(10, 61, 98, 0.9) !important;
        color: #ffffff !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stForm"] {
        background: rgba(10, 61, 98, 0.06) !important;
        border: 1px solid rgba(10, 61, 98, 0.35) !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.1rem 0.6rem 1.1rem !important;
        margin-top: 0.75rem !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stFormSubmitButton"] button,
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stForm"] button,
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stForm"] .stButton > button,
    .stApp [data-testid="column"]:has(.login-panel-marker) .stButton > button {
        background: rgba(10, 61, 98, 0.88) !important;
        background-color: rgba(10, 61, 98, 0.88) !important;
        color: #ffffff !important;
        border: 2px solid #0a3d62 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 2.85rem !important;
        box-shadow: 0 0 14px rgba(10, 61, 98, 0.35) !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stFormSubmitButton"] button p,
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stForm"] button p,
    .stApp [data-testid="column"]:has(.login-panel-marker) .stButton > button p {
        color: #ffffff !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) .stButton {
        margin-top: 0.65rem !important;
    }
    """

    day_css = """
    [data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #1A1A2E !important;
    }
    .stApp button[kind="primary"],
    .stApp [data-testid="stBaseButton-primary"],
    .stApp [data-testid="stFormSubmitButton"] button,
    .stApp [data-testid="stFormSubmitButton"] > button,
    .stApp .stButton > button[kind="primary"] {
        background: rgba(10, 61, 98, 0.88) !important;
        background-color: rgba(10, 61, 98, 0.88) !important;
        color: #ffffff !important;
        border: 2px solid #0a3d62 !important;
        box-shadow: 0 0 14px rgba(10, 61, 98, 0.35) !important;
    }
    .summary-section-title { color: #0a3d62 !important; }
    .summary-card {
        background: rgba(10, 61, 98, 0.08) !important;
        border: 1px solid rgba(10, 61, 98, 0.45) !important;
        box-shadow: 0 0 14px rgba(10, 61, 98, 0.12) !important;
    }
    .summary-card-label { color: #0a3d62 !important; }
    .summary-card-value { color: #0a3d62 !important; }
    .summary-list-panel {
        background: rgba(10, 61, 98, 0.06) !important;
        border: 1px solid rgba(10, 61, 98, 0.35) !important;
        border-left: 4px solid #0a3d62 !important;
    }
    .summary-list-title { color: #0a3d62 !important; }
    .summary-list-panel li,
    .summary-list-panel li strong,
    .summary-list-panel p { color: #1A1A2E !important; }
    .summary-section-caption { color: #444444 !important; }
    .compare-banner {
        background: rgba(10, 61, 98, 0.08) !important;
        border: 1px solid rgba(10, 61, 98, 0.4) !important;
        color: #0a3d62 !important;
        box-shadow: 0 0 14px rgba(10, 61, 98, 0.12) !important;
    }
    """

    login_css = login_day_css if not is_night else login_night_css

    day_final_css = """
    .stApp .user-badge {
        background: rgba(10, 61, 98, 0.75) !important;
        border: 1px solid #0a3d62 !important;
        color: #ffffff !important;
        box-shadow: 0 0 12px rgba(10, 61, 98, 0.35) !important;
    }
    .stApp .summary-card {
        background: rgba(10, 61, 98, 0.75) !important;
        border: 1px solid #0a3d62 !important;
        box-shadow: 0 0 16px rgba(10, 61, 98, 0.35) !important;
    }
    .stApp .summary-card-label { color: #8ecae6 !important; }
    .stApp .summary-card-value { color: #ffffff !important; }
    .stApp .compare-banner {
        background: rgba(10, 61, 98, 0.75) !important;
        border: 1px solid #0a3d62 !important;
        color: #ffffff !important;
        box-shadow: 0 0 16px rgba(10, 61, 98, 0.35) !important;
    }
    .stApp .summary-list-panel {
        background: rgba(10, 61, 98, 0.06) !important;
        border: 1px solid rgba(10, 61, 98, 0.35) !important;
        border-left: 4px solid #0a3d62 !important;
    }
    .stApp .summary-list-title { color: #0a3d62 !important; }
    .stApp .summary-list-panel li,
    .stApp .summary-list-panel li strong,
    .stApp .summary-list-panel p { color: #1A1A2E !important; }
    .stApp .summary-section-title { color: #0a3d62 !important; }
    .stApp .summary-section-caption { color: #444444 !important; }
    .stApp [data-testid="column"]:has(.login-panel-marker) [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(10, 61, 98, 0.06) !important;
        border-color: rgba(10, 61, 98, 0.35) !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) .stButton > button {
        background: rgba(10, 61, 98, 0.88) !important;
        background-color: rgba(10, 61, 98, 0.88) !important;
        color: #ffffff !important;
        border: 2px solid #0a3d62 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 2.85rem !important;
        box-shadow: 0 0 14px rgba(10, 61, 98, 0.35) !important;
    }
    .stApp [data-testid="column"]:has(.login-panel-marker) .stButton > button p {
        color: #ffffff !important;
    }
    """

    night_css = """
    .stApp, [data-testid="stAppViewContainer"], .main, [data-testid="stMain"] {
        background-color: #000000 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #000000 !important;
        border-right: 1px solid #333333 !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p, .stApp label, .stApp span, .stApp li,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp .stMetric label {
        color: #ffffff !important;
    }
    .stApp [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    .stApp [data-testid="stTextInput"] input,
    .stApp [data-baseweb="input"] {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
        caret-color: #ffffff !important;
    }
    .stApp .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
        font-weight: 600 !important;
    }
    .stApp [data-baseweb="select"] > div {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
    }
    .stApp [data-baseweb="select"] span,
    .stApp [data-baseweb="select"] div {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    .stApp [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: transparent !important;
    }
    .stApp [data-testid="stTabs"] button[aria-selected="false"] {
        color: #cccccc !important;
    }
    .stApp [data-testid="column"]:has(.theme-sun) button,
    .stApp [data-testid="column"]:has(.theme-moon) button {
        border-width: 2px !important;
    }
    .stApp hr { border-color: #333333 !important; }
    .stApp .summary-list-panel li,
    .stApp .summary-list-panel p {
        color: #ffffff !important;
        font-size: 0.95rem;
        line-height: 1.55;
        margin-bottom: 0.35rem;
    }
    .stApp .summary-section-caption { color: #aaaaaa !important; }
    """

    theme_css = (night_css if is_night else day_css) + accent_css + login_css
    if not is_night:
        theme_css += day_final_css

    eye_toggle_css = """
    .stApp [data-testid="stTextInputRootElement"],
    .stApp [data-testid="stTextInput"] > div,
    .stApp [data-testid="stTextInput"] > div > div,
    .stApp [data-baseweb="input"],
    .stApp [data-baseweb="input"] > div,
    .stApp [data-testid="stTextInput"] [data-baseweb="base-input"] {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    .stApp [data-baseweb="input"] [data-baseweb="input-suffix"],
    .stApp span[data-baseweb="input-suffix"],
    .stApp [data-baseweb="input"] [data-baseweb="input-suffix"] > div,
    .stApp span[data-baseweb="input-suffix"] > div {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 2px 0 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .stApp [data-baseweb="input"] [data-baseweb="button"],
    .stApp [data-baseweb="input"] button[type="button"],
    .stApp [data-testid="stTextInput"] [data-baseweb="button"],
    .stApp [data-testid="stTextInput"] button {
        background: rgba(10, 61, 98, 0.88) !important;
        background-color: rgba(10, 61, 98, 0.88) !important;
        border: 1px solid #0a3d62 !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        outline: none !important;
        width: 2.2rem !important;
        height: 2.2rem !important;
        min-width: 2.2rem !important;
        min-height: 2.2rem !important;
        max-width: 2.2rem !important;
        max-height: 2.2rem !important;
        padding: 0 !important;
        margin: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        overflow: hidden !important;
        line-height: 1 !important;
    }
    .stApp [data-baseweb="input"] [data-baseweb="button"] > span,
    .stApp [data-baseweb="input"] [data-baseweb="button"] > div,
    .stApp [data-baseweb="input"] button > span,
    .stApp [data-baseweb="input"] button > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        line-height: 0 !important;
    }
    .stApp [data-baseweb="input"] [data-baseweb="button"] svg,
    .stApp [data-baseweb="input"] button svg {
        width: 1rem !important;
        height: 1rem !important;
        min-width: 1rem !important;
        min-height: 1rem !important;
        display: block !important;
        margin: 0 auto !important;
        position: static !important;
        flex-shrink: 0 !important;
    }
    .stApp [data-baseweb="input"] [data-baseweb="button"] svg path,
    .stApp [data-baseweb="input"] button svg path {
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    .stApp [data-baseweb="input"] {
        display: flex !important;
        align-items: stretch !important;
        flex-wrap: nowrap !important;
        gap: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
        border-radius: 8px !important;
    }
    .stApp [data-baseweb="input"] > div:first-child {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        background: transparent !important;
        background-color: transparent !important;
    }
    .stApp [data-testid="stTextInput"] input,
    .stApp [data-baseweb="input"] input {
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: none !important;
        margin: 0 !important;
        width: 100% !important;
    }
    .stApp [data-baseweb="input"] [data-baseweb="button"],
    .stApp [data-baseweb="input"] button[type="button"] {
        border-radius: 0 7px 7px 0 !important;
        margin: 0 !important;
        align-self: stretch !important;
        height: auto !important;
        max-height: none !important;
    }
    """
    theme_css += eye_toggle_css

    sidebar_css = """
    [data-testid="stSidebar"] [data-baseweb="input"] {
        border: 1px solid rgba(10, 61, 98, 0.35) !important;
        background: #ffffff !important;
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #1A1A2E !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button,
    [data-testid="stSidebar"] [data-testid="stButton"] button,
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(10, 61, 98, 0.88) !important;
        background-color: rgba(10, 61, 98, 0.88) !important;
        color: #ffffff !important;
        border: 2px solid #0a3d62 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 2.6rem !important;
        box-shadow: 0 0 12px rgba(10, 61, 98, 0.3) !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button p,
    [data-testid="stSidebar"] .stButton > button p {
        color: #ffffff !important;
    }
    """
    if is_night:
        sidebar_css += """
    [data-testid="stSidebar"] [data-baseweb="input"] {
        border: 1px solid #555555 !important;
        background: #000000 !important;
        background-color: #000000 !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background: #000000 !important;
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button,
    [data-testid="stSidebar"] [data-testid="stButton"] button,
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(46, 134, 171, 0.55) !important;
        background-color: rgba(46, 134, 171, 0.55) !important;
        border: 2px solid #2E86AB !important;
        box-shadow: 0 0 14px rgba(46, 134, 171, 0.4) !important;
    }
    """
    theme_css += sidebar_css

    login_page_css = """
    .stApp:has(.login-page-active) [data-testid="stButton"] > button,
    .stApp:has(.login-page-active) [data-testid="stButton"] button,
    .stApp:has(.login-page-active) .stButton > button {
        background: rgba(10, 61, 98, 0.78) !important;
        background-color: rgba(10, 61, 98, 0.78) !important;
        color: #ffffff !important;
        border: 2px solid #0a3d62 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 2.85rem !important;
        width: 100% !important;
        box-shadow: 0 0 16px rgba(10, 61, 98, 0.45) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    .stApp:has(.login-page-active) [data-testid="stButton"] > button p,
    .stApp:has(.login-page-active) [data-testid="stButton"] > button span,
    .stApp:has(.login-page-active) .stButton > button p {
        color: #ffffff !important;
    }
    .stApp:has(.login-page-active) [data-testid="stButton"] {
        margin-top: 0.65rem !important;
    }
    .stApp:has(.login-page-active) [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(10, 61, 98, 0.12) !important;
        border-color: rgba(10, 61, 98, 0.45) !important;
        border-radius: 14px !important;
        margin-top: 0.75rem !important;
    }
    """
    if is_night:
        login_page_css += """
    .stApp:has(.login-page-active) [data-testid="stButton"] > button,
    .stApp:has(.login-page-active) [data-testid="stButton"] button,
    .stApp:has(.login-page-active) .stButton > button {
        background: rgba(46, 134, 171, 0.55) !important;
        background-color: rgba(46, 134, 171, 0.55) !important;
        border: 2px solid #2E86AB !important;
        box-shadow: 0 0 18px rgba(46, 134, 171, 0.5) !important;
    }
    .stApp:has(.login-page-active) [data-testid="stTabs"] button {
        background: rgba(46, 134, 171, 0.28) !important;
        background-color: rgba(46, 134, 171, 0.28) !important;
        color: #8ecae6 !important;
        border: 1px solid #2E86AB !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.4rem !important;
        font-weight: 700 !important;
        box-shadow: 0 0 12px rgba(46, 134, 171, 0.35) !important;
    }
    .stApp:has(.login-page-active) [data-testid="stTabs"] button[aria-selected="true"] {
        background: rgba(46, 134, 171, 0.65) !important;
        background-color: rgba(46, 134, 171, 0.65) !important;
        color: #ffffff !important;
    }
    .stApp:has(.login-page-active) [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(46, 134, 171, 0.12) !important;
        border-color: rgba(46, 134, 171, 0.55) !important;
    }
    """
    theme_css += login_page_css

    st.markdown(
        f"""
        <style>
        @keyframes sunPulse {{
            0%, 100% {{ transform: scale(1); filter: drop-shadow(0 0 6px #ffb300); }}
            50% {{ transform: scale(1.15); filter: drop-shadow(0 0 18px #ff8f00); }}
        }}
        @keyframes moonGlow {{
            0%, 100% {{ transform: scale(1); filter: drop-shadow(0 0 6px #90caf9); }}
            50% {{ transform: scale(1.12); filter: drop-shadow(0 0 18px #42a5f5); }}
        }}
        @keyframes flashSun {{
            0% {{ opacity: 0; transform: scale(0.4); }}
            30% {{ opacity: 1; transform: scale(1.2); }}
            100% {{ opacity: 0; transform: scale(2.4); }}
        }}
        @keyframes flashMoon {{
            0% {{ opacity: 0; transform: rotate(-20deg) scale(0.5); }}
            35% {{ opacity: 1; transform: rotate(0deg) scale(1.1); }}
            100% {{ opacity: 0; transform: rotate(20deg) scale(2.2); }}
        }}
        .theme-flash-overlay {{
            position: fixed; inset: 0; z-index: 999999;
            display: flex; align-items: center; justify-content: center;
            pointer-events: none;
        }}
        .theme-flash-sun {{
            background: radial-gradient(circle, rgba(255,213,79,0.55), rgba(255,152,0,0.05));
            animation: flashSun 1.2s ease-out forwards;
        }}
        .theme-flash-moon {{
            background: radial-gradient(circle, rgba(66,165,245,0.45), rgba(26,35,126,0.05));
            animation: flashMoon 1.2s ease-out forwards;
        }}
        .theme-flash-icon {{ font-size: 7rem; }}
        div[data-testid="column"]:has(.theme-sun) button {{
            font-size: 1.6rem !important;
            background: linear-gradient(135deg, #fff3e0, #ffe082) !important;
            color: #1a1a2e !important;
            border: 2px solid #ffb300 !important;
            {sun_anim}
        }}
        div[data-testid="column"]:has(.theme-moon) button {{
            font-size: 1.6rem !important;
            background: linear-gradient(135deg, #1a237e, #3949ab) !important;
            color: #e3f2fd !important;
            border: 2px solid #5c6bc0 !important;
            {moon_anim}
        }}
        {theme_css}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _render_theme_flash()


def _render_theme_toggle() -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="theme-sun"></div>', unsafe_allow_html=True)
        if st.button("☀️", key="theme_day_btn", help="Kunduzgi rejim", use_container_width=True):
            _switch_theme("day")
    with c2:
        st.markdown('<div class="theme-moon"></div>', unsafe_allow_html=True)
        if st.button("🌙", key="theme_night_btn", help="Tungi rejim", use_container_width=True):
            _switch_theme("night")


def _fmt_band(value) -> str:
    return "—" if value is None else str(value)


def _summary_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="summary-card">'
        f'<div class="summary-card-label">{label}</div>'
        f'<div class="summary-card-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _summary_section_header(title: str, caption: str = "") -> None:
    cap = f'<div class="summary-section-caption">{caption}</div>' if caption else ""
    st.markdown(
        f'<div class="summary-section-header">'
        f'<div class="summary-section-title">{title}</div>{cap}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _summary_list_panel(title: str, items: list[str], empty: str) -> None:
    if items:
        body = "".join(f"<li>{item}</li>" for item in items)
        inner = f"<ul>{body}</ul>"
    else:
        inner = f'<p>{empty}</p>'
    st.markdown(
        f'<div class="summary-list-panel">'
        f'<div class="summary-list-title">{title}</div>{inner}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _compare_banner(message: str, improved: bool) -> None:
    tone = "compare-banner-good" if improved else "compare-banner-bad"
    st.markdown(
        f'<div class="compare-banner {tone}">{message}</div>',
        unsafe_allow_html=True,
    )


def _session_quotas() -> dict:
    """Fixed section quotas shown in UI and used for IELTS scoring."""
    return {
        "Reading": QUOTAS["Reading"],
        "Grammar": QUOTAS["Grammar"],
        "Vocabulary": QUOTAS["Vocabulary"],
    }


def _build_live_report() -> dict:
    secs = int(time.monotonic() - st.session_state.start_time)
    return session_report.build_session_report(
        st.session_state.bucket_stats,
        st.session_state.skill_stats,
        st.session_state.mastery,
        st.session_state.mistakes,
        secs,
        st.session_state.session_total,
        st.session_state.session_correct,
        quotas=_session_quotas(),
    )


def _build_session_queue() -> list[str]:
    return (
        list(st.session_state.get("reading_order", []))
        + list(st.session_state.get("grammar_order", []))
        + list(st.session_state.get("vocabulary_order", []))
    )


def _load_question_at_index(idx: int) -> None:
    queue = st.session_state.session_queue
    if not queue:
        st.session_state.current = None
        return
    idx = max(0, min(int(idx), len(queue) - 1))
    st.session_state.queue_index = idx
    st.session_state.current = loader.get_question(queue[idx])
    st.session_state.shown_at = time.monotonic()


def _unanswered_items() -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    rpass = st.session_state.get("reading_passage_ids") or []
    for i, qid in enumerate(st.session_state.get("session_queue", [])):
        if qid in st.session_state.get("answers", {}):
            continue
        q = loader.get_question(qid)
        cat = loader.display_category(q["skill_id"])
        if q.get("passage_id"):
            try:
                pnum = rpass.index(q["passage_id"]) + 1
                label = f"READING · Paragraph {pnum}/{len(rpass)} · Exercise {q.get('exercise', '?')}"
            except ValueError:
                label = f"READING · Exercise {q.get('exercise', '?')}"
        else:
            label = loader.display_skill_name(q["skill_id"])
        out.append((i + 1, cat, label))
    return out


def _request_finish_session() -> None:
    missing = _unanswered_items()
    if not missing:
        st.session_state.finish_confirm = True
        return
    if not st.session_state.get("finish_warn_shown"):
        st.session_state.finish_warn_shown = True
        st.session_state.finish_warn_until = time.monotonic() + 10
        lines = [f"#{num} — {label}" for num, _cat, label in missing[:15]]
        extra = len(missing) - 15
        if extra > 0:
            lines.append(f"... va yana {extra} ta savol")
        st.session_state.finish_warn_text = (
            "Belgilanmagan savollar qoldi! 10 soniya davomida ko'rib chiqing:\n"
            + "\n".join(lines)
        )
        return
    st.session_state.finish_confirm = True


def _save_session_to_history() -> None:
    if st.session_state.get("result_saved") or st.session_state.session_total <= 0:
        return
    report = _build_live_report()
    schema.save_session_result(
        st.session_state.user_id,
        report,
        report["elapsed_secs"],
    )
    st.session_state.result_saved = True


def _finish_session() -> None:
    _save_session_to_history()
    st.session_state.finished = True
    if st.session_state.get("api_validated"):
        st.session_state.open_ai_on_summary = True


def _advance_to_next_question() -> None:
    idx = int(st.session_state.get("queue_index", 0))
    queue = st.session_state.get("session_queue") or []
    if idx + 1 < len(queue):
        _load_question_at_index(idx + 1)
    else:
        _finish_session()


def _start_session(user_id: int, username: str) -> None:
    st.session_state.user_id = user_id
    st.session_state.username = username
    st.session_state.session_schema_version = SESSION_SCHEMA_VERSION
    st.session_state.seen_question_ids = set()
    st.session_state.quota_used = {p: 0 for p in PHASE_ORDER}
    st.session_state.session_total = 0
    st.session_state.session_correct = 0
    st.session_state.mistakes = []
    st.session_state.answers = {}
    st.session_state.queue_index = 0
    st.session_state.finish_warn_shown = False
    st.session_state.finish_warn_until = 0.0
    st.session_state.finish_warn_text = ""
    st.session_state.finish_confirm = False
    st.session_state.start_time = time.monotonic()
    st.session_state.finished = False
    st.session_state.result_saved = False
    st.session_state.show_results = False
    st.session_state.rng = random.Random(os.urandom(16))
    st.session_state.mastery = get_mastery(user_id)
    _init_reading_session(st.session_state.rng)
    _init_grammar_session(st.session_state.rng)
    _init_vocabulary_session(st.session_state.rng)
    st.session_state.session_queue = _build_session_queue()
    for pid in st.session_state.reading_passage_ids:
        st.session_state.mastery.setdefault(pid, _BKT.params_for(pid).p_init)
    for qid in (
        st.session_state.grammar_question_ids + st.session_state.vocabulary_question_ids
    ):
        q = loader.get_question(qid)
        st.session_state.mastery.setdefault(q["skill_id"], _BKT.params_for(q["skill_id"]).p_init)
    _init_session_stats()
    _init_ai_defaults()
    if st.session_state.session_queue:
        _load_question_at_index(0)
    else:
        st.session_state.current = None
        _finish_session()
    st.rerun()


def _validate_api_key() -> None:
    key = (st.session_state.get("user_api_key") or "").strip()
    provider = st.session_state.get("ai_provider", "Anthropic")
    model = st.session_state.get("ai_model", ai_provider.default_model(provider))
    if not key:
        st.session_state.api_validated = False
        st.session_state.api_status_msg = "API kalitini kiriting."
        st.session_state.api_status_expires_at = 0.0
        return
    result = ai_provider.validate_key(provider, key, model)
    st.session_state.api_validated = result.ok
    st.session_state.api_status_msg = result.message_uz
    if result.ok:
        st.session_state.api_status_expires_at = time.monotonic() + 10
    else:
        st.session_state.api_status_expires_at = 0.0


def _on_ai_provider_change() -> None:
    provider = st.session_state.get("ai_provider", "Anthropic")
    st.session_state.ai_model = ai_provider.default_model(provider)
    _reset_api_validation()


def _render_api_status() -> None:
    """Success banner with 10-second auto-hide (fragment reruns every second)."""
    if not st.session_state.get("api_validated"):
        return
    msg = st.session_state.get("api_status_msg", "")
    if not msg:
        return
    expires = st.session_state.get("api_status_expires_at", 0.0)
    if time.monotonic() >= expires:
        st.session_state.api_status_msg = ""
        return
    remaining = max(0, int(expires - time.monotonic()))
    st.success(f"{msg} ({remaining}s)")


@st.fragment(run_every=1)
def _api_status_fragment() -> None:
    _render_api_status()


def _reset_password_view() -> None:
    st.title("Parolni tiklash")
    token = st.query_params.get("reset_token", "")
    with st.form("reset_form"):
        new_pw = st.text_input("Yangi parol", type="password")
        confirm = st.text_input("Parolni tasdiqlang", type="password")
        if st.form_submit_button("Parolni yangilash"):
            if len(new_pw) < 8:
                st.error("Parol kamida 8 belgidan iborat bo'lishi kerak.")
            elif new_pw != confirm:
                st.error("Parollar mos kelmadi.")
            elif schema.reset_password_with_token(token, new_pw):
                st.success("Parol yangilandi. Endi kirishingiz mumkin.")
                st.query_params.clear()
                st.rerun()
            else:
                st.error("Havola yaroqsiz yoki muddati tugagan.")
    if st.button("Kirish ekraniga qaytish"):
        st.query_params.clear()
        st.rerun()


def _login_view() -> None:
    _init_ui_prefs()
    _apply_theme_css()
    st.markdown('<div class="login-page-active" style="display:none"></div>', unsafe_allow_html=True)
    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.title(APP_TITLE)
    with top_r:
        _render_theme_toggle()
    st.caption("Bilim darajangizni moslashuvchan tarzda baholaydigan trenajyor.")
    if not loader.reading_bank_available():
        st.warning(
            "O'qish banki hozircha yuklanmagan. "
            "`python scripts/ingest_els_full.py` ni ishga tushiring."
        )
    schema.init_db()

    flash = st.session_state.pop("forgot_flash", None)
    if flash:
        st.success(flash)

    if st.session_state.get("auth_mode") == "forgot":
        st.subheader("Parolni tiklash")
        forgot_email = st.session_state.get("forgot_email", "")
        _render_smtp_setup(forgot_email)
        with st.form("forgot_form"):
            email = st.text_input("Gmail manzilingiz", value=forgot_email)
            if st.form_submit_button("Tiklash havolasini yuborish"):
                st.session_state.forgot_email = email.strip()
                smtp = _session_smtp_override()
                if not auth_email.smtp_configured(smtp):
                    st.error(
                        "Avval Gmail SMTP sozlamalarini kiriting va "
                        "'Ulanishni tekshirish' tugmasini bosing."
                    )
                else:
                    token = schema.create_password_reset_token(email)
                    if token is None:
                        st.session_state.auth_mode = None
                        st.session_state.forgot_flash = (
                            "Agar bu email ro'yxatdan o'tgan bo'lsa, tiklash xabari yuborildi."
                        )
                        st.rerun()
                    result = auth_email.send_password_reset_email(
                        email, token, smtp_override=smtp
                    )
                    if result.ok:
                        st.session_state.auth_mode = None
                        st.session_state.forgot_flash = result.message_uz
                        st.rerun()
                    st.error(result.message_uz)
                    if _dev_mode():
                        base = os.environ.get("APP_BASE_URL", "http://localhost:8501")
                        st.warning("Dev rejimi — havola:")
                        st.code(f"{base.rstrip('/')}/?reset_token={token}")
        if st.button("Kirish ekraniga qaytish"):
            st.session_state.auth_mode = None
            st.rerun()
        return

    _, login_col, _ = st.columns([1, 1.4, 1])
    with login_col:
        st.markdown('<div class="login-panel-marker"></div>', unsafe_allow_html=True)
        tab_in, tab_up = st.tabs(["Kirish", "Ro'yxatdan o'tish"])
        with tab_in:
            with st.container(border=True):
                email = st.text_input("Gmail", placeholder="siz@gmail.com", key="login_email")
                password = st.text_input("Parol", type="password", key="login_password")
            if st.button("Kirish", type="primary", use_container_width=True, key="btn_login"):
                try:
                    uid, name = schema.authenticate_user(email, password)
                    _start_session(uid, name)
                except schema.AuthError:
                    st.error("Email yoki parol noto'g'ri.")
                except ValueError:
                    st.error("Email manzil noto'g'ri.")
            if st.button("Parolni unutdingizmi?", type="primary", use_container_width=True):
                st.session_state.auth_mode = "forgot"
                st.rerun()

        with tab_up:
            with st.container(border=True):
                name = st.text_input("Ismingiz", placeholder="Dilnoza", key="signup_name")
                email = st.text_input("Gmail", placeholder="siz@gmail.com", key="signup_email")
                password = st.text_input("Parol (kamida 8 belgi)", type="password", key="signup_password")
                confirm = st.text_input("Parolni tasdiqlang", type="password", key="signup_confirm")
            if st.button("Ro'yxatdan o'tish", type="primary", use_container_width=True, key="btn_signup"):
                if password != confirm:
                    st.error("Parollar mos kelmadi.")
                elif len(password) < 8:
                    st.error("Parol kamida 8 belgidan iborat bo'lishi kerak.")
                elif not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
                    st.error("Parol harf va raqamdan iborat bo'lishi kerak.")
                else:
                    try:
                        uid = schema.register_user(email, password, name)
                        _start_session(uid, name)
                    except schema.EmailTakenError:
                        st.error("Bu Gmail allaqachon ro'yxatdan o'tgan.")
                    except ValueError as exc:
                        st.error(str(exc))


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f'**Foydalanuvchi:** <span class="user-badge">{st.session_state.username}</span>',
            unsafe_allow_html=True,
        )
        st.metric("Javob berilgan", _answered_count())
        st.markdown(
            f"**READING: {QUOTAS['Reading']} ta** · "
            f"**Grammar: {QUOTAS['Grammar']} ta** · "
            f"**Vocabulary: {QUOTAS['Vocabulary']} ta**"
        )
        cur = st.session_state.get("current")
        if cur and not st.session_state.finished:
            cur_bank = cur.get("bank", "")
            if cur.get("passage_id"):
                cur_phase = "Reading"
            elif cur_bank == "grammar":
                cur_phase = "Grammar"
            elif cur_bank == "vocabulary":
                cur_phase = "Vocabulary"
            else:
                cur_phase = quota_bucket(loader.display_category(cur.get("skill_id", "")))
            st.info(f"Joriy bosqich: **{cur_phase}**")
        st.divider()
        if st.button("📊 NATIJALAR", use_container_width=True, type="primary"):
            st.session_state.show_results = True
            st.rerun()
        st.divider()
        st.markdown("**AI sozlamalari**")
        provider = st.selectbox(
            "AI kompaniya",
            options=list(ai_provider.PROVIDERS.keys()),
            key="ai_provider",
            on_change=_on_ai_provider_change,
        )
        models = ai_provider.PROVIDERS[provider]["models"]
        if st.session_state.get("ai_model") not in models:
            st.session_state.ai_model = ai_provider.default_model(provider)
        st.selectbox(
            "AI model",
            options=models,
            key="ai_model",
            on_change=_reset_api_validation,
        )
        st.text_input(
            "API kaliti",
            type="password",
            key="user_api_key",
            on_change=_reset_api_validation,
        )
        if st.button("API kalitini tekshirish", use_container_width=True, type="primary"):
            _validate_api_key()
        err_msg = st.session_state.get("api_status_msg", "")
        if err_msg and not st.session_state.get("api_validated"):
            st.error(err_msg)
        elif st.session_state.get("api_validated") and err_msg:
            _api_status_fragment()
        st.caption(
            "Kalit faqat joriy sessiyada saqlanadi. AI izoh va suhbat sessiya "
            "yakunida ochiladi."
        )
        st.divider()
        if st.button("Chiqish", use_container_width=True):
            st.session_state.clear()
            st.rerun()


def _answered_count() -> int:
    return len(st.session_state.get("answers") or {})


def _reverse_answer_stats(question: dict, was_correct: bool) -> None:
    skill_id = question["skill_id"]
    bucket = quota_bucket(loader.get_skill(skill_id)["category"])
    bs = st.session_state.bucket_stats[bucket]
    bs["total"] -= 1
    bs["correct"] -= int(was_correct)
    ss = st.session_state.skill_stats.get(skill_id)
    if ss:
        ss["total"] -= 1
        ss["correct"] -= int(was_correct)
    st.session_state.session_correct -= int(was_correct)
    st.session_state.quota_used[bucket] -= 1
    st.session_state.session_total -= 1
    if not was_correct:
        qid = question["id"]
        st.session_state.mistakes = [
            m for m in st.session_state.mistakes if m["question"]["id"] != qid
        ]


def _record_stats(question: dict, is_correct: bool) -> None:
    skill_id = question["skill_id"]
    bucket = quota_bucket(loader.get_skill(skill_id)["category"])
    bs = st.session_state.bucket_stats[bucket]
    bs["total"] += 1
    bs["correct"] += int(is_correct)
    ss = st.session_state.skill_stats.setdefault(skill_id, {"correct": 0, "total": 0})
    ss["total"] += 1
    ss["correct"] += int(is_correct)


def _commit_answer(question: dict, choice: int) -> None:
    """Record a selected answer into stats/DB. Handles re-answers gracefully."""
    qid = question["id"]
    prev = st.session_state.answers.get(qid)
    if prev is not None:
        if prev == choice:
            return
        was_correct = prev == question["correct_answer"]
        _reverse_answer_stats(question, was_correct)

    elapsed_ms = int((time.monotonic() - st.session_state.shown_at) * 1000)
    is_correct = choice == question["correct_answer"]
    try:
        schema.record_response(
            user_id=st.session_state.user_id,
            question_id=question["id"],
            skill_id=question["skill_id"],
            correct=is_correct,
            response_time_ms=elapsed_ms,
        )
    except sqlite3.Error:
        st.toast("Javob diskka saqlanmadi; mashq davom etadi.")
    skill_id = question["skill_id"]
    prior = st.session_state.mastery.get(
        skill_id, _BKT.params_for(skill_id).p_init
    )
    st.session_state.mastery[skill_id] = _BKT.update(prior, is_correct, skill_id)
    st.session_state.seen_question_ids.add(qid)
    bucket = quota_bucket(loader.get_skill(skill_id)["category"])
    if prev is None:
        st.session_state.quota_used[bucket] += 1
        st.session_state.session_total += 1
    st.session_state.session_correct += int(is_correct)
    _record_stats(question, is_correct)
    if not is_correct:
        st.session_state.mistakes = [
            m for m in st.session_state.mistakes if m["question"]["id"] != qid
        ]
        st.session_state.mistakes.append({"question": question, "choice": choice})
    else:
        st.session_state.mistakes = [
            m for m in st.session_state.mistakes if m["question"]["id"] != qid
        ]
    st.session_state.answers[qid] = choice


def _format_question_prompt(question: dict) -> str:
    """Human-readable prompt; garbled Ex.1 ingest rows get a safe fallback."""
    text = question.get("question_text") or ""
    if question.get("exercise") == 1 and loader.is_garbled_prompt(text):
        return (
            "[Exercise 1] Passagedagi so'z bankidan to'g'ri javobni tanlang "
            "(ta'rif buzilgan — variantlardan mosini toping)."
        )
    return text


def _render_finish_warnings() -> None:
    warn_until = st.session_state.get("finish_warn_until", 0.0)
    if warn_until and time.monotonic() < warn_until:
        remaining = max(0, int(warn_until - time.monotonic()))
        text = st.session_state.get("finish_warn_text", "")
        st.warning(f"{text}\n\n⏳ {remaining} soniya")

    if st.session_state.get("finish_confirm"):
        st.error("SIZ ROSTDAN HAM TUGATMOQCHIMISZ?")
        st.markdown('<div class="quiz-confirm-marker"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("HA", type="primary", use_container_width=True, key="finish_yes"):
            st.session_state.finish_confirm = False
            st.session_state.finish_warn_shown = False
            _finish_session()
            st.rerun()
        if c2.button("YO'Q", type="primary", use_container_width=True, key="finish_no"):
            st.session_state.finish_confirm = False
            st.rerun()


def _render_question_panel(question: dict) -> None:
    queue = st.session_state.get("session_queue") or []
    idx = int(st.session_state.get("queue_index", 0))
    qid = question["id"]
    saved = st.session_state.answers.get(qid)
    is_current = queue[idx] == qid if queue else True

    prompt_text = _format_question_prompt(question)
    if prompt_text:
        q_bank = question.get("bank", "")
        if q_bank in ("grammar", "vocabulary") and len(prompt_text) > 100:
            st.markdown(
                f"""<div style="
                    background: var(--secondary-background-color, #f0f2f6);
                    border-radius: 8px;
                    padding: 1rem 1.2rem;
                    margin-bottom: 1rem;
                    max-height: 300px;
                    overflow-y: auto;
                    line-height: 1.6;
                    font-size: 0.95rem;
                    border-left: 4px solid #2e86ab;
                "><strong>{prompt_text}</strong></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.write(f"**{prompt_text}**")

    if saved is not None and not is_current:
        st.info(f"Sizning javobingiz: **{question['options'][saved]}**")
        st.success(f"To'g'ri javob: **{question['options'][question['correct_answer']]}**")
        choice = saved
    else:
        choice = st.radio(
            "Javobni tanlang:",
            options=list(range(len(question["options"]))),
            format_func=lambda i: question["options"][i],
            index=saved if saved is not None else None,
            key=f"choice_{question['id']}",
        )

    st.markdown('<div class="quiz-nav-marker"></div>', unsafe_allow_html=True)
    nav_l, nav_r = st.columns(2)
    with nav_l:
        if st.button("← Orqaga", type="primary", use_container_width=True, disabled=idx <= 0):
            if choice is not None:
                _commit_answer(question, int(choice))
            _load_question_at_index(idx - 1)
            st.rerun()
    with nav_r:
        if st.button("Keyingi →", type="primary", use_container_width=True, disabled=idx >= len(queue) - 1):
            if choice is not None:
                _commit_answer(question, int(choice))
            _load_question_at_index(idx + 1)
            st.rerun()

    st.divider()
    _render_finish_warnings()
    st.markdown('<div class="quiz-finish-marker"></div>', unsafe_allow_html=True)
    if st.button("Sessiyani yakunlash", type="primary", use_container_width=True, key="btn_finish_session"):
        if choice is not None:
            _commit_answer(question, int(choice))
        _request_finish_session()
        st.rerun()


def _quiz_view() -> None:
    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.title(APP_TITLE)
    with top_r:
        _render_theme_toggle()

    @st.fragment(run_every=1.0)
    def _live_timer() -> None:
        secs = int(time.monotonic() - st.session_state.start_time)
        st.markdown(f"### ⏱️ {secs // 60:02d}:{secs % 60:02d}")

    _live_timer()
    question = st.session_state.current
    r_pass = st.session_state.get("reading_passage_ids", [])
    if question.get("passage_id"):
        ex = question.get("exercise")
        _, detail = quiz_caption_details(question, r_pass)
        heading = READING_EXERCISE_HEADINGS.get(ex, f"Exercise {ex}")
        st.subheader(f"📖 READING — {detail}")
        passage = loader.get_reading_passage(question["passage_id"])
        if passage:
            st.markdown(f"**{passage['title']}**")
        st.markdown(f"*{heading}*")
        passage_body = loader.passage_display_text(question["passage_id"])
        st.markdown(
            f"""<div style="
                background: var(--secondary-background-color, #f0f2f6);
                border-radius: 8px;
                padding: 1.2rem 1.5rem;
                margin-bottom: 1rem;
                max-height: 450px;
                overflow-y: auto;
                line-height: 1.7;
                font-size: 1rem;
                border-left: 4px solid #1f77b4;
            ">{passage_body}</div>""",
            unsafe_allow_html=True,
        )
        st.divider()
        _render_question_panel(question)
    else:
        q_bank = question.get("bank", "")
        if q_bank == "grammar":
            idx = int(st.session_state.get("queue_index", 0))
            reading_count = len(st.session_state.get("reading_order", []))
            gram_num = idx - reading_count + 1
            st.subheader(f"📝 GRAMMAR — Savol {gram_num}/{QUOTAS['Grammar']}")
        elif q_bank == "vocabulary":
            idx = int(st.session_state.get("queue_index", 0))
            reading_count = len(st.session_state.get("reading_order", []))
            gram_count = len(st.session_state.get("grammar_order", []))
            vocab_num = idx - reading_count - gram_count + 1
            st.subheader(f"📚 VOCABULARY — Savol {vocab_num}/{QUOTAS['Vocabulary']}")
        _render_question_panel(question)


def _ai_config() -> tuple[str, str, str] | None:
    if not st.session_state.get("api_validated"):
        return None
    key = (st.session_state.get("user_api_key") or "").strip()
    if not key:
        return None
    provider = st.session_state.get("ai_provider", "Anthropic")
    model = st.session_state.get("ai_model", ai_provider.default_model(provider))
    return provider, model, key


def _fetch_mistake_ai(mistake: dict) -> None:
    cfg = _ai_config()
    question, chosen = mistake["question"], mistake["choice"]
    prompt = llm_feedback.build_prompt(question, chosen)
    if cfg is None:
        mistake["ai"] = (
            "API kalit tasdiqlanmagan. Sidebar'dan kalitni kiriting va "
            "'API kalitini tekshirish' tugmasini bosing."
        )
        return
    provider, model, key = cfg
    try:
        mistake["ai"] = ai_provider.mistake_feedback(provider, key, model, prompt)
    except Exception:
        mistake["ai"] = ai_provider.FALLBACK_MESSAGE_UZ


def _render_ai_chat_panel(
    report: dict, correct: int, total: int, cfg: tuple[str, str, str]
) -> None:
    provider, model, key = cfg
    with st.container(border=True):
        st.subheader("AI murabbiy bilan suhbat")
        st.caption(f"Model: {provider} / {model}")
        context = (
            f"To'g'ri: {correct}/{total}, xato: {report['wrong']}, "
            f"IELTS taxmin: {report['overall_band']}, "
            f"zaifliklar: {[w['name'] for w in report['weaknesses'][:3]]}"
        )
        for msg in st.session_state.get("ai_chat", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if prompt := st.chat_input("AI murabbiyga savol bering..."):
            st.session_state.ai_chat.append({"role": "user", "content": prompt})
            reply = ai_provider.session_chat(
                provider, key, model, context, st.session_state.ai_chat[:-1], prompt
            )
            st.session_state.ai_chat.append({"role": "assistant", "content": reply})
            st.rerun()


def _fetch_results_ai(older: dict, newer: dict, cmp_: dict) -> str:
    cfg = _ai_config()
    if cfg is None:
        return (
            "AI tahlil uchun sidebar'dan API kalitni kiriting va "
            "'API kalitini tekshirish' tugmasini bosing."
        )
    provider, model, key = cfg
    prompt = session_history.build_results_ai_prompt(older, newer, cmp_)
    return ai_provider.session_chat(provider, key, model, prompt, [], prompt)


def _results_view() -> None:
    schema.init_db()
    st.title("📊 NATIJALAR / HISTORY")
    st.caption("Barcha sessiyalar saqlanadi — foydalanuvchi o'chira olmaydi.")
    if st.button("← Mashqqa qaytish", type="primary"):
        st.session_state.show_results = False
        st.rerun()

    rows = schema.list_session_results(st.session_state.user_id)
    if not rows:
        st.info("Hali saqlangan natija yo'q. Birinchi sessiyani yakunlang.")
        return

    if len(rows) >= 2:
        newer, older = rows[0], rows[1]
        cmp_ = session_history.compare_sessions(older, newer)
        st.subheader("📈 Taqqoslash (oxirgi vs oldingi sessiya)")
        _compare_banner(cmp_["message"], cmp_["improved"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            _summary_card(
                "Oldingi IELTS (taxmin)",
                _fmt_band(cmp_["older_band"]),
            )
        with m2:
            _summary_card(
                "Yangi IELTS (taxmin)",
                _fmt_band(cmp_["newer_band"]),
            )
        with m3:
            if cmp_["band_delta"] is not None:
                _summary_card("O'sish (ball)", f"{cmp_['band_delta']:+.1f}")
            else:
                _summary_card("O'sish (ball)", "—")
        with m4:
            if cmp_["band_ratio"] is not None:
                _summary_card("Barobar", f"{cmp_['band_ratio']:.2f}x")
            else:
                _summary_card(
                    "Aniqlik o'zgarishi",
                    f"{100 * cmp_['accuracy_delta']:+.0f}%",
                )

        tone_new = "good" if cmp_["improved"] else "bad"
        chart_dark = st.session_state.get("theme") == "night"
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(
                session_history.band_score_pie(
                    older,
                    f"ESKI · {session_history.format_completed_at(older['completed_at'])}",
                    tone="old",
                    dark=chart_dark,
                ),
                use_container_width=True,
            )
        with p2:
            st.plotly_chart(
                session_history.band_score_pie(
                    newer,
                    f"HOZIRGI · {session_history.format_completed_at(newer['completed_at'])}",
                    tone=tone_new,
                    dark=chart_dark,
                ),
                use_container_width=True,
            )

        st.markdown("#### Zaif tomonlar")
        w1, w2 = st.columns(2)
        with w1:
            st.markdown("**ESKI sessiya**")
            for line in session_history.top_weakness_lines(older):
                st.write(f"- {line}")
        with w2:
            st.markdown("**YANGI sessiya**")
            for line in session_history.top_weakness_lines(newer):
                st.write(f"- {line}")

        if st.button("🤖 AI tahlil — muammolar va xatolar", type="primary"):
            with st.spinner("AI tahlil qilmoqda..."):
                st.session_state.results_ai_text = _fetch_results_ai(older, newer, cmp_)
        if st.session_state.get("results_ai_text"):
            st.info(st.session_state.results_ai_text)

    st.divider()
    st.subheader("Barcha sessiyalar")
    for idx, row in enumerate(rows):
        with st.container(border=True):
            h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
            h1.markdown(
                f"**#{len(rows) - idx}** · "
                f"{session_history.format_completed_at(row['completed_at'])}"
            )
            h2.metric("Vaqt", session_history.format_duration(row["duration_secs"]))
            h3.metric("To'g'ri", row["correct"])
            h4.metric("Xato", row["wrong"])
            b1, b2, b3, b4 = st.columns(4)
            b1.write(f"**Jami savol:** {row['total']}")
            b2.write(f"**Aniqlik:** {100 * row['accuracy']:.0f}%")
            band = row["overall_band"]
            b3.write(f"**IELTS (taxmin):** {band if band is not None else '—'}")
            b4.write(
                f"R/G/V: {row['reading_band'] or '—'} / "
                f"{row['grammar_band'] or '—'} / {row['vocabulary_band'] or '—'}"
            )


def _deep_explanation(question: dict, chosen: int) -> str:
    """Generate a detailed explanation for a wrong answer."""
    correct_idx = question["correct_answer"]
    correct_opt = question["options"][correct_idx]
    chosen_opt = question["options"][chosen]
    correct_letter = chr(65 + correct_idx)
    chosen_letter = chr(65 + chosen)

    lines: list[str] = []

    lines.append("#### Batafsil tushuntirish")
    lines.append("")
    lines.append(f"**Siz tanladingiz:** {chosen_letter}) {chosen_opt} ❌")
    lines.append(f"**To'g'ri javob:** {correct_letter}) {correct_opt} ✅")
    lines.append("")

    lines.append("**Nima uchun sizning javobingiz xato?**")
    bank = question.get("bank", "")
    skill = question.get("skill_id", "")
    source = question.get("source", "")

    if bank == "grammar" or "sat_" in skill:
        if "boundaries" in skill.lower() or "boundaries" in source.lower():
            lines.append(
                f'Siz "{chosen_opt}" ni tanladingiz, lekin bu gapning tuzilishiga '
                f"mos kelmaydi. To'g'ri javob \"{correct_opt}\" — chunki bu yerda "
                f"tinish belgilari (vergul, nuqtali vergul, tire) to'g'ri ishlatilgan. "
                f"Mustaqil gaplar orasida nuqtali vergul (;) yoki nuqta (.) kerak, "
                f"bog'liq gaplar orasida esa vergul (,) ishlatiladi."
            )
        elif "transition" in skill.lower():
            lines.append(
                f'Siz "{chosen_opt}" ni tanladingiz, lekin bu so\'z matn oqimiga '
                f"mos kelmaydi. To'g'ri javob \"{correct_opt}\" — chunki bu transition "
                f"so'zi oldingi va keyingi gaplar orasidagi mantiqiy bog'lanishni "
                f"to'g'ri ifodalaydi. Transition so'zlarini o'rganayotganda: "
                f"however/but = zidlik, therefore/thus = sabab-natija, "
                f"moreover/furthermore = qo'shimcha, for example = misol."
            )
        elif "rhetorical" in skill.lower() or "synthesis" in skill.lower():
            lines.append(
                f'Siz "{chosen_opt}" ni tanladingiz, lekin bu savolda berilgan '
                f"notes/ma'lumotlarni eng to'g'ri umumlashtiradigan javob "
                f"\"{correct_opt}\" edi. Rhetorical Synthesis savollarida avval "
                f"savol nimani so'rayotganini aniqlang (compare, explain, emphasize), "
                f"keyin variantlardan shu maqsadga mos keluvchisini tanlang."
            )
        else:
            lines.append(
                f'Siz "{chosen_opt}" ni tanladingiz, lekin grammatik jihatdan '
                f"to'g'ri javob \"{correct_opt}\" edi. Grammar savollarida gapning "
                f"tuzilishiga e'tibor bering: subject-verb agreement, pronoun reference, "
                f"modifier placement, va parallel structure eng ko'p uchraydigan mavzular."
            )
        lines.append("")
        lines.append(
            "**Strategiya:** Har bir variantni gapga qo'yib o'qing. Grammatik va "
            "mantiqiy jihatdan eng tabiiy eshitiladigani — to'g'ri javob."
        )

    elif bank == "vocabulary":
        lines.append(
            f'Siz "{chosen_opt}" ni tanladingiz, lekin matn kontekstiga qarasangiz, '
            f"\"{correct_opt}\" eng mos javob. Vocabulary savollarida so'zning "
            f"lug'aviy ma'nosi emas, balki KONTEKSTDAGI ma'nosi muhim. "
            f"Bir so'z bir nechta ma'noga ega bo'lishi mumkin — matn nimani talab "
            f"qilayotganiga qarang."
        )
        lines.append("")
        lines.append(
            "**Strategiya:** Bo'sh joyni o'rab turgan gaplarni diqqat bilan o'qing. "
            "Ko'pincha oldingi yoki keyingi gapda javobga ishora (clue) bo'ladi. "
            "Tanish bo'lmagan so'zlarni ildiz (root), prefix va suffix orqali "
            "taxmin qilishga harakat qiling."
        )

    elif question.get("passage_id"):
        ex = question.get("exercise", 1)
        if ex == 1:
            lines.append(
                f'Siz "{chosen_opt}" ni tanladingiz, lekin passage matnida '
                f"berilgan ta'rifga eng mos keladigan so'z \"{correct_opt}\" edi. "
                f"Exercise 1 da so'z bankidagi so'zlarning ta'rifi beriladi — "
                f"siz passage matnidan o'sha ta'rifga mos keladigan so'zni topishingiz kerak."
            )
            lines.append("")
            lines.append(
                "**Strategiya:** Avval ta'rifni diqqat bilan o'qing, keyin passage "
                "matnida shu ma'noga ega bo'lgan so'zni qidiring. Sinonimlar va "
                "parafrazlarga e'tibor bering."
            )
        elif ex == 2:
            lines.append(
                f'Siz "{chosen_opt}" ni tanladingiz, lekin passage mazmuniga ko\'ra '
                f"to'g'ri javob \"{correct_opt}\" edi. Exercise 2 da passage haqidagi "
                f"savollar beriladi — javob doim passage matnida topiladi."
            )
            lines.append("")
            lines.append(
                "**Strategiya:** Savol nimani so'rayotganini aniq tushuning, keyin "
                "passage matnining tegishli qismini qayta o'qing. Javobni taxmin "
                "qilmang — dalil passage ichida bo'lishi kerak."
            )
        elif ex == 3:
            lines.append(
                f'Siz "{chosen_opt}" ni tanladingiz, lekin gapning grammatik '
                f"tuzilishi va ma'nosiga ko'ra \"{correct_opt}\" to'g'ri. "
                f"Exercise 3 da so'z bankidan foydalanib gaplarni to'ldirasiz."
            )
            lines.append("")
            lines.append(
                "**Strategiya:** Gapning grammatik tuzilishiga qarang — "
                "ot kerakmi, fe'lmi, sifatmi? So'z shakli (noun/verb/adjective) "
                "to'g'ri kelishini tekshiring, keyin ma'no jihatidan mosini tanlang."
            )
        else:
            lines.append(
                f'To\'g\'ri javob "{correct_opt}" edi. Passage matnini diqqat bilan '
                f"o'qib, javobni aniqlang."
            )
    else:
        lines.append(
            f'Siz "{chosen_opt}" ni tanladingiz, lekin to\'g\'ri javob '
            f'"{correct_opt}" edi. Savolni qaytadan diqqat bilan o\'qib, '
            f"nima uchun to'g'ri javob boshqacha ekanligini tushuning."
        )

    return "\n\n".join(lines)


def _summary_view() -> None:
    _save_session_to_history()
    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.title("Sessiya yakuni — natijalar")
    with top_r:
        _render_theme_toggle()
    secs = int(time.monotonic() - st.session_state.start_time)
    total = st.session_state.session_total
    correct = st.session_state.session_correct
    quotas = _session_quotas()
    report = session_report.build_session_report(
        st.session_state.bucket_stats,
        st.session_state.skill_stats,
        st.session_state.mastery,
        st.session_state.mistakes,
        secs,
        total,
        correct,
        quotas=quotas,
    )

    cfg = _ai_config()
    if cfg and st.session_state.pop("open_ai_on_summary", False):
        if not st.session_state.get("ai_chat"):
            st.session_state.ai_chat = [
                {
                    "role": "assistant",
                    "content": (
                        "Sessiya yakunlandi! Zaif tomonlaringiz va keyingi qadamlar "
                        "haqida savol bering — yordam beraman."
                    ),
                }
            ]

    if cfg:
        _render_ai_chat_panel(report, correct, total, cfg)
        st.divider()

    _summary_section_header("📊 Sessiya statistikasi")
    total_quota = report["total_quota"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _summary_card("Sarflangan vaqt", f"{secs // 60:02d}:{secs % 60:02d}")
    with c2:
        _summary_card("To'g'ri javoblar", f"{correct} / {total_quota}")
    with c3:
        _summary_card("Xato javoblar", str(total - correct))
    with c4:
        _summary_card(
            "Aniqlik",
            f"{100 * report['accuracy']:.0f}%" if total_quota else "—",
        )

    _summary_section_header(
        "🎯 Taxminiy IELTS bali (Speaking va Writing'siz)",
        "Bu ball faqat shu mashq sessiyasidagi Reading, Grammar va Vocabulary natijalaringizga "
        "asoslangan taxminiy ko'rsatkich (Speaking va Writing hisoblanmaydi). "
        f"Jami {total_quota} ta savoldan {correct} ta to'g'ri = {100 * report['accuracy']:.1f}% aniqlik.",
    )
    bands = report["bucket_bands"]
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        r_stats = st.session_state.bucket_stats.get("Reading", {})
        r_correct = r_stats.get("correct", 0)
        _summary_card("Reading", f"{_fmt_band(bands.get('Reading'))}")
    with b2:
        g_stats = st.session_state.bucket_stats.get("Grammar", {})
        g_correct = g_stats.get("correct", 0)
        _summary_card("Grammar", f"{_fmt_band(bands.get('Grammar'))}")
    with b3:
        v_stats = st.session_state.bucket_stats.get("Vocabulary", {})
        v_correct = v_stats.get("correct", 0)
        _summary_card("Vocabulary", f"{_fmt_band(bands.get('Vocabulary'))}")
    with b4:
        _summary_card("Umumiy o'rtacha", _fmt_band(report["overall_band"]))

    detail_c1, detail_c2, detail_c3 = st.columns(3)
    with detail_c1:
        r_q = quotas.get("Reading", 40)
        st.caption(f"Reading: {r_correct}/{r_q} to'g'ri ({100*r_correct/r_q:.0f}%)" if r_q else "")
    with detail_c2:
        g_q = quotas.get("Grammar", 50)
        st.caption(f"Grammar: {g_correct}/{g_q} to'g'ri ({100*g_correct/g_q:.0f}%)" if g_q else "")
    with detail_c3:
        v_q = quotas.get("Vocabulary", 50)
        st.caption(f"Vocabulary: {v_correct}/{v_q} to'g'ri ({100*v_correct/v_q:.0f}%)" if v_q else "")

    weakness_lines = []
    for w in report["weaknesses"]:
        line = (
            f"<strong>{w['name']}</strong> — "
            f"{w['correct']}/{w['quota']} to'g'ri ({100 * w['accuracy']:.0f}%)"
        )
        if w.get("unanswered", 0) > 0:
            line += f" · {w['unanswered']} ta javobsiz"
        weakness_lines.append(line)
    _summary_list_panel(
        "Zaif tomonlar",
        weakness_lines,
        "Bu sessiyada zaiflik aniqlanmadi.",
    )
    _summary_list_panel(
        "Tavsiyalar",
        [f"{rec}" for rec in report["recommendations"]],
        "Qo'shimcha tavsiya yo'q.",
    )

    st.divider()

    section_labels = {"Reading": "READING", "Grammar": "GRAMMAR", "Vocabulary": "VOCABULARY"}

    wrong_grouped: dict[str, list] = {"Reading": [], "Grammar": [], "Vocabulary": []}
    correct_grouped: dict[str, list] = {"Reading": [], "Grammar": [], "Vocabulary": []}
    answers = st.session_state.get("answers", {})
    queue = st.session_state.get("session_queue", [])
    for qid in queue:
        if qid not in answers:
            continue
        q = loader.get_question(qid)
        if q is None:
            continue
        chosen = answers[qid]
        cat = quota_bucket(loader.display_category(q["skill_id"]))
        if chosen == q["correct_answer"]:
            correct_grouped[cat].append({"question": q, "choice": chosen})
        else:
            wrong_grouped[cat].append({"question": q, "choice": chosen})

    total_wrong = sum(len(v) for v in wrong_grouped.values())
    total_correct = sum(len(v) for v in correct_grouped.values())

    # ── XATOLAR BO'LIMI ──
    st.subheader(f"❌ XATOLAR — {total_wrong} ta")
    if total_wrong > 0:
        for cat in PHASE_ORDER:
            items = wrong_grouped.get(cat, [])
            if not items:
                continue
            label = section_labels.get(cat, cat)
            st.markdown(f"### {label} — {len(items)} ta xato")

            for idx, mistake in enumerate(items):
                question, chosen = mistake["question"], mistake["choice"]
                prompt_text = _format_question_prompt(question)
                short = prompt_text[:120]
                if len(prompt_text) > 120:
                    short += "..."
                ex_info = ""
                if question.get("passage_id"):
                    ex = question.get("exercise", "?")
                    ex_info = f"[Exercise {ex}] "

                with st.expander(f"{idx + 1}. {ex_info}{short}", expanded=True):
                    st.markdown(f"**Savol:** {prompt_text}")
                    st.write("")
                    for i, option in enumerate(question["options"]):
                        letter = chr(65 + i)
                        if i == chosen:
                            st.error(f"{letter}) {option} — sizning javobingiz ❌")
                        elif i == question["correct_answer"]:
                            st.success(f"{letter}) {option} — to'g'ri javob ✅")
                        else:
                            st.write(f"  {letter}) {option}")
                    st.markdown("---")
                    explanation = _deep_explanation(question, chosen)
                    st.markdown(explanation)
                    st.markdown("**AI izohi**")
                    if "ai" in mistake:
                        st.info(mistake["ai"])
                    else:
                        m_from_state = next(
                            (m for m in st.session_state.mistakes if m["question"]["id"] == question["id"]),
                            None,
                        )
                        if m_from_state and "ai" in m_from_state:
                            st.info(m_from_state["ai"])
                        elif st.button("Izohni yuklash", key=f"ai_wrong_{cat}_{idx}"):
                            if m_from_state:
                                _fetch_mistake_ai(m_from_state)
                            st.rerun()
    else:
        st.success("Xato javob yo'q — barakalla! 🎉")

    st.divider()

    # ── TO'G'RI JAVOBLAR BO'LIMI ──
    st.subheader(f"✅ TO'G'RI JAVOBLAR — {total_correct} ta")
    if total_correct > 0:
        for cat in PHASE_ORDER:
            items = correct_grouped.get(cat, [])
            if not items:
                continue
            label = section_labels.get(cat, cat)
            st.markdown(f"### {label} — {len(items)} ta to'g'ri")

            for idx, entry in enumerate(items):
                question, chosen = entry["question"], entry["choice"]
                prompt_text = _format_question_prompt(question)
                short = prompt_text[:120]
                if len(prompt_text) > 120:
                    short += "..."
                ex_info = ""
                if question.get("passage_id"):
                    ex = question.get("exercise", "?")
                    ex_info = f"[Exercise {ex}] "

                with st.expander(f"{idx + 1}. {ex_info}{short}"):
                    st.markdown(f"**Savol:** {prompt_text}")
                    st.write("")
                    for i, option in enumerate(question["options"]):
                        letter = chr(65 + i)
                        if i == chosen:
                            st.success(f"{letter}) {option} — sizning javobingiz ✅")
                        else:
                            st.write(f"  {letter}) {option}")
    else:
        st.info("Bu sessiyada to'g'ri javob topilmadi.")

    if not cfg:
        st.info(
            "AI suhbati uchun sidebar'dan kompaniya, model va API kalitni "
            "kiritib tasdiqlang."
        )

    st.divider()
    st.markdown('<div class="new-session-marker"></div>', unsafe_allow_html=True)
    if st.button("Yangi sessiya boshlash", type="primary", use_container_width=True, key="btn_new_session"):
        _start_session(st.session_state.user_id, st.session_state.username)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    st.set_page_config(page_title="AdaptivPrep", layout="wide")
    _bridge_streamlit_secrets()
    _init_ui_prefs()

    if st.query_params.get("reset_token"):
        _apply_theme_css()
        _reset_password_view()
        return

    if "user_id" not in st.session_state:
        _login_view()
        return

    _apply_theme_css()

    if not loader.reading_bank_available():
        st.error(
            "O'qish banki (`data/reading_bank.json`) topilmadi yoki bo'sh. "
            "Terminalda `python scripts/ingest_els_full.py` buyrug'ini ishga tushiring."
        )
        if st.button("Chiqish"):
            st.session_state.clear()
            st.rerun()
        return

    if _session_needs_repair():
        _start_session(st.session_state.user_id, st.session_state.username)
        return

    _sidebar()
    if st.session_state.get("show_results"):
        _results_view()
    elif st.session_state.finished:
        _summary_view()
    elif st.session_state.get("current") is None:
        _finish_session()
        st.rerun()
    else:
        _quiz_view()


if __name__ == "__main__":
    main()
