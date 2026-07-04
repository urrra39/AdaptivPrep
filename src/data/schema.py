"""SQLite persistence layer for AdaptivPrep.

Defines the database schema (``users``, ``responses``) and small helper
functions used by the quiz app and, later, by the knowledge-tracing models
to reconstruct a learner's history.

The response log is the single source of truth: Bayesian Knowledge Tracing
(Phase 3) replays these rows to estimate per-skill mastery, so every answer
is stored together with its skill, correctness, latency and timestamp.

Content bank contract (``data/questions.json``, not stored in SQLite):
    Required keys per item: ``id``, ``skill_id``, ``question_text``,
    ``options`` (list of 4 or 5 strings), ``correct_answer`` (0-based index),
    ``difficulty`` (``easy`` | ``medium`` | ``hard``).
    Optional keys: ``passage_text`` (long-form reading passage rendered in the
    split-pane IELTS layout when present), ``source`` (provenance citation).
    Responses reference items by ``question_id`` only; passage text lives in
    the static JSON bank and is joined at render time via ``loader``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

# Default DB location: <project_root>/data/responses.db (git-ignored).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "responses.db"

PathLike = Union[str, Path]

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL,
    pin_hash    TEXT
);

CREATE TABLE IF NOT EXISTS responses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL,
    question_id       TEXT    NOT NULL,
    skill_id          TEXT    NOT NULL,
    correct           INTEGER NOT NULL CHECK (correct IN (0, 1)),
    response_time_ms  INTEGER,
    timestamp         TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE INDEX IF NOT EXISTS idx_responses_user       ON responses (user_id);
CREATE INDEX IF NOT EXISTS idx_responses_skill      ON responses (skill_id);
CREATE INDEX IF NOT EXISTS idx_responses_user_skill ON responses (user_id, skill_id);

CREATE TABLE IF NOT EXISTS session_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL,
    completed_at        TEXT    NOT NULL,
    duration_secs       INTEGER NOT NULL,
    total               INTEGER NOT NULL,
    correct             INTEGER NOT NULL,
    wrong               INTEGER NOT NULL,
    accuracy            REAL    NOT NULL,
    overall_band        REAL,
    reading_band        REAL,
    grammar_band        REAL,
    vocabulary_band     REAL,
    bucket_stats_json   TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE INDEX IF NOT EXISTS idx_session_results_user
    ON session_results (user_id, completed_at DESC);
"""


def _resolve_db_path(db_path: Optional[PathLike] = None) -> Path:
    """Resolve the DB path from an explicit arg, the ADAPTIVPREP_DB env var, or the default."""
    if db_path is not None:
        return Path(db_path)
    return Path(os.environ.get("ADAPTIVPREP_DB", str(DEFAULT_DB_PATH)))


def get_connection(db_path: Optional[PathLike] = None) -> sqlite3.Connection:
    """Open a connection with row access by name and foreign keys enforced."""
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[PathLike] = None) -> None:
    """Create tables and indexes if they do not already exist (idempotent)."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        # Migration for DBs created before pin_hash existed: add the column if
        # missing.  ALTER TABLE ADD COLUMN is the sqlite-supported, in-place
        # migration; the try/except keeps init_db idempotent on new DBs where
        # the CREATE TABLE above already includes the column.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
        migrations = {
            "pin_hash": "TEXT",
            "email": "TEXT",
            "password_hash": "TEXT",
            "display_name": "TEXT",
            "reset_token": "TEXT",
            "reset_token_expires": "TEXT",
        }
        for col, typ in migrations.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col} {typ}")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS session_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             INTEGER NOT NULL,
                completed_at        TEXT    NOT NULL,
                duration_secs       INTEGER NOT NULL,
                total               INTEGER NOT NULL,
                correct             INTEGER NOT NULL,
                wrong               INTEGER NOT NULL,
                accuracy            REAL    NOT NULL,
                overall_band        REAL,
                reading_band        REAL,
                grammar_band        REAL,
                vocabulary_band     REAL,
                bucket_stats_json   TEXT    NOT NULL,
                weaknesses_json     TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
            CREATE INDEX IF NOT EXISTS idx_session_results_user
                ON session_results (user_id, completed_at DESC);
            """
        )
        result_cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(session_results)")
        }
        if "weaknesses_json" not in result_cols:
            conn.execute(
                "ALTER TABLE session_results ADD COLUMN weaknesses_json TEXT"
            )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email "
            "ON users (email) WHERE email IS NOT NULL"
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# PIN hashing (collision prevention for a public demo - not full auth)        #
# --------------------------------------------------------------------------- #
# PBKDF2-HMAC-SHA256 with a per-user random salt.  A 4-digit PIN has only
# 10k possibilities, so this is a speed bump against casual name collisions,
# not a defence against offline brute force - documented as such rather than
# oversold.  Stored format: "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>".
_PBKDF2_ITERATIONS = 100_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_RESET_TTL_HOURS = 1


def hash_pin(pin: str, salt: Optional[bytes] = None) -> str:
    """Return a salted PBKDF2 hash string for ``pin``."""
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def _check_pin(pin: str, stored: str) -> bool:
    """Constant-time verification of ``pin`` against a stored hash string."""
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac(
            "sha256", pin.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BadPinError(Exception):
    """Raised when an existing user's PIN does not match on login."""


def get_user(username: str, db_path: Optional[PathLike] = None) -> Optional[sqlite3.Row]:
    """Return the users row (incl. pin_hash) for ``username``, or None."""
    conn = get_connection(db_path)
    try:
        return conn.execute(
            "SELECT id, username, pin_hash FROM users WHERE username = ?",
            ((username or "").strip(),),
        ).fetchone()
    finally:
        conn.close()


def verify_pin(user_id: int, pin: str, db_path: Optional[PathLike] = None) -> bool:
    """True iff ``pin`` matches the user's stored hash. False if none is set."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT pin_hash FROM users WHERE id = ?", (int(user_id),)
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["pin_hash"] is None:
        return False
    return _check_pin(pin, row["pin_hash"])


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Return a salted PBKDF2 hash for an account password (min 8 chars)."""
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    return hash_pin(password, salt=salt)


def _normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise ValueError("invalid email address")
    return email


class AuthError(Exception):
    """Raised on failed email/password authentication."""


class EmailTakenError(Exception):
    """Raised when registering an email that already exists."""


class UsernameTakenError(Exception):
    """Raised when display name / username is already taken."""


def get_user_by_email(email: str, db_path: Optional[PathLike] = None) -> Optional[sqlite3.Row]:
    conn = get_connection(db_path)
    try:
        return conn.execute(
            "SELECT id, username, email, password_hash, display_name, pin_hash "
            "FROM users WHERE email = ?",
            (_normalize_email(email),),
        ).fetchone()
    except ValueError:
        return None
    finally:
        conn.close()


def register_user(
    email: str,
    password: str,
    display_name: str,
    db_path: Optional[PathLike] = None,
) -> int:
    """Create an email/password account; ``username`` mirrors ``display_name``."""
    email = _normalize_email(email)
    display_name = (display_name or "").strip()
    if not display_name:
        raise ValueError("display name required")
    if get_user_by_email(email, db_path=db_path) is not None:
        raise EmailTakenError("email already registered")
    if get_user(display_name, db_path=db_path) is not None:
        raise UsernameTakenError("username already taken")
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """INSERT INTO users
                   (username, created_at, email, password_hash, display_name)
               VALUES (?, ?, ?, ?, ?)""",
            (display_name, _utc_now_iso(), email, hash_password(password), display_name),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def authenticate_user(
    email: str, password: str, db_path: Optional[PathLike] = None
) -> tuple[int, str]:
    """Return (user_id, display_name) when credentials match."""
    row = get_user_by_email(email, db_path=db_path)
    if row is None or row["password_hash"] is None:
        raise AuthError("invalid credentials")
    if not _check_pin(password, row["password_hash"]):
        raise AuthError("invalid credentials")
    return int(row["id"]), row["display_name"] or row["username"]


def create_password_reset_token(email: str, db_path: Optional[PathLike] = None) -> Optional[str]:
    """Issue a one-hour reset token for ``email``; None if account unknown."""
    row = get_user_by_email(email, db_path=db_path)
    if row is None:
        return None
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(hours=_RESET_TTL_HOURS)).isoformat(
        timespec="seconds"
    )
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE users SET reset_token = ?, reset_token_expires = ? WHERE id = ?",
            (token, expires, int(row["id"])),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def reset_password_with_token(
    token: str, new_password: str, db_path: Optional[PathLike] = None
) -> bool:
    """Set a new password when ``token`` is valid and not expired."""
    token = (token or "").strip()
    if not token or len(new_password) < 8:
        return False
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id, reset_token_expires FROM users WHERE reset_token = ?", (token,)
        ).fetchone()
        if row is None or row["reset_token_expires"] is None:
            return False
        expires = datetime.fromisoformat(row["reset_token_expires"])
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires:
            return False
        conn.execute(
            "UPDATE users SET password_hash = ?, reset_token = NULL, "
            "reset_token_expires = NULL WHERE id = ?",
            (hash_password(new_password), int(row["id"])),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_or_create_user(
    username: str, pin: Optional[str] = None, db_path: Optional[PathLike] = None
) -> int:
    """Return the id of ``username``, creating the row on first sight.

    ``pin`` gates access when the username already exists (collision
    prevention for a public demo):

    * ``pin=None`` - legacy behaviour: create-or-return, no PIN check. Used by
      analytics/CLI paths that already hold a trusted user id.
    * new username - the row is created; a supplied ``pin`` is stored (hashed).
    * existing username with **no** stored PIN (legacy row or created without
      one) - a supplied ``pin`` is set now; without one, access is allowed.
    * existing username **with** a stored PIN - the ``pin`` must match, else
      :class:`BadPinError` is raised (the account is not reachable). The PIN
      never appears in the exception.
    """
    username = (username or "").strip()
    if not username:
        raise ValueError("username must be a non-empty string")
    if pin is not None and (len(pin) != 4 or not pin.isdigit()):
        raise ValueError("pin must be exactly 4 digits")
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id, pin_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO users (username, created_at, pin_hash) VALUES (?, ?, ?)",
                (username, _utc_now_iso(), hash_pin(pin) if pin else None),
            )
            conn.commit()
            return int(cur.lastrowid)

        user_id = int(row["id"])
        if row["pin_hash"] is None:
            if pin is not None:  # first PIN for a legacy/pin-less account
                conn.execute(
                    "UPDATE users SET pin_hash = ? WHERE id = ?",
                    (hash_pin(pin), user_id),
                )
                conn.commit()
            return user_id
        if pin is not None and not _check_pin(pin, row["pin_hash"]):
            raise BadPinError("incorrect PIN")
        return user_id
    finally:
        conn.close()


def record_response(
    user_id: int,
    question_id: str,
    skill_id: str,
    correct: bool,
    response_time_ms: Optional[int] = None,
    db_path: Optional[PathLike] = None,
) -> int:
    """Persist a single answered question and return its row id."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """INSERT INTO responses
                   (user_id, question_id, skill_id, correct, response_time_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                int(user_id),
                str(question_id),
                str(skill_id),
                1 if correct else 0,
                None if response_time_ms is None else int(response_time_ms),
                _utc_now_iso(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_responses(
    user_id: int,
    skill_id: Optional[str] = None,
    db_path: Optional[PathLike] = None,
) -> list:
    """Return a user's responses (optionally filtered by skill), oldest first."""
    conn = get_connection(db_path)
    try:
        if skill_id is None:
            rows = conn.execute(
                "SELECT * FROM responses WHERE user_id = ? ORDER BY id",
                (int(user_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM responses WHERE user_id = ? AND skill_id = ? ORDER BY id",
                (int(user_id), str(skill_id)),
            ).fetchall()
        return list(rows)
    finally:
        conn.close()


def get_user_stats(user_id: int, db_path: Optional[PathLike] = None) -> dict:
    """Return {'answered', 'correct', 'accuracy'} aggregated over all responses."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(correct), 0) AS n_correct "
            "FROM responses WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
        n = int(row["n"])
        n_correct = int(row["n_correct"])
        return {
            "answered": n,
            "correct": n_correct,
            "accuracy": (n_correct / n) if n else 0.0,
        }
    finally:
        conn.close()


def save_session_result(
    user_id: int,
    report: dict,
    duration_secs: int,
    db_path: Optional[PathLike] = None,
) -> int:
    """Persist a completed quiz session. Users cannot delete rows from the app."""
    bands = report.get("bucket_bands") or {}
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """INSERT INTO session_results
                   (user_id, completed_at, duration_secs, total, correct, wrong,
                    accuracy, overall_band, reading_band, grammar_band,
                    vocabulary_band, bucket_stats_json, weaknesses_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(user_id),
                _utc_now_iso(),
                int(duration_secs),
                int(report.get("total") or 0),
                int(report.get("correct") or 0),
                int(report.get("wrong") or 0),
                float(report.get("accuracy") or 0.0),
                report.get("overall_band"),
                bands.get("Reading"),
                bands.get("Grammar"),
                bands.get("Vocabulary"),
                json.dumps(report.get("bucket_stats") or {}, ensure_ascii=False),
                json.dumps(report.get("weaknesses") or [], ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_session_results(
    user_id: int,
    limit: int = 100,
    db_path: Optional[PathLike] = None,
) -> list[dict]:
    """Return a user's saved session results, newest first (read-only)."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT id, completed_at, duration_secs, total, correct, wrong,
                      accuracy, overall_band, reading_band, grammar_band,
                      vocabulary_band, bucket_stats_json, weaknesses_json
               FROM session_results
               WHERE user_id = ?
               ORDER BY completed_at DESC, id DESC
               LIMIT ?""",
            (int(user_id), int(limit)),
        ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "completed_at": r["completed_at"],
                    "duration_secs": int(r["duration_secs"]),
                    "total": int(r["total"]),
                    "correct": int(r["correct"]),
                    "wrong": int(r["wrong"]),
                    "accuracy": float(r["accuracy"]),
                    "overall_band": r["overall_band"],
                    "reading_band": r["reading_band"],
                    "grammar_band": r["grammar_band"],
                    "vocabulary_band": r["vocabulary_band"],
                    "bucket_stats": json.loads(r["bucket_stats_json"] or "{}"),
                    "weaknesses": json.loads(r["weaknesses_json"] or "[]"),
                }
            )
        return out
    finally:
        conn.close()
