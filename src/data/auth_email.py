"""Password-reset email delivery via SMTP (env, Streamlit secrets, or session override).

Required for live delivery: SMTP_HOST, SMTP_USER, SMTP_PASSWORD.
Optional: SMTP_PORT (587), SMTP_FROM, SMTP_USE_SSL (auto for 465), APP_BASE_URL.

Gmail: enable 2FA, create an App Password, then set SMTP_USER to your Gmail and
SMTP_PASSWORD to the 16-character app password.
"""
from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import Any


@dataclass(frozen=True)
class EmailSendResult:
    ok: bool
    message_uz: str


def _parse_port(raw: str | None) -> int:
    try:
        port = int(raw or "587")
    except (TypeError, ValueError):
        return 587
    return port if 1 <= port <= 65535 else 587


def _smtp_from_env() -> dict[str, Any]:
    return {
        "host": (os.environ.get("SMTP_HOST") or "").strip(),
        "port": _parse_port(os.environ.get("SMTP_PORT")),
        "user": (os.environ.get("SMTP_USER") or "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", ""),
        "from_addr": (os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER") or "").strip(),
        "use_ssl": os.environ.get("SMTP_USE_SSL", "").lower() in ("1", "true", "yes"),
    }


def smtp_configured(override: dict[str, Any] | None = None) -> bool:
    cfg = _merge_smtp(override)
    return bool(cfg["host"] and cfg["user"] and cfg["password"])


def _merge_smtp(override: dict[str, Any] | None) -> dict[str, Any]:
    cfg = _smtp_from_env()
    if override:
        for key in ("host", "port", "user", "password", "from_addr", "use_ssl"):
            if override.get(key) not in (None, ""):
                cfg[key] = override[key]
    if not cfg["from_addr"]:
        cfg["from_addr"] = cfg["user"]
    if cfg["port"] == 465:
        cfg["use_ssl"] = True
    return cfg


def validate_smtp(override: dict[str, Any] | None = None) -> EmailSendResult:
    """Test SMTP login without sending mail."""
    cfg = _merge_smtp(override)
    if not cfg["host"] or not cfg["user"]:
        return EmailSendResult(False, "SMTP host yoki email kiritilmagan.")
    if not cfg["password"]:
        return EmailSendResult(False, "SMTP parol (Gmail App Password) kiritilmagan.")
    try:
        _with_smtp(cfg, lambda s: None)
        return EmailSendResult(True, "Email server ulandi. Xabar yuborish mumkin.")
    except smtplib.SMTPAuthenticationError:
        return EmailSendResult(
            False,
            "Gmail autentifikatsiya xato. App Password to'g'riligini tekshiring "
            "(Google Account → Security → App passwords).",
        )
    except (smtplib.SMTPException, OSError, TimeoutError):
        return EmailSendResult(
            False,
            "Email serverga ulanib bo'lmadi. Host, port va internetni tekshiring.",
        )


def send_password_reset_email(
    to_email: str,
    reset_token: str,
    *,
    smtp_override: dict[str, Any] | None = None,
) -> EmailSendResult:
    """Send reset link. Returns status with Uzbek message for the UI."""
    cfg = _merge_smtp(smtp_override)
    if not cfg["host"] or not cfg["user"]:
        return EmailSendResult(False, "SMTP sozlanmagan.")
    if not cfg["password"]:
        return EmailSendResult(False, "SMTP parol kiritilmagan.")

    base = os.environ.get("APP_BASE_URL", "http://localhost:8501").rstrip("/")
    link = f"{base}/?reset_token={reset_token}"
    body = (
        "Salom,\n\n"
        "AdaptivPrep parolingizni tiklash uchun quyidagi havolani bosing:\n"
        f"{link}\n\n"
        "Havola 1 soat amal qiladi.\n\n"
        "Agar siz bu so'rovni yubormagan bo'lsangiz, xabarni e'tiborsiz qoldiring.\n"
    )
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = "AdaptivPrep — parolni tiklash"
    msg["From"] = cfg["from_addr"]
    msg["To"] = to_email

    try:
        def _send(server: smtplib.SMTP) -> None:
            server.sendmail(cfg["from_addr"], [to_email], msg.as_string())

        _with_smtp(cfg, _send)
        return EmailSendResult(
            True,
            f"Tiklash havolasi {to_email} manziliga yuborildi. Spam papkasini ham tekshiring.",
        )
    except smtplib.SMTPAuthenticationError:
        return EmailSendResult(
            False,
            "Gmail App Password noto'g'ri. Google Account → Security → App passwords "
            "bo'limidan yangi parol yarating.",
        )
    except (smtplib.SMTPException, OSError, TimeoutError):
        return EmailSendResult(
            False,
            "Email yuborib bo'lmadi. SMTP sozlamalarini qayta tekshiring.",
        )


def _with_smtp(cfg: dict[str, Any], action) -> None:
    host = cfg["host"]
    port = int(cfg["port"])
    user = cfg["user"]
    password = cfg["password"]
    context = ssl.create_default_context()

    if cfg["use_ssl"] or port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=20, context=context) as server:
            server.login(user, password)
            action(server)
        return

    with smtplib.SMTP(host, port, timeout=20) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(user, password)
        action(server)


def gmail_defaults_for(email: str) -> dict[str, Any]:
    """Prefill Gmail SMTP when user enters a @gmail.com address."""
    if email.lower().endswith("@gmail.com"):
        return {"host": "smtp.gmail.com", "port": 587, "user": email.strip(), "use_ssl": False}
    return {}
