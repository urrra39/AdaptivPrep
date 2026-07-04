"""Multi-provider AI layer: validation, mistake feedback, and session chat.

Supports bring-your-own-key for Anthropic and OpenAI.  Validation performs a
minimal live API call so invalid keys and empty balances surface immediately
with Uzbek error strings suitable for the sidebar UI.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 15.0

FALLBACK_MESSAGE_UZ = (
    "Tushuntirish hozircha mavjud emas. To'g'ri javobni diqqat bilan ko'rib "
    "chiqing va shu mavzuni yana bir bor takrorlashni tavsiya qilamiz."
)

MISTAKE_SYSTEM = (
    "You are an experienced IELTS tutor for Uzbek-speaking students. "
    "Explain the student's specific mistake in UZBEK (Latin script): why their "
    "choice is wrong and why the correct answer is right. 2-3 short sentences. "
    "Keep English terms under discussion in English. No greeting."
)

CHAT_SYSTEM = (
    "You are an IELTS tutor for Uzbek-speaking students. The student just "
    "finished a practice session. Answer in UZBEK (Latin script), be specific "
    "about weaknesses and study advice. Keep English exam terms in English."
)

_BALANCE_HINTS = re.compile(
    r"insufficient.?quota|credit.?balance|billing|payment.?required|"
    r"exceeded.?your.?current.?quota|insufficient.?funds|add.?credits|"
    r"purchase.?credits|out.?of.?credits",
    re.I,
)

MSG_INVALID_KEY_UZ = "Bu API key emas yoki noto'g'ri."
MSG_NO_BALANCE_UZ = "PUL yo'q — API key hisobida mablag' yo'q."
MSG_GENERIC_FAIL_UZ = "API bilan bog'lanib bo'lmadi. Keyinroq urinib ko'ring."
MSG_MODEL_FAIL_UZ = "Tanlangan model mavjud emas. Sidebar'dan boshqa model tanlang."
MSG_OK_UZ = "API ishladi — iltimos sessiyani tugatgandan keyin ishlating!!!"

PROVIDERS: dict[str, dict] = {
    "Anthropic": {
        # Newest → oldest (7 models)
        "models": [
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-5",
        ],
        "default_model": "claude-sonnet-4-6",
    },
    "OpenAI": {
        "models": [
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.2",
            "gpt-5-mini",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "default_model": "gpt-4o-mini",
    },
}


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a live API-key probe."""

    ok: bool
    message_uz: str


def default_model(provider: str) -> str:
    return PROVIDERS[provider]["default_model"]


def _is_balance_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if _BALANCE_HINTS.search(text):
        return True
    status = getattr(exc, "status_code", None)
    if status == 402:
        return True
    # OpenAI: insufficient_quota is a billing issue; generic 429 is often rate limit.
    code = getattr(exc, "code", None)
    if code == "insufficient_quota":
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", {})
        if err.get("code") == "insufficient_quota":
            return True
        if _BALANCE_HINTS.search(str(err.get("message", ""))):
            return True
    return False


def validate_key(provider: str, api_key: str, model: str) -> ValidationResult:
    """Probe ``api_key`` with a minimal completion; never log the key."""
    key = (api_key or "").strip()
    if not key:
        return ValidationResult(False, MSG_INVALID_KEY_UZ)
    if provider not in PROVIDERS:
        return ValidationResult(False, MSG_GENERIC_FAIL_UZ)
    if model not in PROVIDERS[provider]["models"]:
        model = default_model(provider)
    try:
        if provider == "Anthropic":
            anthropic.Anthropic(api_key=key, timeout=REQUEST_TIMEOUT_S).messages.create(
                model=model,
                max_tokens=8,
                messages=[{"role": "user", "content": "ping"}],
            )
        else:
            from openai import OpenAI

            OpenAI(api_key=key, timeout=REQUEST_TIMEOUT_S).chat.completions.create(
                model=model,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
    except anthropic.AuthenticationError:
        return ValidationResult(False, MSG_INVALID_KEY_UZ)
    except anthropic.NotFoundError:
        return ValidationResult(False, MSG_MODEL_FAIL_UZ)
    except Exception as exc:
        if _is_balance_error(exc):
            return ValidationResult(False, MSG_NO_BALANCE_UZ)
        err_name = type(exc).__name__
        if "Authentication" in err_name or "401" in str(exc):
            return ValidationResult(False, MSG_INVALID_KEY_UZ)
        if "NotFound" in err_name or "404" in str(exc):
            return ValidationResult(False, MSG_MODEL_FAIL_UZ)
        if "Permission" in err_name or "403" in str(exc):
            return ValidationResult(False, MSG_INVALID_KEY_UZ)
        logger.warning("API validation failed (%s): %s", provider, err_name)
        return ValidationResult(False, MSG_GENERIC_FAIL_UZ)
    return ValidationResult(True, MSG_OK_UZ)


def _anthropic_text(provider: str, api_key: str, model: str, system: str, user: str) -> Optional[str]:
    try:
        resp = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_S).messages.create(
            model=model,
            max_tokens=400,
            system=system,
            temperature=0.3,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError:
        return None
    except Exception as exc:
        if _is_balance_error(exc):
            return None
        logger.warning("Anthropic call failed: %s", type(exc).__name__)
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    return text.strip() if text else None


def _openai_text(api_key: str, model: str, system: str, user: str) -> Optional[str]:
    try:
        from openai import OpenAI

        resp = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_S).chat.completions.create(
            model=model,
            max_tokens=400,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:
        if _is_balance_error(exc):
            return None
        err_name = type(exc).__name__
        if "Authentication" in err_name:
            return None
        logger.warning("OpenAI call failed: %s", err_name)
        return None
    text = resp.choices[0].message.content if resp.choices else None
    return text.strip() if text else None


def complete(
    provider: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
) -> Optional[str]:
    """Single-turn completion; ``None`` on failure."""
    if provider == "Anthropic":
        return _anthropic_text(provider, api_key, model, system, user)
    return _openai_text(api_key, model, system, user)


def mistake_feedback(
    provider: str,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    return complete(provider, api_key, model, MISTAKE_SYSTEM, prompt) or FALLBACK_MESSAGE_UZ


def session_chat(
    provider: str,
    api_key: str,
    model: str,
    context: str,
    history: list[dict],
    user_message: str,
) -> str:
    """Multi-turn chat grounded in session context."""
    if provider == "Anthropic":
        system = f"{CHAT_SYSTEM}\n\nSession context:\n{context}"
        messages = [{"role": turn["role"], "content": turn["content"]} for turn in history]
        messages.append({"role": "user", "content": user_message})
        try:
            resp = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_S).messages.create(
                model=model,
                max_tokens=500,
                system=system,
                temperature=0.4,
                messages=messages,
            )
        except Exception:
            return FALLBACK_MESSAGE_UZ
        text = next((b.text for b in resp.content if b.type == "text"), None)
        return text.strip() if text else FALLBACK_MESSAGE_UZ

    from openai import OpenAI

    oai_messages = [{"role": "system", "content": f"{CHAT_SYSTEM}\n\nSession context:\n{context}"}]
    for turn in history:
        oai_messages.append({"role": turn["role"], "content": turn["content"]})
    oai_messages.append({"role": "user", "content": user_message})
    try:
        resp = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_S).chat.completions.create(
            model=model,
            max_tokens=500,
            temperature=0.4,
            messages=oai_messages,
        )
    except Exception:
        return FALLBACK_MESSAGE_UZ
    text = resp.choices[0].message.content if resp.choices else None
    return text.strip() if text else FALLBACK_MESSAGE_UZ
