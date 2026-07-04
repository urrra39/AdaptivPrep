"""AI Feedback Layer - mistake-targeted explanations via the Anthropic API.

When a learner answers incorrectly, a static "wrong, the answer was B" tells
them *that* they failed but not *why*.  This module asks Claude to produce a
short explanation targeting the specific distractor the learner chose - the
pedagogical moment where feedback has the highest value (immediately after an
error, addressed to the actual misconception rather than the topic at large).

Engineering choices, and why:

* **Model:** ``claude-sonnet-4-6`` - the latest production Sonnet.  Feedback
  generation is a short, well-scoped task; Sonnet's speed/cost profile
  ($3/MTok in) fits a per-mistake call far better than an Opus-tier model,
  and quality is more than sufficient for 2-3 sentence explanations.
* **Language split:** the prompt and code are English; the *output* is Uzbek
  (Latin script) - the project's audience is the Uzbek IELTS-prep community,
  and mother-tongue explanations of English-language mistakes are exactly
  the gap existing tools leave open.
* **Graceful degradation:** the tutor must never crash the quiz.  Every API
  failure mode (auth, rate limit, HTTP error, network/timeout) is caught,
  logged in English for operators, and surfaced to the learner as a static
  Uzbek fallback.  ``get_feedback`` returns ``None`` on failure by contract.
* **Session cache:** identical (question, wrong choice) pairs within a
  process reuse the first explanation - a learner re-hitting the same
  distractor should not cost a second API call.
* **Timeout discipline:** the call sits on the interactive path of a quiz,
  so we cap the request at 15 s with a single retry instead of the SDK's
  10-minute default - a fast fallback beats a hanging spinner.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

from src.data import loader

# Pull ANTHROPIC_API_KEY from a local .env (git-ignored) into the process env.
load_dotenv()

logger = logging.getLogger(__name__)

MODEL_ID = "claude-sonnet-4-6"  # latest production Sonnet (verified 2026-07)
MAX_TOKENS = 300                # 2-3 sentences; hard ceiling against runaway cost
REQUEST_TIMEOUT_S = 15.0        # interactive path: fail fast, show fallback
MAX_RETRIES = 1                 # one retry on transient errors, then degrade

# Static fallback shown when the API is unavailable (Uzbek, per language strategy).
FALLBACK_MESSAGE_UZ = (
    "Tushuntirish hozircha mavjud emas. To'g'ri javobni diqqat bilan ko'rib "
    "chiqing va shu mavzuni yana bir bor takrorlashni tavsiya qilamiz."
)

# The system prompt pins persona, length, and output language.  Output is
# Uzbek; English is kept only for the linguistic material being taught.
SYSTEM_PROMPT = (
    "You are an experienced, encouraging IELTS tutor for Uzbek-speaking "
    "students. A student has just answered a multiple-choice question "
    "incorrectly. Explain their specific mistake in UZBEK (Latin script): "
    "first why the option they chose is wrong, then why the correct option "
    "is right. Keep the English words, phrases or grammar terms under "
    "discussion in English. Write exactly 2-3 short sentences. No greeting, "
    "no preamble - start directly with the explanation."
)

# Deliberately shared across Streamlit sessions - and thread-safe: the
# Anthropic client is stateless over a thread-safe httpx.Client, cache
# entries are user-independent ((question_id, choice) -> text), and dict
# get/set are atomic under the GIL.  Worst-case races are benign: one
# redundant client construction, or an identical cache value rewritten.
# _clients is keyed by API key so bring-your-own-key sessions each get their
# own client; keys live only in this dict and are never logged or persisted.
_clients: dict = {}
_session_cache: dict = {}


def _resolve_key(api_key: Optional[str] = None) -> Optional[str]:
    """Key resolution priority: session-provided key > environment > None."""
    return api_key or os.environ.get("ANTHROPIC_API_KEY") or None


def is_configured(api_key: Optional[str] = None) -> bool:
    """True when an API key is available (feature is enabled)."""
    return _resolve_key(api_key) is not None


def _get_client(api_key: str) -> anthropic.Anthropic:
    """Lazy per-key client cache (one client per distinct API key)."""
    client = _clients.get(api_key)
    if client is None:
        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=REQUEST_TIMEOUT_S,
            max_retries=MAX_RETRIES,
        )
        _clients[api_key] = client
    return client


def build_prompt(question: dict, chosen_index: int) -> str:
    """Render the user message describing the exact mistake.

    Pure function (no I/O) so tests can pin the prompt content without
    touching the network.
    """
    options = question["options"]
    correct_index = question["correct_answer"]
    lines = [
        f"Skill: {loader.skill_name(question['skill_id'])}",
    ]
    if question.get("passage_text"):
        excerpt = question["passage_text"]
        if len(excerpt) > 600:
            excerpt = excerpt[:600] + "..."
        lines.append(f"Reading passage (excerpt): {excerpt}")
    lines.extend([
        f"Question: {question['question_text']}",
        "Options:",
    ])
    for i, option in enumerate(options):
        marker = ""
        if i == chosen_index:
            marker = "  <- the student chose this (WRONG)"
        elif i == correct_index:
            marker = "  <- correct answer"
        lines.append(f"  {chr(65 + i)}) {option}{marker}")
    lines.append(f"Difficulty: {question.get('difficulty', 'unknown')}")
    return "\n".join(lines)


def get_feedback(
    question: dict,
    chosen_index: int,
    use_cache: bool = True,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Generate a mistake-specific explanation in Uzbek.

    ``api_key`` (a session-provided, bring-your-own key) takes priority over
    the ANTHROPIC_API_KEY environment variable.  Returns the explanation
    text, or ``None`` on any failure (missing key, auth, rate limit, HTTP
    error, network/timeout) - callers show ``FALLBACK_MESSAGE_UZ`` in that
    case.  Failures are logged in English; the key itself is never logged.
    """
    key = _resolve_key(api_key)
    if key is None:
        logger.info("No API key available (session or env); AI feedback disabled.")
        return None

    cache_key = (question["id"], int(chosen_index))
    if use_cache and cache_key in _session_cache:
        return _session_cache[cache_key]

    try:
        response = _get_client(key).messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            # Low temperature: pedagogically consistent phrasing across
            # learners (sampling params remain valid on Sonnet 4.6).
            temperature=0.3,
            messages=[{"role": "user", "content": build_prompt(question, chosen_index)}],
        )
    # Most-specific first; every branch degrades to the static fallback.
    except anthropic.AuthenticationError:
        logger.error("Anthropic auth failed - check ANTHROPIC_API_KEY.")
        return None
    except anthropic.RateLimitError:
        logger.warning("Anthropic rate limit hit; serving fallback feedback.")
        return None
    except anthropic.APIStatusError as exc:
        logger.warning("Anthropic API error %s: %s", exc.status_code, exc.message)
        return None
    except anthropic.APIConnectionError:  # includes APITimeoutError
        logger.warning("Anthropic connection/timeout error; serving fallback.")
        return None

    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text or not text.strip():
        logger.warning("Anthropic response contained no text block.")
        return None

    text = text.strip()
    _session_cache[cache_key] = text
    return text


def feedback_or_fallback(
    question: dict, chosen_index: int, api_key: Optional[str] = None
) -> str:
    """Always-text variant: the explanation, or the static Uzbek fallback."""
    return get_feedback(question, chosen_index, api_key=api_key) or FALLBACK_MESSAGE_UZ
