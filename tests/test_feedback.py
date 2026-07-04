"""Unit tests for the AI Feedback Layer.

The Anthropic API is never called: a fake client object is injected in place
of the lazy singleton, so tests pin (a) prompt construction, (b) response
parsing, (c) the graceful-degradation contract, and (d) session caching.
"""
import anthropic
import httpx
import pytest

from src.feedback import llm_feedback

QUESTION = {
    "id": "grammar_articles_q1",
    "skill_id": "grammar_articles",
    "question_text": "She is ______ honest and reliable colleague.",
    "options": ["a", "an", "the", "(no article)"],
    "correct_answer": 1,
    "difficulty": "easy",
}


class FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeBlock(text)]


class FakeMessages:
    def __init__(self, result=None, error=None):
        self.calls = 0
        self._result = result
        self._error = error

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._result


class FakeClient:
    def __init__(self, result=None, error=None):
        self.messages = FakeMessages(result=result, error=error)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Fresh cache + configured key for every test; no real client leakage."""
    monkeypatch.setattr(llm_feedback, "_session_cache", {})
    monkeypatch.setattr(llm_feedback, "_clients", {})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    yield


def _http_error(status_code):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return anthropic.APIStatusError(
        f"error {status_code}", response=response, body=None
    )


def inject(monkeypatch, client):
    monkeypatch.setattr(llm_feedback, "_get_client", lambda *_a, **_k: client)
    return client


class TestPrompt:
    def test_prompt_contains_mistake_and_correct_answer(self):
        prompt = llm_feedback.build_prompt(QUESTION, chosen_index=0)
        assert QUESTION["question_text"] in prompt
        assert "A) a  <- the student chose this (WRONG)" in prompt
        assert "B) an  <- correct answer" in prompt
        assert "grammar_articles" in prompt

    def test_system_prompt_demands_uzbek_and_brevity(self):
        s = llm_feedback.SYSTEM_PROMPT
        assert "UZBEK" in s and "2-3" in s


class TestGetFeedback:
    def test_happy_path_returns_stripped_text(self, monkeypatch):
        client = inject(monkeypatch, FakeClient(FakeResponse("  Javob tushuntirishi.  ")))
        out = llm_feedback.get_feedback(QUESTION, 0)
        assert out == "Javob tushuntirishi."
        assert client.messages.kwargs["model"] == llm_feedback.MODEL_ID
        assert client.messages.kwargs["system"] == llm_feedback.SYSTEM_PROMPT

    def test_missing_api_key_returns_none_without_calling(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = inject(monkeypatch, FakeClient(FakeResponse("x")))
        assert llm_feedback.get_feedback(QUESTION, 0) is None
        assert client.messages.calls == 0

    @pytest.mark.parametrize("status", [401, 429, 500, 529])
    def test_http_errors_degrade_to_none(self, monkeypatch, status):
        inject(monkeypatch, FakeClient(error=_http_error(status)))
        assert llm_feedback.get_feedback(QUESTION, 0) is None

    def test_connection_error_degrades_to_none(self, monkeypatch):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        err = anthropic.APIConnectionError(request=request)
        inject(monkeypatch, FakeClient(error=err))
        assert llm_feedback.get_feedback(QUESTION, 0) is None

    def test_empty_response_degrades_to_none(self, monkeypatch):
        inject(monkeypatch, FakeClient(FakeResponse("   ")))
        assert llm_feedback.get_feedback(QUESTION, 0) is None

    def test_session_cache_avoids_second_call(self, monkeypatch):
        client = inject(monkeypatch, FakeClient(FakeResponse("Tushuntirish.")))
        first = llm_feedback.get_feedback(QUESTION, 0)
        second = llm_feedback.get_feedback(QUESTION, 0)
        assert first == second == "Tushuntirish."
        assert client.messages.calls == 1  # second answer served from cache

    def test_fallback_variant_always_returns_text(self, monkeypatch):
        inject(monkeypatch, FakeClient(error=_http_error(500)))
        out = llm_feedback.feedback_or_fallback(QUESTION, 0)
        assert out == llm_feedback.FALLBACK_MESSAGE_UZ


class TestKeyResolution:
    """Bring-your-own-key: session key > env var > disabled."""

    def test_session_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        assert llm_feedback._resolve_key("session-key") == "session-key"

    def test_env_var_is_the_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        assert llm_feedback._resolve_key(None) == "env-key"

    def test_missing_both_resolves_none_and_unconfigured(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert llm_feedback._resolve_key(None) is None
        assert not llm_feedback.is_configured()
        assert llm_feedback.is_configured("sk-session")  # session key alone enables

    def test_empty_session_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        assert llm_feedback._resolve_key("") == "env-key"

    def test_session_key_enables_feedback_without_env(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = inject(monkeypatch, FakeClient(FakeResponse("Izoh.")))
        assert llm_feedback.get_feedback(QUESTION, 0, api_key="sk-user") == "Izoh."
        assert client.messages.calls == 1

    def test_missing_both_returns_none_without_calling(self, monkeypatch):
        # Exact pre-BYO-key contract: no key anywhere -> None, zero API calls.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = inject(monkeypatch, FakeClient(FakeResponse("x")))
        assert llm_feedback.get_feedback(QUESTION, 0) is None
        assert llm_feedback.feedback_or_fallback(QUESTION, 0) == \
            llm_feedback.FALLBACK_MESSAGE_UZ
        assert client.messages.calls == 0
