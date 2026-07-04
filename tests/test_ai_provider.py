"""Tests for multi-provider AI validation and chat helpers."""
from types import SimpleNamespace

from src.feedback import ai_provider


class TestAiProviderModels:
    def test_seven_models_per_provider(self):
        assert len(ai_provider.PROVIDERS["Anthropic"]["models"]) == 7
        assert len(ai_provider.PROVIDERS["OpenAI"]["models"]) == 7

    def test_models_newest_first(self):
        assert ai_provider.PROVIDERS["Anthropic"]["models"][0] == "claude-opus-4-8"
        assert ai_provider.PROVIDERS["OpenAI"]["models"][0] == "gpt-5.5"

    def test_ok_message_mentions_session_end(self):
        assert "sessiyani tugatgandan keyin" in ai_provider.MSG_OK_UZ.lower()


class TestBalanceDetection:
    def test_rate_limit_without_quota_is_not_balance(self):
        exc = SimpleNamespace(status_code=429)
        assert not ai_provider._is_balance_error(exc)

    def test_insufficient_quota_is_balance(self):
        exc = SimpleNamespace(code="insufficient_quota", status_code=429)
        assert ai_provider._is_balance_error(exc)

    def test_payment_required_is_balance(self):
        exc = SimpleNamespace(status_code=402)
        assert ai_provider._is_balance_error(exc)

    def test_credit_message_is_balance(self):
        exc = Exception("Your credit balance is too low")
        assert ai_provider._is_balance_error(exc)


class TestSessionChat:
    def test_anthropic_session_chat_does_not_duplicate_assistant(self, monkeypatch):
        captured = {}

        class FakeContent:
            type = "text"
            text = "Javob"

        class FakeResp:
            content = [FakeContent()]

        class FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResp()

        class FakeClient:
            messages = FakeMessages()

        monkeypatch.setattr(
            ai_provider.anthropic,
            "Anthropic",
            lambda **kwargs: FakeClient(),
        )
        history = [
            {
                "role": "assistant",
                "content": "Sessiya yakunlandi! Savol bering.",
            }
        ]
        reply = ai_provider.session_chat(
            "Anthropic",
            "key",
            "claude-sonnet-4-6",
            "ctx",
            history,
            "Salom",
        )
        assert reply == "Javob"
        roles = [m["role"] for m in captured["messages"]]
        assert roles == ["assistant", "user"]
        assert "Session context" in captured["system"]
