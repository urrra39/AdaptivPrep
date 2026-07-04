"""Tests for SMTP password-reset email helpers."""
from unittest.mock import MagicMock, patch

from src.data import auth_email


class TestAuthEmail:
    def test_smtp_not_configured_without_password(self, monkeypatch):
        monkeypatch.delenv("SMTP_HOST", raising=False)
        monkeypatch.delenv("SMTP_USER", raising=False)
        assert not auth_email.smtp_configured()

    def test_smtp_configured_with_env(self, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
        monkeypatch.setenv("SMTP_USER", "a@gmail.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret")
        assert auth_email.smtp_configured()

    def test_send_without_config_returns_uz_message(self, monkeypatch):
        monkeypatch.delenv("SMTP_HOST", raising=False)
        monkeypatch.delenv("SMTP_USER", raising=False)
        result = auth_email.send_password_reset_email("u@gmail.com", "tok")
        assert not result.ok
        assert "SMTP" in result.message_uz

    def test_gmail_defaults(self):
        d = auth_email.gmail_defaults_for("user@gmail.com")
        assert d["host"] == "smtp.gmail.com"
        assert d["user"] == "user@gmail.com"

    @patch("src.data.auth_email._with_smtp")
    def test_send_success(self, mock_smtp, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
        monkeypatch.setenv("SMTP_USER", "a@gmail.com")
        monkeypatch.setenv("SMTP_PASSWORD", "apppass")
        mock_smtp.side_effect = lambda cfg, action: action(MagicMock())
        result = auth_email.send_password_reset_email("b@gmail.com", "abc123")
        assert result.ok
        assert "yuborildi" in result.message_uz

    def test_smtp_port_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
        monkeypatch.setenv("SMTP_USER", "a@gmail.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret")
        monkeypatch.setenv("SMTP_PORT", "not-a-port")
        cfg = auth_email._smtp_from_env()
        assert cfg["port"] == 587

    @patch("src.data.auth_email._with_smtp")
    def test_validate_smtp_auth_error(self, mock_smtp, monkeypatch):
        import smtplib

        monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
        monkeypatch.setenv("SMTP_USER", "a@gmail.com")
        monkeypatch.setenv("SMTP_PASSWORD", "bad")
        mock_smtp.side_effect = smtplib.SMTPAuthenticationError(535, b"auth")
        result = auth_email.validate_smtp()
        assert not result.ok
        assert "App Password" in result.message_uz
