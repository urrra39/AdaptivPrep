"""Tests for email/password account registration and authentication."""
import pytest

from src.data import schema


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "auth.db"
    schema.init_db(path)
    return path


class TestEmailAuth:
    def test_register_and_authenticate(self, db):
        uid = schema.register_user("alice@gmail.com", "secret123", "Alice", db_path=db)
        got_id, name = schema.authenticate_user("alice@gmail.com", "secret123", db_path=db)
        assert got_id == uid
        assert name == "Alice"

    def test_duplicate_email_rejected(self, db):
        schema.register_user("bob@gmail.com", "password99", "Bob", db_path=db)
        with pytest.raises(schema.EmailTakenError):
            schema.register_user("bob@gmail.com", "otherpass", "Bob2", db_path=db)

    def test_wrong_password_rejected(self, db):
        schema.register_user("carol@gmail.com", "rightpass1", "Carol", db_path=db)
        with pytest.raises(schema.AuthError):
            schema.authenticate_user("carol@gmail.com", "wrongpass", db_path=db)

    def test_password_reset_flow(self, db):
        schema.register_user("dave@gmail.com", "oldpass123", "Dave", db_path=db)
        token = schema.create_password_reset_token("dave@gmail.com", db_path=db)
        assert token
        assert schema.reset_password_with_token(token, "newpass456", db_path=db)
        schema.authenticate_user("dave@gmail.com", "newpass456", db_path=db)

    def test_password_reset_rejects_short_password(self, db):
        schema.register_user("dave@gmail.com", "oldpass123", "Dave", db_path=db)
        token = schema.create_password_reset_token("dave@gmail.com", db_path=db)
        assert not schema.reset_password_with_token(token, "short", db_path=db)

    def test_duplicate_username_rejected(self, db):
        schema.register_user("a@gmail.com", "password99", "Ali", db_path=db)
        with pytest.raises(schema.UsernameTakenError):
            schema.register_user("b@gmail.com", "password88", "Ali", db_path=db)

    def test_short_password_rejected(self, db):
        with pytest.raises(ValueError):
            schema.register_user("eve@gmail.com", "short", "Eve", db_path=db)
