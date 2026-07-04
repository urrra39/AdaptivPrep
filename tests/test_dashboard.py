"""Dashboard access-gate test: the PIN that protects a user's quiz login must
also block viewing that user's analytics. This closes the bug where the
dashboard rendered any selected user's mastery data with no PIN check."""
import pytest

from src.app import dashboard
from src.data import schema


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "d.db"
    schema.init_db(path)
    return path


def test_wrong_or_unknown_pin_blocks_viewing_another_user(db):
    owner = schema.get_or_create_user("alice", pin="1234", db_path=db)
    schema.record_response(owner, "q1", "vocab_health", True, db_path=db)

    assert dashboard._authorized_to_view("alice", "1234", db_path=db) == (True, "ok")
    # The bug: an impostor with the right username but wrong PIN must be blocked.
    assert dashboard._authorized_to_view("alice", "9999", db_path=db) == (False, "bad_pin")
    assert dashboard._authorized_to_view("ghost", "1234", db_path=db) == (False, "no_user")


def test_pinless_account_cannot_be_viewed(db):
    schema.get_or_create_user("legacy", db_path=db)  # created without a PIN
    assert dashboard._authorized_to_view("legacy", "0000", db_path=db) == (False, "no_pin")
