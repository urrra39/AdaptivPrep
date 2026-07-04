"""Tests for the SQLite layer and PIN-based collision prevention.

The headline test is ``test_wrong_pin_blocks_access_to_another_user`` - it
asserts the exact bug being fixed: a second visitor typing an existing
username cannot reach the first user's account or data.
"""
import sqlite3

import pytest

from src.data import schema


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "t.db"
    schema.init_db(path)
    return path


class TestPinHashing:
    def test_hash_verify_round_trip(self):
        h = schema.hash_pin("1234")
        assert schema._check_pin("1234", h)
        assert not schema._check_pin("0000", h)

    def test_hash_is_salted_and_not_plaintext(self):
        h = schema.hash_pin("1234")
        assert "1234" not in h
        assert schema.hash_pin("1234") != h  # fresh random salt each call


class TestGetOrCreateUser:
    def test_new_user_created_with_pin(self, db):
        uid = schema.get_or_create_user("alice", pin="1234", db_path=db)
        assert isinstance(uid, int)
        assert schema.verify_pin(uid, "1234", db_path=db)

    def test_legacy_no_pin_path_is_idempotent(self, db):
        # pin=None keeps the pre-feature contract (analytics/CLI callers).
        uid1 = schema.get_or_create_user("bob", db_path=db)
        uid2 = schema.get_or_create_user("bob", db_path=db)
        assert uid1 == uid2

    def test_correct_pin_returns_same_user(self, db):
        uid = schema.get_or_create_user("alice", pin="1234", db_path=db)
        assert schema.get_or_create_user("alice", pin="1234", db_path=db) == uid

    def test_wrong_pin_blocks_access_to_another_user(self, db):
        owner = schema.get_or_create_user("alice", pin="1234", db_path=db)
        schema.record_response(owner, "q1", "vocab_health", True, db_path=db)
        with pytest.raises(schema.BadPinError):
            schema.get_or_create_user("alice", pin="9999", db_path=db)
        # The impostor never obtained the owner's id, and the data is intact.
        assert not schema.verify_pin(owner, "9999", db_path=db)
        assert len(schema.get_responses(owner, db_path=db)) == 1

    def test_legacy_null_pin_user_can_set_one(self, db):
        uid = schema.get_or_create_user("carol", db_path=db)  # created pin-less
        assert schema.get_or_create_user("carol", pin="4321", db_path=db) == uid
        assert schema.verify_pin(uid, "4321", db_path=db)
        with pytest.raises(schema.BadPinError):  # now protected
            schema.get_or_create_user("carol", pin="0000", db_path=db)

    @pytest.mark.parametrize("bad", ["12", "12345", "abcd", "12a4", ""])
    def test_pin_must_be_four_digits(self, db, bad):
        with pytest.raises(ValueError):
            schema.get_or_create_user("dave", pin=bad, db_path=db)


class TestVerifyPin:
    def test_no_pin_set_returns_false(self, db):
        uid = schema.get_or_create_user("erin", db_path=db)
        assert schema.verify_pin(uid, "1234", db_path=db) is False


class TestMigration:
    def test_pin_hash_added_to_pre_feature_db(self, tmp_path):
        # A DB created before pin_hash existed: users has no such column.
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "username TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO users (username, created_at) VALUES ('old', 'x')")
        conn.commit()
        conn.close()

        schema.init_db(path)  # migration must add the column, not lock 'old' out

        conn = schema.get_connection(path)
        try:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
        finally:
            conn.close()
        assert "pin_hash" in cols
        row = schema.get_user("old", db_path=path)
        assert row is not None and row["pin_hash"] is None  # reachable, sets PIN later
