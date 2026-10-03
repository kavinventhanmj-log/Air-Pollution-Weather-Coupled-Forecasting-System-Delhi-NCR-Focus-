"""Guard against pytest ever dropping tables on a non-disposable database.

``db_session`` calls ``Base.metadata.drop_all``. If ``DATABASE_URL`` is exported
to a production database, an ordinary ``pytest`` run would destroy it. These
tests pin the refusal and the explicit opt-in, and touch no database at all.
"""

from __future__ import annotations

import pytest


class TestDisposableTestDatabase:
    def test_sqlite_url_is_allowed(self, db_guard):
        db_guard.ensure_disposable_test_database(
            {"DATABASE_URL": "sqlite:///tmp/test.db"}
        )

    def test_missing_url_is_allowed(self, db_guard):
        db_guard.ensure_disposable_test_database({})

    def test_non_sqlite_url_without_optin_is_refused(self, db_guard):
        with pytest.raises(db_guard.live_test_database_refused) as exc:
            db_guard.ensure_disposable_test_database(
                {"DATABASE_URL": "postgresql://user:hunter2@db.example.invalid/app"}
            )
        # The refusal must never echo the URL or its password.
        assert "hunter2" not in str(exc.value)
        assert "db.example.invalid" not in str(exc.value)

    def test_non_sqlite_url_with_optin_is_allowed(self, db_guard):
        db_guard.ensure_disposable_test_database(
            {
                "DATABASE_URL": "postgresql://user:hunter2@db.example.invalid/app",
                db_guard.allow_env: "1",
            }
        )

    def test_optin_is_truthy_only_for_affirmatives(self, db_guard):
        for value in ("", "0", "no", "off"):
            with pytest.raises(db_guard.live_test_database_refused):
                db_guard.ensure_disposable_test_database(
                    {
                        "DATABASE_URL": "postgresql://user:hunter2@db.example.invalid/app",
                        db_guard.allow_env: value,
                    }
                )


class TestDisposableBackend:
    def test_sqlite_backend_allowed(self, db_guard):
        db_guard.ensure_disposable_backend("sqlite", {})

    def test_non_sqlite_backend_refused_without_optin(self, db_guard):
        with pytest.raises(db_guard.live_test_database_refused):
            db_guard.ensure_disposable_backend("postgresql", {})

    def test_non_sqlite_backend_allowed_with_optin(self, db_guard):
        db_guard.ensure_disposable_backend("postgresql", {db_guard.allow_env: "true"})
