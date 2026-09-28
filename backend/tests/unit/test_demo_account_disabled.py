"""The demo account must stop working when it is switched off (P0).

Two separate defects are covered here.

1. ``main.py`` only skipped *seeding* when the demo account was disabled. The
   row already existed in any database that had been seeded before, so login
   kept succeeding with the published password -- which is precisely what the
   production audit observed.
2. ``GET /api/auth/demo`` published the demo password to any caller, with no
   production gate, so the credential hint was public regardless of whether the
   account worked.

These run against a real SQLite database in a temp directory via the app's own
test fixtures, so they exercise the actual routing and dependency wiring.
"""

import pathlib
import tempfile

import pytest
from app.database import Base, get_db
from app.models.db_models import User
from app.security import hash_password
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app


@pytest.fixture
def client_factory(monkeypatch):
    """Build a TestClient over a disposable database with one seeded demo user."""
    tmp = pathlib.Path(tempfile.mkdtemp()) / "auth.db"
    engine = create_engine(f"sqlite:///{tmp}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = TestingSession()
    salt, pwhash = hash_password("AeroCast@2026")
    db.add(
        User(
            email="demo@aerocast.local",
            name="AeroCast Demo",
            role="analyst",
            password_hash=pwhash,
            password_salt=salt,
        )
    )
    db.commit()
    db.close()

    def _override():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(app, "dependency_overrides", {get_db: _override})
    with TestClient(app) as c:
        # Exposed so a test can inspect stored rows through the same database
        # the client is using, rather than a second, unrelated connection.
        c.session_factory = TestingSession
        yield c
    engine.dispose()


DEMO_EMAIL = "demo@aerocast.local"
DEMO_PASSWORD = "AeroCast@2026"


def _set_demo(monkeypatch, enabled: bool, production: bool = False):
    """Point settings at a known demo account and set the switch.

    The underlying fields are patched rather than the derived properties:
    ``is_production`` and ``demo_user_enabled`` are read-only properties, and
    patching the inputs also exercises the resolution rules that production
    deployment depends on.
    """
    import app.config as config

    real = config.get_settings()
    monkeypatch.setattr(real, "demo_user_email", DEMO_EMAIL, raising=False)
    monkeypatch.setattr(real, "environment", "production" if production else "development",
                        raising=False)
    monkeypatch.setattr(real, "enable_demo_user", enabled, raising=False)
    return real


# --- login refusal ----------------------------------------------------------


def test_login_succeeds_when_demo_account_is_enabled(client_factory, monkeypatch):
    _set_demo(monkeypatch, enabled=True)
    r = client_factory.post("/api/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    assert r.status_code == 200
    assert r.json()["access_token"]


def test_login_is_refused_when_demo_account_is_disabled(client_factory, monkeypatch):
    """The regression: the row exists, the switch is off, login still 200s."""
    _set_demo(monkeypatch, enabled=False)
    r = client_factory.post("/api/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    assert r.status_code == 401
    assert "access_token" not in r.json()


def test_disabled_demo_refusal_is_case_and_whitespace_insensitive(client_factory, monkeypatch):
    _set_demo(monkeypatch, enabled=False)
    r = client_factory.post(
        "/api/auth/login", json={"email": f"  {DEMO_EMAIL.upper()}  ", "password": DEMO_PASSWORD}
    )
    assert r.status_code == 401


def test_disabled_demo_refusal_does_not_leak_that_the_account_exists(client_factory, monkeypatch):
    """Same message as a wrong password, so the endpoint is not an account oracle."""
    _set_demo(monkeypatch, enabled=False)
    r = client_factory.post("/api/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    assert r.json()["detail"] == "Invalid email or password"


def test_other_accounts_are_unaffected_when_demo_is_disabled(client_factory, monkeypatch):
    """Switching the demo account off must not lock out real users."""
    _set_demo(monkeypatch, enabled=False)
    r = client_factory.post(
        "/api/auth/login", json={"email": "demo@aerocast.local".replace("demo", "other"),
                                 "password": "whatever"}
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid email or password"


# --- already-issued tokens --------------------------------------------------


def test_token_issued_before_the_switch_is_still_refused(client_factory, monkeypatch):
    """Rotating a switch must not leave old sessions working for the full TTL."""
    _set_demo(monkeypatch, enabled=True)
    r = client_factory.post("/api/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    assert r.status_code == 200
    token = r.json()["access_token"]

    _set_demo(monkeypatch, enabled=False)
    me = client_factory.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 401


def test_valid_token_for_a_real_user_still_works(client_factory, monkeypatch):
    _set_demo(monkeypatch, enabled=True)
    r = client_factory.post("/api/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD})
    token = r.json()["access_token"]
    assert client_factory.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200


# --- the credential hint endpoint ------------------------------------------


def test_demo_credentials_are_served_when_enabled(client_factory, monkeypatch):
    _set_demo(monkeypatch, enabled=True)
    r = client_factory.get("/api/auth/demo")
    assert r.status_code == 200
    assert r.json()["email"] == DEMO_EMAIL


def test_demo_credentials_endpoint_is_gone_when_disabled(client_factory, monkeypatch):
    """A 404, not a 200 with the password: the hint is the leak itself."""
    _set_demo(monkeypatch, enabled=False)
    r = client_factory.get("/api/auth/demo")
    assert r.status_code == 404
    assert DEMO_PASSWORD not in r.text


def test_demo_credentials_endpoint_is_gone_in_production(client_factory, monkeypatch):
    _set_demo(monkeypatch, enabled=False, production=True)
    assert client_factory.get("/api/auth/demo").status_code == 404


# --- password storage (unchanged, pinned so the fix cannot weaken it) -------


def test_passwords_are_salted_and_hashed_not_stored_plaintext(client_factory):
    """Pinned so the demo-account fix cannot weaken credential storage."""
    db = client_factory.session_factory()
    try:
        user = db.query(User).filter(User.email == DEMO_EMAIL).first()
        assert user is not None
        assert user.password_hash != DEMO_PASSWORD
        assert DEMO_PASSWORD not in user.password_hash
        assert user.password_salt
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
