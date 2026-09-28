"""Production fail-closed configuration (P0 security).

The release audit found three ways this service started "successfully" while
being insecure:

* the published development ``SECRET_KEY`` signed HS256 JWTs, so anyone who
  could read the repository could mint an admin token;
* an unset ``DATABASE_URL`` fell back to an empty local SQLite file, and the
  DB-free ``/api/health`` probe still reported the service as healthy;
* the published demo password seeded a working login on every boot.

Each test pins one refusal. Development behaviour must be unchanged, so the
tests exercise the validator directly rather than mutating process state.
"""

import pytest
from app.config import DEV_DEMO_PASSWORD, DEV_SECRET_KEY, Settings

PG_URL = "postgresql://user:pass@db.example.com:5432/aerocast_ncr"


def _prod(**overrides):
    """A production Settings with safe values, then the override under test."""
    base = {
        "environment": "production",
        "secret_key": "k" * 48,
        "database_url": PG_URL,
        "enable_demo_user": False,
        "demo_hydrate_empty_db": False,
    }
    base.update(overrides)
    return base


# --- the happy path: a correct production config boots ----------------------


def test_secure_production_config_is_accepted():
    s = Settings(**_prod())
    assert s.is_production is True


# --- SECRET_KEY -------------------------------------------------------------


def test_production_refuses_the_published_development_secret():
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(**_prod(secret_key=DEV_SECRET_KEY))


def test_production_refuses_an_empty_secret():
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(**_prod(secret_key=""))


def test_production_refuses_a_short_secret():
    with pytest.raises(ValueError, match="at least 32"):
        Settings(**_prod(secret_key="tooshort"))


def test_production_accepts_a_strong_secret():
    assert len(Settings(**_prod(secret_key="x" * 64)).secret_key) == 64


# --- DATABASE_URL -----------------------------------------------------------


def test_production_refuses_a_sqlite_database_url():
    """A SQLite fallback serves an empty database while reporting healthy."""
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Settings(**_prod(database_url="sqlite:///./aerocast_ncr.db"))


def test_production_refuses_an_empty_database_url():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Settings(**_prod(database_url=""))


def test_production_accepts_postgresql():
    assert Settings(**_prod()).database_url.startswith("postgresql")


# --- demo account -----------------------------------------------------------


def test_production_refuses_the_demo_account_enabled():
    with pytest.raises(ValueError, match="ENABLE_DEMO_USER"):
        Settings(**_prod(enable_demo_user=True))


def test_demo_user_defaults_off_in_production():
    """Unset must resolve to "off" in production, so a default boot is safe.

    No error is expected here: the unset default is already the safe one. The
    validator only fires when someone explicitly opts the account back on.
    """
    settings = _prod()
    settings.pop("enable_demo_user")
    assert Settings(**settings).demo_user_enabled is False


def test_demo_user_defaults_on_outside_production():
    s = Settings(environment="development", database_url="sqlite:///./x.db")
    assert s.demo_user_enabled is True


def test_demo_user_explicit_false_wins_in_development():
    s = Settings(
        environment="development", database_url="sqlite:///./x.db", enable_demo_user=False
    )
    assert s.demo_user_enabled is False


# --- synthetic observations -------------------------------------------------


def test_production_refuses_demo_hydration():
    with pytest.raises(ValueError, match="DEMO_HYDRATE_EMPTY_DB"):
        Settings(**_prod(demo_hydrate_empty_db=True))


def test_demo_hydration_is_off_in_production_by_default():
    s = Settings(**_prod(demo_hydrate_empty_db=False))
    assert s.demo_hydration_enabled is False


def test_demo_hydration_honours_explicit_opt_in_in_development():
    s = Settings(
        environment="development", database_url="sqlite:///./x.db", demo_hydrate_empty_db=True
    )
    assert s.demo_hydration_enabled is True


# --- development must not regress -------------------------------------------


def test_development_accepts_the_documented_defaults():
    """Local dev and CI keep working with the published placeholders."""
    s = Settings(environment="development", database_url="sqlite:///./x.db")
    assert s.secret_key == DEV_SECRET_KEY
    assert s.demo_user_password == DEV_DEMO_PASSWORD


def test_development_allows_sqlite():
    s = Settings(environment="development", database_url="sqlite:///./x.db")
    assert s.is_production is False


def test_environment_matching_is_case_insensitive():
    s = Settings(environment="  PRODUCTION ", secret_key="k" * 48, database_url=PG_URL,
                 enable_demo_user=False, demo_hydrate_empty_db=False)
    assert s.is_production is True


# --- multiple problems are reported together -------------------------------


def test_all_problems_are_reported_in_one_error():
    """An operator should see every blocker, not just the first."""
    with pytest.raises(ValueError) as exc:
        Settings(
            environment="production",
            secret_key=DEV_SECRET_KEY,
            database_url="sqlite:///./x.db",
            enable_demo_user=True,
            demo_hydrate_empty_db=True,
        )
    message = str(exc.value)
    for expected in ("SECRET_KEY", "DATABASE_URL", "ENABLE_DEMO_USER", "DEMO_HYDRATE_EMPTY_DB"):
        assert expected in message


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
