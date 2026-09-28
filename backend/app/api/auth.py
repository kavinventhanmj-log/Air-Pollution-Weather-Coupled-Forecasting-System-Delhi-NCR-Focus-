"""Authentication endpoints for the AeroCast-NCR portal (SIH26082).

Additive layer over the existing public data API. Only these endpoints are
auth-aware: ``POST /api/auth/login``, ``GET /api/auth/me`` and
``POST /api/auth/logout``. All forecasting / data endpoints remain public so
scripts, notebooks and the existing test suite keep working unchanged.
"""
import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models.db_models import User
from ..security import create_access_token, decode_access_token, hash_password, verify_password

logger = logging.getLogger("aerocast.auth")

router = APIRouter()


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
    expires_in: int


class DemoCredentials(BaseModel):
    email: str
    password: str
    name: str
    role: str


# ---------------------------------------------------------------------------
# Demo account seeding (idempotent)
# ---------------------------------------------------------------------------
def ensure_demo_user(db: Session) -> None:
    """Create/refresh the env-configured demo account at startup."""
    settings = get_settings()
    user = db.query(User).filter(User.email == settings.demo_user_email).first()
    salt, pwhash = hash_password(settings.demo_user_password)
    if user is None:
        db.add(User(
            email=settings.demo_user_email,
            name=settings.demo_user_name,
            role=settings.demo_user_role,
            password_hash=pwhash,
            password_salt=salt,
        ))
        logger.info("Seeded demo user %s", settings.demo_user_email)
    else:
        # Keep the stored demo credentials in sync with DEMO_USER_* overrides.
        user.name = settings.demo_user_name
        user.role = settings.demo_user_role
        user.password_hash = pwhash
        user.password_salt = salt
    db.commit()


def _to_user_response(u: User) -> UserResponse:
    return UserResponse(id=u.id, email=u.email, name=u.name, role=u.role)


def _is_disabled_demo_account(email: str) -> bool:
    """Whether ``email`` is the demo account while that account is switched off.

    Disabling the demo account is not enough on its own. The row already exists
    in any database that was ever seeded, so refusing to *create* it leaves a
    working login behind -- which is exactly the state the production audit
    found: the published demo password still authenticated. The refusal has to
    live on the login path as well as the seeding path.
    """
    settings = get_settings()
    if settings.demo_user_enabled:
        return False
    return (email or "").strip().lower() == settings.demo_user_email.strip().lower()


def get_current_user(
    authorization: str | None = Header(None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> UserResponse:
    """FastAPI dependency: resolve and validate the bearer token's user.

    The profile is taken from the signed token claims (name/role/email) so no
    database round-trip is needed — the hosted Postgres can take seconds to
    resume, and this dependency gates every protected page. Tokens issued
    before the claims were embedded fall back to a database lookup.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_access_token(token, get_settings().secret_key)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # A token minted before the demo account was disabled stays cryptographically
    # valid until it expires, so the switch-off is enforced here too. Otherwise
    # revoking the account would leave previously issued sessions working for
    # the full token lifetime.
    if _is_disabled_demo_account(payload.get("email", "")):
        logger.warning("Rejected a token belonging to the disabled demo account")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    name = payload.get("name")
    role = payload.get("role")
    email = payload.get("email")
    if name and role and email:
        return UserResponse(id=int(payload["sub"]), email=email, name=name, role=role)

    # Backward compatibility for tokens minted before claims were embedded.
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return _to_user_response(user)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate credentials and return a signed HS256 JWT."""
    settings = get_settings()

    # Checked before the database so a disabled demo account is refused
    # identically whether or not the row was ever seeded, and the same
    # "Invalid email or password" is returned to avoid confirming which
    # addresses exist.
    if _is_disabled_demo_account(body.email):
        logger.warning("Rejected login for the disabled demo account")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = db.query(User).filter(User.email == body.email.lower()).first()
    if user is None or not verify_password(
        body.password, user.password_salt, user.password_hash
    ):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(
        user.id, user.email, settings.secret_key, name=user.name, role=user.role
    )
    return LoginResponse(
        access_token=token,
        user=_to_user_response(user),
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.get("/auth/me", response_model=UserResponse)
def me(user: UserResponse = Depends(get_current_user)):
    """Return the profile of the currently authenticated user."""
    return user


@router.post("/auth/logout", response_model=UserResponse)
def logout(user: UserResponse = Depends(get_current_user)):
    """Idempotent logout — tokens are stateless, so this is a client action.

    The endpoint exists so the portal can call it for future server-side
    token-blacklisting without changing the client contract.
    """
    return user


@router.get("/auth/demo", response_model=DemoCredentials)
def demo_credentials():
    """Expose the env-configured demo login so the UI can show a hint.

    Returns 404 whenever the demo account is switched off. The published
    credential hint is what put the demo password in front of every visitor in
    the first place, so it must not be served from a production deployment even
    if the account happens to still exist in the database.
    """
    settings = get_settings()
    if not settings.demo_user_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    return DemoCredentials(
        email=settings.demo_user_email,
        password=settings.demo_user_password,
        name=settings.demo_user_name,
        role=settings.demo_user_role,
    )
