"""Repo-root compatibility shim for managed-cloud (Render) deploys.

Render's stock Python web-service start command is ``uvicorn app.main:app``
running from the repository root, where the real package lives under
``backend/app``. This shim lets that stock command boot the FastAPI app
unchanged. The render.yaml blueprint (``uvicorn backend.app.main:app``) does not
use it.
"""

from backend.app.main import app  # noqa: F401