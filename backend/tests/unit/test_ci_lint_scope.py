"""Guard that CI lints the operational-safety code, not just services and tests.

The hardening in commit 587fcdd lives largely *outside* ``backend/app``:
``alembic/env.py`` delegates database-target resolution to
``backend/scripts/alembic_db_url.py``, and the provenance reconciliation is
``backend/scripts/backfill_fire_provenance.py``. Both were originally absent
from the CI lint scope, so a regression that re-introduced silent fall-through
(or removed a production guard) would pass every pipeline.

These tests read ``.github/workflows/ci.yml`` and assert the coverage, so a
future narrowing of the scope fails the build instead of quietly dropping the
safety files. They assert *coverage*, not lint cleanliness -- the latter is the
ruff step's job.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: Files that implement the fail-closed database and provenance guarantees. Each
#: must appear in the CI ruff invocation.
SAFETY_FILES = (
    "backend/conftest.py",
    "backend/scripts/alembic_db_url.py",
    "backend/scripts/backfill_fire_provenance.py",
    "alembic/env.py",
)


@pytest.fixture(scope="module")
def ruff_scope() -> str:
    """The path arguments of the CI ``ruff check`` step.

    The step uses a folded ``>-`` scalar, so the command spans several lines.
    Continuation lines are joined before matching, otherwise a target split
    across a line break would be invisible to the assertions below.
    """
    if not CI_WORKFLOW.exists():
        pytest.skip(f"CI workflow not present: {CI_WORKFLOW}")
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"python -m ruff check(?P<paths>[^\n]*(?:\n[ \t]+[^\n]*)*)", text)
    assert match, "no `python -m ruff check` step found in ci.yml"
    paths: list[str] = []
    for line in match.group("paths").splitlines():
        stripped = line.strip()
        # Stop at the end of the folded `run:` block rather than swallowing the
        # following step's commands.
        if stripped.startswith("- ") or stripped.startswith("run:"):
            break
        paths.extend(stripped.split())
    assert paths, "ruff step resolved to an empty scope"
    return " ".join(paths)


def _covered(target: str, scope: str) -> bool:
    """True when ``target`` is linted, directly or via an enclosing directory.

    CI passes directories (``backend/scripts``) rather than enumerating every
    module, so a plain substring test would report a real, covered file as
    unprotected. A scope entry covers a target when it is the target itself or
    one of its parent directories.
    """
    target_path = PurePosixPath(target)
    for entry in scope.split():
        entry_path = PurePosixPath(entry)
        if target_path == entry_path:
            return True
        if entry_path in target_path.parents:
            return True
    return False


@pytest.mark.parametrize("target", SAFETY_FILES)
def test_safety_file_is_linted_in_ci(ruff_scope: str, target: str) -> None:
    """Each safety file must be linted in CI, directly or via its directory."""
    assert _covered(target, ruff_scope), (
        f"{target} implements an operational-safety guarantee but is not linted in "
        f"CI. Scope found: `ruff check {ruff_scope}`"
    )


def test_ci_lint_scope_includes_core_backend(ruff_scope: str) -> None:
    """The pre-existing scope must not be narrowed while safety files are added."""
    for target in ("backend/app", "backend/tests"):
        assert _covered(target, ruff_scope), (
            f"{target} was dropped from the CI lint scope"
        )


def test_ci_lint_scope_covers_every_backend_script(ruff_scope: str) -> None:
    """Guard against a *new* safety-relevant script being added unlinted.

    Every top-level ``backend/scripts/*.py`` is required in scope. This is
    deliberately stricter than enumerating today's files so that the next
    operator-facing script cannot be added without CI protection.
    """
    scripts = sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in (REPO_ROOT / "backend" / "scripts").glob("*.py")
    )
    assert scripts, "no scripts found under backend/scripts"
    missing = [s for s in scripts if not _covered(s, ruff_scope)]
    assert not missing, (
        f"backend/scripts module(s) {missing} are not linted in CI; add them to the "
        "ruff step in ci.yml"
    )
