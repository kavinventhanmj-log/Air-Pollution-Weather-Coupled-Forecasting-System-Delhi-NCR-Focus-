"""Fire observation provenance (synthetic / source)

Adds ``synthetic`` + ``source`` to ``fire_readings`` so a simulated
stubble-fire history row can always be told apart from a real NASA FIRMS
detection, and no downstream consumer presents a simulated fire as live
FIRMS data.

* ``synthetic`` — boolean, NOT NULL, default false ("not synthetic").
  Existing rows are left false: a heuristic cannot distinguish a legacy
  real detection from a legacy simulated one, and guessing would mislabel
  real measurements (same policy as the pollution ``re_stamped`` column in
  ``b8d3f1a9c4e2``). Provenance for the pre-existing simulated CSV rows is
  restored deliberately and exactly via
  ``backend/scripts/backfill_fire_provenance.py``, never by this migration.
* ``source`` — nullable string naming the writer:
  ``"synthetic_sim"`` / ``"firms_csv"`` / ``"firms_live"``.

SQLite dev DBs receive the same additive columns via ``apply_migrations()``
in ``backend/app/database.py``.

Revision ID: c5d7e9f1a3b0
Revises: b8d3f1a9c4e2
Create Date: 2026-10-01 09:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'c5d7e9f1a3b0'
down_revision: str | None = 'b8d3f1a9c4e2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'fire_readings',
        sa.Column('synthetic', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        'fire_readings',
        sa.Column('source', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('fire_readings', 'source')
    op.drop_column('fire_readings', 'synthetic')
