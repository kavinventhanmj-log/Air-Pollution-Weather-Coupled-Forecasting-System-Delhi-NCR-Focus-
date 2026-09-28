"""stations.state + unique (station_id, timestamp) on pollution_readings

Phase 1 (official data.gov.in CPCB pipeline) adds:
  * stations.state                -- required station field
  * pollution_readings unique constraint on (station_id, timestamp)
    so the official ingestion endpoint cannot create duplicate rows.

Revision ID: b9c9f4d1a7e2
Revises: 0964e5b227e3
Create Date: 2026-09-09 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b9c9f4d1a7e2'
down_revision: Union[str, None] = '0964e5b227e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('stations', sa.Column('state', sa.String(), nullable=True))
    # Pre-existing rows (from the earlier opencity.in pipeline) may contain
    # duplicate (station_id, timestamp) pairs; deduplicate before adding the
    # unique constraint so Postgres accepts it (keep the row with max id).
    # Correlated form rather than PostgreSQL's ``DELETE ... USING`` so this
    # migration also runs on the SQLite development database.
    op.execute(
        """
        DELETE FROM pollution_readings
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM pollution_readings
            GROUP BY station_id, timestamp
        )
        """
    )
    # Create the unique constraint as a UNIQUE INDEX rather than a table
    # constraint. On PostgreSQL a table constraint added via ALTER TABLE leaves
    # the id sequence alone, but the earlier version of this migration used
    # Alembic batch mode, which rebuilds the table and REPLACES the owned
    # sequence with ``_alembic_tmp_<table>_id_seq``. That orphaned name then
    # broke the very next revision (d4a1e4c9f0b2) with
    # ``relation "pollution_readings_id_seq" does not exist``. A unique index
    # enforces the same thing without touching the sequence.
    op.create_index('uq_pollution_station_ts', 'pollution_readings',
                    ['station_id', 'timestamp'], unique=True)


def downgrade() -> None:
    op.drop_index('uq_pollution_station_ts', table_name='pollution_readings')
    op.drop_column('stations', 'state')