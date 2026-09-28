"""Fire observation source columns

Adds instrument + brightness to ``fire_readings`` and enforces the
hotspot uniqueness key (satellite, latitude, longitude, acq_date) so the
same FIRMS detection is never stored twice. SQLite dev DBs receive the
same additive columns via ``apply_migrations()`` in ``backend/app/database.py``.

Revision ID: e2b1c3d4a5f7
Revises: c3a1e4f0b2d6
Create Date: 2026-09-12 14:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e2b1c3d4a5f7'
down_revision: Union[str, None] = 'c3a1e4f0b2d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('fire_readings', sa.Column('instrument', sa.String(), nullable=True))
    op.add_column('fire_readings', sa.Column('brightness', sa.Float(), nullable=True))
    op.create_index('idx_fire_lat_lon_time', 'fire_readings', ['latitude', 'longitude', 'acq_date'], unique=False)
    # SQLite needs a table rebuild for a new constraint; PostgreSQL must use
    # plain DDL because batch mode drops the table's id sequence with the table.
    if op.get_bind().dialect.name == 'sqlite':
        with op.batch_alter_table(
            'fire_readings', naming_convention={'uq': 'uq_fire_lat_lon_time'}
        ) as batch:
            batch.create_unique_constraint(
                'uq_fire_lat_lon_time', ['satellite', 'latitude', 'longitude', 'acq_date']
            )
    else:
        op.create_unique_constraint(
            'uq_fire_lat_lon_time', 'fire_readings',
            ['satellite', 'latitude', 'longitude', 'acq_date'],
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == 'sqlite':
        with op.batch_alter_table(
            'fire_readings', naming_convention={'uq': 'uq_fire_lat_lon_time'}
        ) as batch:
            batch.drop_constraint('uq_fire_lat_lon_time', type_='unique')
    else:
        op.drop_constraint('uq_fire_lat_lon_time', 'fire_readings', type_='unique')
    op.drop_index('idx_fire_lat_lon_time', table_name='fire_readings')
    op.drop_column('fire_readings', 'brightness')
    op.drop_column('fire_readings', 'instrument')