"""add pollution_observations.nh3 and .pb (CPCB criteria pollutants, unscored)

The official data.gov.in CPCB feed publishes ``NH3`` and ``Pb`` alongside the six
pollutants this project stored, but ``cpcb_service.POLLUTANT_MAP`` dropped them
and ``normalize_records`` discarded the rows. This migration adds the two
nullable columns so those real measurements are retained instead of lost.

Both columns are ``ug/m3`` and nullable. They are deliberately **not** scored
into the AQI: no verified CPCB sub-index breakpoint table for NH3 or Pb exists in
this repository, and inventing one would emit a confidently wrong regulatory
number. ``aqi_calculator.evaluate_aqi`` reports them through
``data_availability`` as ``official_cpcb_table_not_vendored_in_repository``.

No backfill: historical rows keep NULL, which is the correct representation of
"not measured". Nothing is ever defaulted to zero.

Purely additive and nullable, so it is safe on the 133k-row production table.

Revision ID: a7c3e91b5d24
Revises: 5c1b7d9a2f6e
Create Date: 2026-09-27 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = 'a7c3e91b5d24'
down_revision: Union[str, None] = '5c1b7d9a2f6e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('pollution_observations', sa.Column('nh3', sa.Float(), nullable=True))
    op.add_column('pollution_observations', sa.Column('pb', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('pollution_observations', 'pb')
    op.drop_column('pollution_observations', 'nh3')
