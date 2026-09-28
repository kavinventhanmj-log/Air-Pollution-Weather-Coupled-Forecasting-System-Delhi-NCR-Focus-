"""station FKs, forecast uniqueness/audit, pollution re-stamp provenance

Closes the referential-integrity and provenance gaps found in the P0 audit.

1. ``pollution_observations.re_stamped`` (boolean, NOT NULL, default false)
   Marks a row that is a forward re-stamp of an older observation rather than a
   fresh measurement (``backend/scripts/bootstrap_recent.py``). Without this
   column, forecast provenance reported synthetic "recent" history as live
   sensor data. Existing rows are left false: no heuristic can tell a legacy
   re-stamp from a genuine repeat, and guessing would mislabel real
   measurements. See the note in ``upgrade()`` and
   ``backend/scripts/audit_re_stamps.py``.

2. Foreign keys on ``weather_observations.station_id``, ``forecasts.station_id``
   and ``alerts.station_id``. ``pollution_observations`` already had one; these
   three did not, so a forecast could reference a station that no longer exists
   and nothing would complain. Rows whose station is missing are deleted first,
   because the NOT NULL columns cannot be nulled out and a forecast for a
   deleted station is not recoverable data. The deleted count is logged.

3. ``forecasts`` unique on ``(station_id, horizon_hours)`` plus an index on
   ``horizon_hours``. Previously every regeneration appended a new row for the
   same horizon, so 24h of forecast could look like N days of history. Existing
   duplicates are collapsed to the most recent row per horizon (max id) before
   the constraint is added.

4. New ``forecast_runs`` audit table recording every generation attempt
   (succeeded / fallback / refused) with the provenance that was shown to the
   caller, so a published number can always be traced to the data behind it.

5. Index on ``forecast_runs(station_id, created_at)``.

Revision ID: b8d3f1a9c4e2
Revises: a7c3e91b5d24
Create Date: 2026-09-28 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = 'b8d3f1a9c4e2'
down_revision: Union[str, None] = 'a7c3e91b5d24'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: Tables gaining a station_id foreign key, and the column to backfill-null or
#: delete orphans on. All three columns are NOT NULL, so orphans are deleted.
FK_TABLES = ('weather_observations', 'forecasts', 'alerts')

#: Named so the constraint can be dropped again in downgrade().
FK_CONSTRAINTS = {
    'weather_observations': 'fk_weather_station_id',
    'forecasts': 'fk_forecast_station_id',
    'alerts': 'fk_alert_station_id',
}


def _delete_orphans(table: str) -> None:
    """Remove rows whose station_id no longer resolves to a station.

    Required before adding the FK: the column is NOT NULL, so orphans cannot be
    preserved by nulling them, and a weather/forecast/alert row attached to a
    non-existent station is not usable data.
    """
    result = op.get_bind().execute(
        sa.text(
            f"DELETE FROM {table} "  # noqa: S608 - table name from a fixed tuple
            "WHERE station_id IS NOT NULL "
            "AND station_id NOT IN (SELECT id FROM stations)"
        )
    )
    deleted = result.rowcount or 0
    if deleted:
        print(f"[{table}] deleted {deleted} orphan row(s) with no matching station")


def upgrade() -> None:
    # -- 1. pollution_observations.re_stamped --------------------------------
    op.add_column(
        'pollution_observations',
        sa.Column('re_stamped', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # Existing rows are left ``re_stamped = false``.
    #
    # An earlier draft of this migration tried to backfill the marker by finding
    # rows whose pm25 repeated an earlier reading for the same station within 7
    # days. That heuristic was measured against the real database and is unsafe
    # in both directions:
    #
    #   * On pm25 alone it flags 116,029 of 133,407 rows (87%), because genuine
    #     CPCB history is dense and pm25 repeats constantly at hourly cadence.
    #   * Requiring a full (pm25, pm10, o3, no2, so2, co, aqi) tuple match does
    #     not rescue it either: the largest repeated groups are all-NULL rows
    #     with aqi = 0 spanning 2023-2025, which are missing measurements, not
    #     bootstrap re-stamps.
    #
    # There is no way to distinguish a legacy re-stamp from a genuine repeat
    # after the fact. Guessing would label real measurements as synthetic, which
    # is the same fabrication this column exists to prevent. Instead:
    #
    #   * Existing rows default to false, i.e. "not known to be a re-stamp".
    #   * ``backend/scripts/bootstrap_recent.py`` stamps every row it writes from
    #     now on, so provenance is correct for all future data.
    #   * Forecasts that consumed legacy un-marked rows are still covered by the
    #     staleness window (``STALE_AFTER_HOURS``), which refuses rather than
    #     presenting old data as recent.
    #   * ``backend/scripts/audit_re_stamps.py`` lists candidate legacy rows for
    #     manual review instead of silently rewriting them.

    # -- 2. station_id foreign keys -----------------------------------------
    for table in FK_TABLES:
        _delete_orphans(table)

    # SQLite cannot ALTER TABLE to add a constraint; it needs a table rebuild.
    # PostgreSQL must NOT use batch mode: it drops and recreates the table, which
    # also drops the id sequence the table owns, breaking all later inserts with
    # "relation <table>_id_seq does not exist".
    is_sqlite = op.get_bind().dialect.name == 'sqlite'
    for table, name in FK_CONSTRAINTS.items():
        if is_sqlite:
            with op.batch_alter_table(table, naming_convention={'fk': name}) as batch:
                batch.create_foreign_key(
                    name, 'stations', ['station_id'], ['id'], ondelete='CASCADE'
                )
        else:
            op.create_foreign_key(
                name, table, 'stations', ['station_id'], ['id'], ondelete='CASCADE'
            )

    # -- 3. forecasts uniqueness --------------------------------------------
    # Collapse existing duplicate horizons to the newest row per horizon.
    if is_sqlite:
        op.execute(
            """
            DELETE FROM forecasts
            WHERE id NOT IN (
                SELECT MAX(id) FROM forecasts GROUP BY station_id, horizon_hours
            )
            """
        )
    else:
        op.execute(
            """
            DELETE FROM forecasts AS stale
            WHERE stale.id < (
                SELECT max(fresh.id)
                FROM forecasts AS fresh
                WHERE fresh.station_id = stale.station_id
                  AND fresh.horizon_hours = stale.horizon_hours
            )
            """
        )
    if is_sqlite:
        with op.batch_alter_table('forecasts', naming_convention={'uq': 'uq_forecast_station_horizon'}) as batch:
            batch.create_unique_constraint(
                'uq_forecast_station_horizon', ['station_id', 'horizon_hours']
            )
    else:
        op.create_unique_constraint(
            'uq_forecast_station_horizon', 'forecasts', ['station_id', 'horizon_hours']
        )
    op.create_index('idx_forecast_horizon', 'forecasts', ['horizon_hours'], unique=False)

    # -- 4. forecast_runs audit table ---------------------------------------
    op.create_table(
        'forecast_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('station_id', sa.Integer(), nullable=False),
        sa.Column('station_name', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('refusal_code', sa.String(), nullable=True),
        sa.Column('refusal_reason', sa.String(), nullable=True),
        sa.Column('model', sa.String(), nullable=True),
        sa.Column('model_artifact', sa.String(), nullable=True),
        sa.Column('model_artifact_sha256', sa.String(), nullable=True),
        sa.Column('fallback_used', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('fallback_reason', sa.String(), nullable=True),
        sa.Column('horizons', sa.String(), nullable=True),
        sa.Column('history_rows', sa.Integer(), server_default='0', nullable=False),
        sa.Column('pollution_rows', sa.Integer(), server_default='0', nullable=False),
        sa.Column('weather_rows', sa.Integer(), server_default='0', nullable=False),
        sa.Column('fire_rows', sa.Integer(), server_default='0', nullable=False),
        sa.Column('window_start', sa.DateTime(), nullable=True),
        sa.Column('window_end', sa.DateTime(), nullable=True),
        sa.Column('latest_observation', sa.DateTime(), nullable=True),
        sa.Column('observation_age_hours', sa.Float(), nullable=True),
        sa.Column('is_stale', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('is_demo', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('is_re_stamped', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('data_source', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['station_id'], ['stations.id'], name='fk_forecast_run_station_id', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    # Matches the ORM's ``index=True`` on ForecastRun.id so autogenerate does
    # not report a spurious index rename. The primary key already provides
    # uniqueness, so this is a plain (non-unique) index for lookup.
    op.create_index(op.f('ix_forecast_runs_id'), 'forecast_runs', ['id'], unique=False)
    op.create_index(
        'idx_forecast_run_station_time', 'forecast_runs', ['station_id', 'created_at'], unique=False
    )


def downgrade() -> None:
    op.drop_index('idx_forecast_run_station_time', table_name='forecast_runs')
    op.drop_table('forecast_runs')

    op.drop_index('idx_forecast_horizon', table_name='forecasts')
    with op.batch_alter_table('forecasts', naming_convention={'uq': 'uq_forecast_station_horizon'}) as batch:
        batch.drop_constraint('uq_forecast_station_horizon', type_='unique')

    for table in reversed(FK_TABLES):
        name = FK_CONSTRAINTS[table]
        with op.batch_alter_table(table, naming_convention={'fk': name}) as batch:
            batch.drop_constraint(name, type_='foreignkey')

    op.drop_column('pollution_observations', 're_stamped')
