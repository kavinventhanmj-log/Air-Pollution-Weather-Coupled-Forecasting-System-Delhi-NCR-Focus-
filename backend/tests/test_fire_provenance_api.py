"""API-level fire provenance: synthetic history must never surface as real.

SIH26082 regression guards:
* every operational endpoint (hotspots, fire-activity, plume-risk, latest,
  summary, data-quality) reports real FIRMS observations only;
* the simulated overlay is opt-in (``include_synthetic=1``) and every returned
  hotspot is labelled with its provenance;
* a dense cluster of recent synthetic fires alone can no longer push
  plume-risk to HIGH (the saturated-score defect).
"""

from datetime import UTC, datetime, timedelta

from app.models.db_models import FireReading

SYNTHETIC_SOURCE = "synthetic_sim"


def _seed_synthetic(db_session, n=6):
    base = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    for i in range(n):
        db_session.add(
            FireReading(
                latitude=round(29.8 + i * 0.1, 4),
                longitude=round(75.8 + i * 0.1, 4),
                acq_date=base + timedelta(minutes=i * 5),
                confidence="high",
                frp=200.0,
                satellite="Aqua",
                daynight="D",
                synthetic=True,
                source=SYNTHETIC_SOURCE,
            )
        )
    db_session.commit()


def _drop_real_fires(db_session):
    for f in db_session.query(FireReading).filter(FireReading.synthetic.is_(False)).all():
        db_session.delete(f)
    db_session.commit()


def test_hotspots_default_excludes_synthetic(client, db_session):
    _seed_synthetic(db_session, n=3)
    body = client.get("/api/fire/hotspots").json()
    hotspots = body["hotspots"]
    assert hotspots, "seeded real fires should be returned"
    assert all(h["synthetic"] is False for h in hotspots)

    # The recent synthetic rows do exist in the DB — they are just hidden here.
    later = client.get("/api/fire/hotspots", params={"include_synthetic": True}).json()
    included = later["hotspots"]
    labeled = [h for h in included if h["synthetic"]]
    assert labeled, "include_synthetic=1 must surface the simulated overlay"
    assert all(h["source"] == SYNTHETIC_SOURCE for h in labeled)
    assert all(h["synthetic"] is False for h in included if not h["synthetic"])


def test_hotspots_variants_use_separate_cache_entries(client, db_session, monkeypatch):
    """The two variants must never share one cached entry.

    Under a single cache key the entry written by whichever variant landed
    first was served to the other for the next 5 minutes: the real-only payload
    hid the simulated overlay from ``include_synthetic=1``, and the simulated
    payload put synthetic points on the default operational map. Production
    enables this cache (Postgres); under pytest it is refused, so it is enabled
    explicitly here to exercise the real request path.
    """
    from app.services import ttl_cache

    monkeypatch.setattr(ttl_cache, "_cache_enabled", lambda: True)
    ttl_cache.invalidate_all()

    _seed_synthetic(db_session, n=3)

    default = client.get("/api/fire/hotspots").json()["hotspots"]
    with_synthetic = client.get("/api/fire/hotspots", params={"include_synthetic": True}).json()["hotspots"]

    assert all(h["synthetic"] is False for h in default)
    assert any(h["synthetic"] for h in with_synthetic)

    # Repeat in the opposite order, still warm: neither variant may be answered
    # from the other's entry.
    with_synthetic_again = client.get("/api/fire/hotspots", params={"include_synthetic": True}).json()["hotspots"]
    default_again = client.get("/api/fire/hotspots").json()["hotspots"]

    assert len(with_synthetic_again) == len(with_synthetic)
    assert len(default_again) == len(default)
    assert all(h["synthetic"] is False for h in default_again)
    assert any(h["synthetic"] for h in with_synthetic_again)

    # Two dimensions, two entries.
    assert ttl_cache.cache_info()["entries"] == 2
    ttl_cache.invalidate_all()


def test_hotspots_synthetic_first_when_most_recent(client, db_session):
    # Synthetic rows are *newer* than the seeded real ones: without filtering
    # they would be the top of every result set and dominate the map.
    _seed_synthetic(db_session, n=4)
    body = client.get("/api/fire/hotspots").json()
    assert all(h["synthetic"] is False for h in body["hotspots"])


def test_plume_risk_is_not_saturated_by_synthetic_cluster(client, db_session):
    """The original defect: ~490 synthetic fires around Delhi => clamped to 1.0.

    With real observations removed, a dense synthetic cluster must produce an
    empty, LOW result and an explicit provenance note — never a HIGH score.
    """
    _drop_real_fires(db_session)
    _seed_synthetic(db_session, n=8)
    body = client.get("/api/plume-risk").json()
    assert body["fire_count"] == 0
    assert body["risk_score"] == 0.1
    assert body["risk_level"] == "LOW"
    assert body["synthetic_fire_count"] >= 8
    assert "excluded" in body["fire_basis"]


def test_plume_risk_counts_real_fires_only(client, db_session):
    _seed_synthetic(db_session, n=6)
    body = client.get("/api/plume-risk").json()
    assert body["fire_count"] == 5  # the conftest real seeds
    assert body["synthetic_fire_count"] == 6
    assert 0 < body["risk_score"] <= 1 and body["risk_level"] in {"LOW", "MODERATE", "HIGH"}
    assert "Real FIRMS" in body["fire_basis"]


def test_fire_activity_excludes_synthetic(client, db_session):
    _seed_synthetic(db_session, n=5)
    body = client.get("/api/fire-activity").json()
    assert body["total_fires"] == 5  # real seeds only


def test_latest_fires_exclude_synthetic_and_label_provenance_optional(client, db_session):
    _seed_synthetic(db_session, n=3)
    body = client.get("/api/fires/latest", params={"hours": 48}).json()
    assert body["count"] == 5
    assert all(e["synthetic"] is False for e in body["fires"])


def test_summary_active_fires_is_real_only(client, db_session):
    _drop_real_fires(db_session)
    _seed_synthetic(db_session, n=4)
    body = client.get("/api/summary").json()
    assert body["active_fires_24h"] == 0

    db_session.add(
        FireReading(
            latitude=30.5,
            longitude=76.1,
            acq_date=datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1),
            confidence="high",
            frp=90.0,
            satellite="SNPP",
            daynight="D",
            synthetic=False,
            source="firms_live",
        )
    )
    db_session.commit()
    body = client.get("/api/summary").json()
    assert body["active_fires_24h"] >= 1


def test_data_quality_splits_fire_provenance(client, db_session):
    _seed_synthetic(db_session, n=3)
    body = client.get("/api/data-quality").json()
    fr = body["tables"]["fire_readings"]
    assert fr["synthetic"] >= 3
    assert fr["real"] == 5
    assert fr["total"] == fr["synthetic"] + fr["real"]


def test_real_events_and_hotspots_carry_source_label(client, db_session):
    """Live/real rows must be distinguishable from synthetic in the JSON."""
    _seed_synthetic(db_session, n=2)
    latest = client.get("/api/fires/latest", params={"hours": 48}).json()
    assert {"synthetic", "source"} <= set(latest["fires"][0])
    hotspots = client.get("/api/fire/hotspots", params={"include_synthetic": True}).json()
    assert {"synthetic", "source"} <= set(hotspots["hotspots"][0])
