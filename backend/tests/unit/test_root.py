"""Tests for the API root landing behaviour (`GET /`).

A bare browser visit to the deployed backend URL previously answered
``{"detail": "Not Found"}``. When ``FRONTEND_URL`` is configured the root
must 307-redirect to the web app; otherwise it degrades to a useful landing
JSON pointing at the docs and health probes.
"""

from __future__ import annotations

from app.main import settings


def test_root_redirects_to_frontend_when_configured(client, monkeypatch):
    monkeypatch.setattr(
        settings, "frontend_url", "https://air-pollution-weather-coupled-forec-eight.vercel.app"
    )
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert (
        resp.headers["location"]
        == "https://air-pollution-weather-coupled-forec-eight.vercel.app/"
    )


def test_root_lands_on_docs_links_without_frontend_url(client, monkeypatch):
    monkeypatch.setattr(settings, "frontend_url", "")
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 200
    body = resp.json()
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"
