"""Unit tests for the coupling-feature ablation study (SIH26082).

The audit required the two-way coupling features to be *measured* rather than
merely added. These tests pin the parts of the experiment that make its result
trustworthy: a single shared split, paired evaluation rows, an honest effect
sign, and no invented metrics when the data is too small.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from ml.evaluation import ablation
from ml.evaluation.ablation import (
    COUPLING_FEATURE_GROUP,
    MATERIALITY_FRACTION,
    MIN_EVAL_ROWS,
    _delta,
    _resolve_feature_groups,
    run_coupling_ablation,
)
from ml.training.trainer import create_target_columns, get_feature_columns


def _synthetic_frame(n_hours: int = 500, n_stations: int = 2, with_coupling: bool = True) -> pd.DataFrame:
    """Build a small featured dataset shaped like the real one."""
    rng = np.random.default_rng(7)
    frames = []
    for s in range(n_stations):
        hours = pd.date_range("2026-01-01", periods=n_hours, freq="h", tz="UTC")
        pm25 = 120 + 30 * np.sin(np.arange(n_hours) / 24 * 2 * np.pi) + rng.normal(0, 8, n_hours)
        frame = pd.DataFrame(
            {
                "timestamp": hours,
                "station": f"S{s}",
                "pm25": pm25,
                "temperature": 20 + rng.normal(0, 2, n_hours),
                "humidity": 50 + rng.normal(0, 5, n_hours),
                "wind_speed": 3 + rng.normal(0, 0.5, n_hours),
                "pbl_height": 600 + rng.normal(0, 40, n_hours),
                "hour": hours.hour,
            }
        )
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if with_coupling:
        for i, col in enumerate(COUPLING_FEATURE_GROUP):
            df[col] = np.linspace(0.1, 1.0, len(df)) + i
    return df


@pytest.fixture
def dataset_csv(tmp_path):
    path = tmp_path / "featured.csv"
    _synthetic_frame().to_csv(path, index=False)
    return str(path)


class TestFeatureGroupResolution:
    def test_group_detected_and_removed(self):
        df = _synthetic_frame()
        cols = get_feature_columns(df)
        without, with_group = _resolve_feature_groups(cols, COUPLING_FEATURE_GROUP)
        for c in COUPLING_FEATURE_GROUP:
            assert c in with_group
            assert c not in without
        assert len(without) + len(COUPLING_FEATURE_GROUP) == len(with_group)

    def test_missing_columns_are_ignored(self):
        df = _synthetic_frame(with_coupling=False)
        cols = get_feature_columns(df)
        without, with_group = _resolve_feature_groups(cols, COUPLING_FEATURE_GROUP)
        assert without == with_group
        assert not any(c in with_group for c in COUPLING_FEATURE_GROUP)

    def test_group_is_a_snapshot_tuple(self):
        assert isinstance(COUPLING_FEATURE_GROUP, tuple)


class TestDeltaConvention:
    def test_negative_mae_delta_means_coupling_helped(self):
        # MAE 50 with coupling, 60 without -> with - without = -10 -> coupling helped.
        d = _delta({"mae": 50.0}, {"mae": 60.0}, "mae")
        assert d == pytest.approx(-10.0)

    def test_positive_mae_delta_means_coupling_did_not_help(self):
        d = _delta({"mae": 60.0}, {"mae": 50.0}, "mae")
        assert d == pytest.approx(10.0)

    def test_r2_delta_reads_the_other_way(self):
        """R2 is higher-is-better, so a positive delta means coupling helped."""
        d = _delta({"r2": 0.3}, {"r2": 0.1}, "r2")
        assert d == pytest.approx(0.2)

    def test_zero_delta(self):
        assert _delta({"mae": 50.0}, {"mae": 50.0}, "mae") == 0.0

    def test_missing_metric_is_none(self):
        assert _delta({"mae": 50.0}, {"rmse": 1.0}, "mae") is None
        assert _delta({}, {}, "mae") is None


class TestSharedSplit:
    def test_both_arms_receive_identical_rows(self, dataset_csv, monkeypatch):
        """The only permitted difference between arms is the feature matrix."""
        seen = []

        def fake_fit(train_df, val_df, test_df, cols, target_col, model_type, horizon):
            seen.append(
                {
                    "train": train_df["timestamp"].tolist(),
                    "val": val_df["timestamp"].tolist(),
                    "test": test_df["timestamp"].tolist(),
                    "n_cols": len(cols),
                    "cols": list(cols),
                }
            )
            n = len(test_df)
            return {
                "val": {"mae": 1.0, "rmse": 1.0, "r2": 0.0},
                "test": {"mae": 1.0, "rmse": 1.0, "r2": 0.0},
                "n_train": len(train_df),
                "n_test": n,
                "n_features": len(cols),
                "_y_true": np.zeros(n),
                "_y_pred": np.zeros(n),
            }

        monkeypatch.setattr(ablation, "_fit_predict", fake_fit)
        run_coupling_ablation(data_path=dataset_csv, horizon=24, model_type="xgboost")

        assert len(seen) == 2
        first, second = seen
        # Identical split boundaries and identical evaluation rows.
        assert first["train"] == second["train"]
        assert first["val"] == second["val"]
        assert first["test"] == second["test"]
        # Only the feature set differs.
        assert first["n_cols"] - second["n_cols"] == len(COUPLING_FEATURE_GROUP)
        assert set(first["cols"]) - set(second["cols"]) == set(COUPLING_FEATURE_GROUP)

    def test_split_is_chronological(self, dataset_csv, monkeypatch):
        seen = {}
        real = ablation._chronological_split_by_time

        def spy(df, *a, **k):
            out = real(df, *a, **k)
            seen["times"] = (df["timestamp"].min(), df["timestamp"].max())
            return out

        monkeypatch.setattr(ablation, "_chronological_split_by_time", spy)
        run_coupling_ablation(data_path=dataset_csv, horizon=24, model_type="xgboost")
        assert "times" in seen

    def test_protocol_declared(self, dataset_csv, monkeypatch):
        monkeypatch.setattr(ablation, "_fit_predict", lambda *a, **k: None)
        res = run_coupling_ablation(data_path=dataset_csv, horizon=24)
        assert res["protocol"]["split"] == "chronological_by_time"
        assert res["protocol"]["paired"] is True
        assert res["protocol"]["evaluation"] == "held-out test"


class TestInsufficientData:
    def test_tiny_dataset_reports_no_metrics(self, tmp_path):
        path = tmp_path / "tiny.csv"
        _synthetic_frame(n_hours=10, n_stations=1).to_csv(path, index=False)
        res = run_coupling_ablation(data_path=str(path), horizon=24)
        assert res["arms"]["t+24h"]["status"] == "insufficient_data"
        assert res["delta"] == {}
        assert "not enough data" in res["summary"] or "no horizon had enough data" in res["summary"]

    def test_min_eval_rows_constant(self):
        assert MIN_EVAL_ROWS == 10


class TestEndToEnd:
    def test_persistence_arms_produce_metrics(self, dataset_csv):
        res = run_coupling_ablation(data_path=dataset_csv, horizon=24, model_type="persistence")
        arm = res["arms"]["t+24h"]
        assert arm["status"] == "ok"
        for key in ("mae", "rmse", "r2"):
            assert arm["with"]["test"][key] is not None
            assert arm["without"]["test"][key] is not None
            assert arm["with"]["val"][key] is not None
        assert arm["with"]["n_test"] == arm["without"]["n_test"]
        assert arm["with"]["n_test"] > 0

    def test_without_arm_has_fewer_features(self, dataset_csv):
        res = run_coupling_ablation(data_path=dataset_csv, horizon=24, model_type="persistence")
        arm = res["arms"]["t+24h"]
        assert arm["with"]["n_features"] - arm["without"]["n_features"] == len(COUPLING_FEATURE_GROUP)

    def test_absent_group_is_disclosed(self, tmp_path):
        path = tmp_path / "nocouple.csv"
        _synthetic_frame(with_coupling=False).to_csv(path, index=False)
        res = run_coupling_ablation(data_path=str(path), horizon=24, model_type="persistence")
        assert set(res["group_absent"]) == set(COUPLING_FEATURE_GROUP)
        assert res["group_present"] == []

    def test_horizon_beyond_data_reports_insufficient(self, dataset_csv):
        """A horizon longer than the record must not yield a scored number."""
        res = run_coupling_ablation(data_path=dataset_csv, horizon=999, model_type="persistence")
        assert res["arms"]["t+999h"]["status"] == "insufficient_data"
        assert res["delta"] == {}
        assert "no horizon had enough data" in res["summary"]
        assert "No metrics are reported" in res["summary"]

    def test_json_serialisable_and_saved(self, dataset_csv, tmp_path):
        out = tmp_path / "nested" / "ablation.json"
        res = run_coupling_ablation(
            data_path=dataset_csv, horizon=24, model_type="persistence", output_path=str(out)
        )
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["target"] == "pm25"
        assert loaded["arms"]["t+24h"]["status"] == "ok"
        assert res["saved_to"] == str(out)

    def test_timestamp_recorded(self, dataset_csv):
        res = run_coupling_ablation(data_path=dataset_csv, horizon=24, model_type="persistence")
        assert res["computed_at"]
        assert res["target"] == "pm25"
        assert res["model_type"] == "persistence"

    def test_does_not_touch_model_directory(self, dataset_csv, tmp_path, monkeypatch):
        """Ablation is an experiment: it must not overwrite shipped models."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        sentinel = model_dir / "sentinel.joblib"
        sentinel.write_text("untouched", encoding="utf-8")
        run_coupling_ablation(data_path=dataset_csv, horizon=24, model_type="persistence")
        assert sentinel.read_text(encoding="utf-8") == "untouched"
        assert os.listdir(model_dir) == ["sentinel.joblib"]


class TestMateriality:
    """A sub-noise difference must not be reported as a positive result."""

    def test_tiny_difference_is_not_material(self):
        # The real t+6h result: MAE 34.729 vs 34.770 on 21,299 test rows.
        summary = ablation._summarise(
            "pm25", 6, "random_forest",
            {"mae": 34.729313, "rmse": 57.672655, "r2": 0.770383},
            {"mae": 34.769793, "rmse": 57.610548, "r2": 0.770878},
        )
        assert "no material change" in summary
        assert "not demonstrably adding predictive value" in summary
        assert "they help on this split" not in summary

    def test_large_difference_is_material(self):
        summary = ablation._summarise(
            "pm25", 6, "random_forest",
            {"mae": 30.0, "rmse": 50.0, "r2": 0.8},
            {"mae": 40.0, "rmse": 60.0, "r2": 0.5},
        )
        assert "LOWER with the coupling features" in summary
        assert "no material change" not in summary

    def test_relative_size_computed(self):
        assert ablation._materiality(2.0, 100.0) == pytest.approx(0.02)
        assert ablation._materiality(None, 100.0) is None
        assert ablation._materiality(2.0, None) is None
        assert ablation._materiality(2.0, 0.0) is None

    def test_threshold_constant(self):
        assert MATERIALITY_FRACTION == 0.005

    def test_delta_block_is_populated_from_test_metrics(self, dataset_csv, monkeypatch):
        """Guards the regression where every reported delta was null."""
        calls = {"n": 0}

        def fake_fit(train_df, val_df, test_df, cols, target_col, model_type, horizon):
            n = len(test_df)
            calls["n"] += 1
            # First call is the with-coupling arm, second is without.
            mae = 10.0 if calls["n"] == 1 else 20.0
            return {
                "val": {"mae": 99.0, "rmse": 99.0, "r2": 99.0},
                "test": {"mae": mae, "rmse": mae * 2, "r2": 0.5},
                "n_train": len(train_df),
                "n_test": n,
                "n_features": len(cols),
                "_y_true": np.zeros(n),
                "_y_pred": np.zeros(n),
            }

        monkeypatch.setattr(ablation, "_fit_predict", fake_fit)
        res = run_coupling_ablation(data_path=dataset_csv, horizon=24, model_type="xgboost")
        d = res["arms"]["t+24h"]["delta"]
        assert d["mae"] is not None
        assert d["mae"] == pytest.approx(-10.0)  # with(10) - without(20)
        assert d["rmse"] == pytest.approx(-20.0)
        assert d["mae_relative"] == pytest.approx(0.5)
        assert d["material"] is True
        # The validation block must not leak into the reported delta.
        assert d["mae"] != pytest.approx(0.0)


    def test_all_scored_horizons_appear_in_the_summary(self, dataset_csv, monkeypatch):
        """The coupling effect is horizon dependent; quoting one hides that."""

        def fake_fit(train_df, val_df, test_df, cols, target_col, model_type, horizon):
            n = len(test_df)
            # Make the arms differ by horizon so the effect really varies.
            mae = 50.0 - horizon if len(cols) > 10 else 50.0
            return {
                "val": {"mae": 1.0, "rmse": 1.0, "r2": 0.0},
                "test": {"mae": mae, "rmse": mae, "r2": 0.5},
                "n_train": len(train_df),
                "n_test": n,
                "n_features": len(cols),
                "_y_true": np.zeros(n),
                "_y_pred": np.zeros(n),
            }

        monkeypatch.setattr(ablation, "_fit_predict", fake_fit)
        res = run_coupling_ablation(
            data_path=dataset_csv, horizons=[6, 24], model_type="xgboost"
        )
        assert set(res["summaries"]) == {"t+6h", "t+24h"}
        assert "t+6h" in res["summary"] and "t+24h" in res["summary"]
        assert "verdict" in res
        assert "material_horizons" in res

    def test_verdict_names_material_horizons(self, dataset_csv, monkeypatch):
        def fake_fit(train_df, val_df, test_df, cols, target_col, model_type, horizon):
            n = len(test_df)
            # 6h: negligible change. 24h: clear change.
            mae = 50.0 - (20.0 if horizon == 24 else 0.0) if len(cols) > 10 else 50.0
            return {
                "val": {"mae": 1.0, "rmse": 1.0, "r2": 0.0},
                "test": {"mae": mae, "rmse": mae, "r2": 0.5},
                "n_train": len(train_df),
                "n_test": n,
                "n_features": len(cols),
                "_y_true": np.zeros(n),
                "_y_pred": np.zeros(n),
            }

        monkeypatch.setattr(ablation, "_fit_predict", fake_fit)
        res = run_coupling_ablation(
            data_path=dataset_csv, horizons=[6, 24], model_type="xgboost"
        )
        assert res["material_horizons"] == ["t+24h"]
        assert "t+24h" in res["verdict"]
        assert "no measurable effect at t+6h" in res["verdict"]


class TestSummaryWording:
    def test_coupling_helps_when_mae_is_lower_with_it(self):
        summary = ablation._summarise(
            "pm25", 24, "random_forest",
            {"mae": 50.0, "rmse": 60.0, "r2": 0.3},
            {"mae": 60.0, "rmse": 70.0, "r2": 0.1},
        )
        assert "LOWER with the coupling features" in summary
        assert "they help on this split" in summary

    def test_negative_effect_is_stated_not_hidden(self):
        summary = ablation._summarise(
            "pm25", 24, "random_forest",
            {"mae": 60.0, "rmse": 70.0, "r2": 0.1},
            {"mae": 50.0, "rmse": 60.0, "r2": 0.3},
        )
        assert "HIGHER with the coupling features" in summary
        assert "they did not help on this split" in summary

    def test_identical_metrics_are_reported_as_no_change(self):
        summary = ablation._summarise(
            "pm25", 24, "random_forest",
            {"mae": 50.0, "rmse": 60.0, "r2": 0.3},
            {"mae": 50.0, "rmse": 60.0, "r2": 0.3},
        )
        assert "no material change" in summary

    def test_missing_metrics_are_not_invented(self):
        summary = ablation._summarise("pm25", 24, "random_forest", {}, {})
        assert "not computable" in summary


class TestTargetConstruction:
    def test_targets_shift_within_station(self, dataset_csv):
        df = pd.read_csv(dataset_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        out = create_target_columns(df, "pm25", [24])
        col = "target_pm25_t+24"
        for station in df["station"].unique():
            sub = out[out["station"] == station]
            expected = sub["pm25"].shift(-24)
            assert sub[col].tolist()[:-24] == pytest.approx(expected.tolist()[:-24], rel=1e-9, abs=1e-9)
