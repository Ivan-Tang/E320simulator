"""Tests for src/kalman_tracker.py and src/hough_baseline.py."""
import numpy as np
import pytest
from src.kalman_tracker import (
    KalmanConfig,
    _F, _Q, _R,
    _predict, _update,
    _build_seeds,
    _process_event_kalman,
)
from src.hough_baseline import (
    HoughConfig,
    _binning,
    _encode_key,
    _decode_key,
    _theta_rho_from_hits,
    _process_event_hough,
)

Z_LAYERS = np.array([0.0, 20.0, 40.0, 60.0, 80.0])


def _straight_track(ax=0.005, ay=0.003, bx=0.0, by=0.0, noise=0.0):
    rng = np.random.default_rng(42)
    z = Z_LAYERS.copy()
    x = ax * z + bx + rng.normal(0, noise, 5)
    y = ay * z + by + rng.normal(0, noise, 5)
    return x, y, z, np.arange(5, dtype=np.int8), np.arange(5, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# Kalman Tracker
# ══════════════════════════════════════════════════════════════════════════════

class TestKalmanMatrices:
    def setup_method(self):
        self.cfg = KalmanConfig()

    def test_F_is_4x4(self):
        assert _F(20.0).shape == (4, 4)

    def test_F_identity_at_zero_dz(self):
        np.testing.assert_array_equal(_F(0.0), np.eye(4))

    def test_F_propagates_position(self):
        """State [x=1, sx=0.5, y=0, sy=0], dz=10 → x = 1 + 0.5*10 = 6."""
        F = _F(10.0)
        state = np.array([1.0, 0.5, 0.0, 0.0])
        new_state = F @ state
        assert new_state[0] == pytest.approx(6.0, abs=1e-9)
        assert new_state[2] == pytest.approx(0.0, abs=1e-9)

    def test_Q_diagonal(self):
        Q = _Q(20.0, KalmanConfig())
        # Off-diagonal elements must be zero
        np.testing.assert_array_equal(Q - np.diag(np.diag(Q)), 0)

    def test_Q_scales_with_dz(self):
        Q10 = np.diag(_Q(10.0, KalmanConfig()))
        Q20 = np.diag(_Q(20.0, KalmanConfig()))
        np.testing.assert_array_almost_equal(Q20, 2 * Q10)

    def test_R_shape(self):
        assert _R(KalmanConfig()).shape == (2, 2)

    def test_R_contains_sigma_squared(self):
        cfg = KalmanConfig()
        R = _R(cfg)
        assert R[0, 0] == pytest.approx(cfg.sigma_x_mm ** 2, rel=1e-6)
        assert R[1, 1] == pytest.approx(cfg.sigma_y_mm ** 2, rel=1e-6)


class TestKalmanPredictUpdate:
    def setup_method(self):
        self.cfg = KalmanConfig()
        self.F = _F(20.0)
        self.Q = _Q(20.0, self.cfg)
        self.R = _R(self.cfg)
        self.state0 = np.array([0.0, 0.0, 0.0, 0.0])
        self.cov0 = np.eye(4) * 0.01

    def test_predict_increases_covariance_trace(self):
        _, cov_pred = _predict(self.state0, self.cov0, self.F, self.Q)
        assert np.trace(cov_pred) > np.trace(self.cov0)

    def test_predict_zero_Q_unchanged_cov(self):
        Q_zero = np.zeros((4, 4))
        _, cov_pred = _predict(self.state0, self.cov0, self.F, Q_zero)
        expected = self.F @ self.cov0 @ self.F.T
        np.testing.assert_array_almost_equal(cov_pred, expected)

    def test_update_pulls_state_toward_measurement(self):
        """With measurement at x=5, update should pull state x toward 5."""
        state_pred = np.array([0.0, 0.0, 0.0, 0.0])
        cov_pred = np.eye(4)
        meas = np.array([5.0, 0.0])
        state_upd, _, _ = _update(state_pred, cov_pred, meas, self.R)
        assert 0.0 < state_upd[0] < 5.0, f"Expected x in (0,5), got {state_upd[0]}"

    def test_update_decreases_covariance_trace(self):
        cov_pred = np.eye(4)
        meas = np.array([0.0, 0.0])
        _, cov_upd, _ = _update(self.state0, cov_pred, meas, self.R)
        assert np.trace(cov_upd) < np.trace(cov_pred)

    def test_perfect_measurement_near_zero_chi2(self):
        """Measurement exactly at predicted position → chi2 near 0."""
        state_pred = np.array([3.0, 0.01, 1.0, 0.005])
        meas = np.array([state_pred[0], state_pred[2]])  # x and y from state
        _, _, chi2 = _update(state_pred, np.eye(4) * 1e6, meas, self.R)
        assert chi2 == pytest.approx(0.0, abs=1e-3)


class TestBuildSeeds:
    def test_returns_seeds_for_valid_pair(self):
        x, y, z, layer, nid = _straight_track()
        seeds = _build_seeds(x, y, z, layer, nid, KalmanConfig())
        assert len(seeds) >= 1

    def test_no_layer0_returns_empty(self):
        x = np.array([0.0, 0.0])
        y = np.array([0.0, 0.0])
        z = np.array([20.0, 40.0])
        layer = np.array([1, 2], dtype=np.int8)
        nid = np.array([0, 1], dtype=np.int64)
        seeds = _build_seeds(x, y, z, layer, nid, KalmanConfig())
        assert seeds == []

    def test_seed_state_is_4d(self):
        x, y, z, layer, nid = _straight_track()
        seeds = _build_seeds(x, y, z, layer, nid, KalmanConfig())
        assert seeds[0][0].shape == (4,)

    def test_seed_covariance_is_4x4(self):
        x, y, z, layer, nid = _straight_track()
        seeds = _build_seeds(x, y, z, layer, nid, KalmanConfig())
        assert seeds[0][1].shape == (4, 4)


class TestProcessEventKalman:
    def test_recovers_track(self):
        x, y, z, layer, nid = _straight_track()
        result = _process_event_kalman(0, x, y, z, layer, nid, KalmanConfig())
        kept = [c for c in result if c["is_kept"]]
        assert len(kept) >= 1
        assert max(c["n_layers"] for c in kept) >= 4

    def test_empty_event_returns_empty(self):
        result = _process_event_kalman(
            0,
            np.array([0.0]), np.array([0.0]), np.array([0.0]),
            np.array([0], dtype=np.int8), np.array([0], dtype=np.int64),
            KalmanConfig(),
        )
        assert result == []

    def test_candidate_fields(self):
        x, y, z, layer, nid = _straight_track()
        result = _process_event_kalman(0, x, y, z, layer, nid, KalmanConfig())
        if result:
            required = {"event_id", "candidate_id", "node_ids", "n_layers",
                        "ax", "bx", "ay", "by", "chi2", "rms", "is_kept"}
            assert required.issubset(set(result[0].keys()))


# ══════════════════════════════════════════════════════════════════════════════
# Hough Tracker
# ══════════════════════════════════════════════════════════════════════════════

class TestHoughHelpers:
    def setup_method(self):
        self.cfg = HoughConfig()

    def test_binning_returns_eight_values(self):
        result = _binning(self.cfg)
        assert len(result) == 8

    def test_encode_decode_roundtrip(self):
        n_rho_x = n_theta_y = n_rho_y = 100
        jtx, jrx, jty, jry = 3, 5, 7, 9
        key = _encode_key(
            np.array([jtx]), np.array([jrx]), np.array([jty]), np.array([jry]),
            n_rho_x, n_theta_y, n_rho_y,
        )
        out_jtx, out_jrx, out_jty, out_jry = _decode_key(
            int(key[0]), n_rho_x, n_theta_y, n_rho_y
        )
        assert (out_jtx, out_jrx, out_jty, out_jry) == (jtx, jrx, jty, jry)

    def test_theta_in_range_zero_to_pi(self):
        """Returned theta values must lie in [0, π)."""
        k1 = np.random.default_rng(0).uniform(-5, 5, 20)
        k2 = np.random.default_rng(1).uniform(-5, 5, 20)
        z1 = np.zeros(20)
        z2 = np.ones(20) * 20.0
        theta, _ = _theta_rho_from_hits(k1, k2, z1, z2, self.cfg)
        assert np.all(theta >= 0.0)
        assert np.all(theta < np.pi)

    def test_theta_rho_from_hits_known_pair(self):
        """Analytical check: two hits on x-axis with x2=x1+dz*slope → known theta."""
        k1 = np.array([0.0])
        k2 = np.array([0.1])  # slope=0.1/20 in x
        z1 = np.array([0.0])
        z2 = np.array([20.0])
        theta, rho = _theta_rho_from_hits(k1, k2, z1, z2, self.cfg)
        # rho = k1*sin(theta) + z1*cos(theta) = 0 (k1=z1=0)
        assert rho[0] == pytest.approx(0.0, abs=1e-9)


class TestProcessEventHough:
    def test_recovers_track(self):
        x, y, z, layer, nid = _straight_track()
        result = _process_event_hough(0, x, y, z, layer, nid, HoughConfig())
        kept = [c for c in result if c["is_kept"]]
        assert len(kept) >= 1

    def test_candidate_fields(self):
        x, y, z, layer, nid = _straight_track()
        result = _process_event_hough(0, x, y, z, layer, nid, HoughConfig())
        if result:
            required = {"event_id", "candidate_id", "node_ids", "n_layers",
                        "ax", "bx", "ay", "by", "chi2", "rms", "is_kept"}
            assert required.issubset(set(result[0].keys()))
