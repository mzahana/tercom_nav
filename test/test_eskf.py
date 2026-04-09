"""Unit tests for the Error-State Kalman Filter."""
import numpy as np
import pytest


BASE_CONFIG = {
    'imu_rate_hz': 100.0,
    'accel_noise': 0.1,
    'gyro_noise': 0.01,
    'accel_bias_noise': 0.001,
    'gyro_bias_noise': 0.0001,
    'bias_time_constant': 300.0,
    'init_pos_std': 5.0,
    'init_vel_std': 1.0,
    'init_att_std': 0.05,
    'init_abias_std': 0.5,
    'init_wbias_std': 0.01,
    'world_origin_alt': 1859.7,
}


def make_eskf():
    from tercom_nav.core.eskf import ESKF
    eskf = ESKF(BASE_CONFIG.copy())
    p0 = np.array([100.0, 200.0, 50.0])
    v0 = np.array([5.0, 0.0, 0.0])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    eskf.initialize(p0, v0, q0, stamp_s=0.0)
    return eskf


class TestESKFInitialize:
    def test_state_set_correctly(self):
        eskf = make_eskf()
        assert eskf.initialized is True
        np.testing.assert_allclose(eskf.state.position, [100.0, 200.0, 50.0])
        np.testing.assert_allclose(eskf.state.velocity, [5.0, 0.0, 0.0])
        np.testing.assert_allclose(eskf.state.accel_bias, [0.0, 0.0, 0.0])

    def test_covariance_diagonal(self):
        eskf = make_eskf()
        # Diagonal should be positive
        diag = np.diag(eskf.P)
        assert np.all(diag > 0)
        # Position std should match init_pos_std
        assert abs(np.sqrt(diag[0]) - 5.0) < 0.01


class TestESKFPredict:
    def test_zero_imu_position_unchanged(self):
        """With zero accel/gyro (correcting for gravity), position should drift slowly."""
        from tercom_nav.core.eskf import ESKF
        eskf = ESKF(BASE_CONFIG.copy())
        # Initialize with identity quaternion
        eskf.initialize(np.zeros(3), np.zeros(3), np.array([1., 0., 0., 0.]), 0.0)

        # IMU reading that exactly cancels gravity in ENU: [0, 0, +9.81]
        # (sensor reads gravity as upward, bias-corrected)
        accel = np.array([0.0, 0.0, 9.81])  # body frame, z-up = no acceleration
        gyro = np.zeros(3)

        P_before = eskf.P.copy()
        eskf.predict(accel, gyro, stamp_s=0.01)

        # Position should be near zero
        assert np.linalg.norm(eskf.state.position) < 0.01

        # Covariance should grow (prediction adds process noise)
        assert np.trace(eskf.P) > np.trace(P_before)

    def test_covariance_grows_with_time(self):
        eskf = make_eskf()
        P_traces = [np.trace(eskf.P)]
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.zeros(3)
        for i in range(1, 6):
            eskf.predict(accel, gyro, stamp_s=i * 0.01)
            P_traces.append(np.trace(eskf.P))
        # Trace should be monotonically increasing
        for i in range(1, len(P_traces)):
            assert P_traces[i] > P_traces[i-1]

    def test_dt_from_timestamps(self):
        """dt computed correctly from consecutive stamps."""
        from tercom_nav.core.eskf import ESKF
        eskf = ESKF(BASE_CONFIG.copy())
        eskf.initialize(np.zeros(3), np.array([1.0, 0.0, 0.0]),
                        np.array([1., 0., 0., 0.]), 0.0)
        accel = np.array([0.0, 0.0, 9.81])
        eskf.predict(accel, np.zeros(3), stamp_s=1.0)
        # With v=1 m/s east and dt=1s, position should be ~1m east
        assert abs(eskf.state.position[0] - 1.0) < 0.1


class TestESKFUpdate:
    def test_update_reduces_covariance(self):
        eskf = make_eskf()
        P_before = np.trace(eskf.P)
        z_xy = eskf.state.position[:2].copy()  # perfect measurement
        R_xy = np.eye(2) * 25.0
        eskf.update_position_2d(z_xy, R_xy, stamp_s=0.1)
        assert np.trace(eskf.P) < P_before

    def test_update_corrects_position(self):
        """Update at true position should not change state."""
        eskf = make_eskf()
        true_pos = eskf.state.position.copy()
        z_xy = true_pos[:2]
        R_xy = np.eye(2) * 1.0
        eskf.update_position_2d(z_xy, R_xy, stamp_s=0.1)
        np.testing.assert_allclose(eskf.state.position[:2], true_pos[:2], atol=0.1)

    def test_nis_computation(self):
        """NIS should be finite and non-negative."""
        eskf = make_eskf()
        z_xy = eskf.state.position[:2] + np.array([5.0, -3.0])  # offset
        R_xy = np.eye(2) * 25.0
        result = eskf.update_position_2d(z_xy, R_xy, stamp_s=0.1)
        assert 'NIS' in result
        assert result['NIS'] >= 0.0
        assert np.isfinite(result['NIS'])

    def test_baro_update(self):
        eskf = make_eskf()
        # Measure altitude at expected position
        z_alt_msl = eskf.state.position[2] + BASE_CONFIG['world_origin_alt']
        result = eskf.update_altitude(z_alt_msl, 9.0, stamp_s=0.1)
        assert result['accepted'] is True

    def test_velocity_update(self):
        eskf = make_eskf()
        z_vel = eskf.state.velocity.copy()
        R_vel = np.eye(3) * 0.25
        result = eskf.update_velocity(z_vel, R_vel, stamp_s=0.1)
        assert result['accepted'] is True

    def test_covariance_symmetry(self):
        """P must remain symmetric after updates."""
        eskf = make_eskf()
        accel = np.array([0.1, 0.0, 9.81])
        for i in range(5):
            eskf.predict(accel, np.zeros(3), stamp_s=(i+1)*0.01)
        z_xy = eskf.state.position[:2] + np.array([2.0, -1.0])
        eskf.update_position_2d(z_xy, np.eye(2) * 25.0, stamp_s=0.06)
        diff = np.max(np.abs(eskf.P - eskf.P.T))
        assert diff < 1e-10, f"P not symmetric, max diff = {diff}"

    def test_before_init_no_crash(self):
        from tercom_nav.core.eskf import ESKF
        eskf = ESKF(BASE_CONFIG.copy())
        # Should not crash, just return gracefully
        result = eskf.update_position_2d(np.zeros(2), np.eye(2), stamp_s=0.0)
        assert result['accepted'] is False


class TestHelpers:
    def test_skew_symmetric(self):
        from tercom_nav.core.eskf import skew_symmetric
        v = np.array([1.0, 2.0, 3.0])
        S = skew_symmetric(v)
        assert S.shape == (3, 3)
        np.testing.assert_allclose(S, -S.T)
        # Cross product check: S @ e_x should equal v x e_x
        ex = np.array([1., 0., 0.])
        np.testing.assert_allclose(S @ ex, np.cross(v, ex), atol=1e-10)

    def test_quaternion_multiply_identity(self):
        from tercom_nav.core.eskf import quaternion_multiply
        q = np.array([0.9239, 0.3827, 0.0, 0.0])
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = quaternion_multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_rotation_matrix_orthogonal(self):
        from tercom_nav.core.eskf import quaternion_to_rotation_matrix
        q = np.array([0.9239, 0.3827, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10
