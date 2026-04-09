"""Error-State Kalman Filter (ESKF) for TERCOM GPS-denied navigation.

The ESKF separates state into:
- Nominal state: propagated by IMU integration (large signals)
- Error state: estimated by Kalman filter (small deviations)

This is more numerically stable than a standard EKF for navigation
because orientation errors remain small in the error-state formulation.

State vector (error state, 15 dimensions):
    dx = [dp(3), dv(3), dtheta(3), da_b(3), dw_b(3)]

Quaternion convention: [w, x, y, z] (scalar first, scipy convention)
Gravity in local ENU: [0, 0, -9.81] m/s^2 (z-up, gravity points down)

References:
    Joan Sola, "A micro Lie theory for state estimation in robotics" (2018)
    Titterton & Weston, "Strapdown Inertial Navigation Technology" (2004)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial.transform import Rotation


# Gravity vector in local ENU frame (z is Up, so gravity points -z)
GRAVITY_ENU = np.array([0.0, 0.0, -9.81])


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric (cross-product) matrix for vector v."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix (body->ENU)."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def small_angle_quaternion(dtheta: np.ndarray) -> np.ndarray:
    """Convert small rotation vector to quaternion [w,x,y,z].

    For small angles: q = [1, dtheta/2]
    """
    half = dtheta * 0.5
    w = 1.0 / np.sqrt(1.0 + np.dot(half, half))
    xyz = half * w
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


@dataclass
class NominalState:
    """Nominal state propagated by IMU integration.

    Attributes:
        position: [x, y, z] in local ENU frame (meters)
        velocity: [vx, vy, vz] in local ENU frame (m/s)
        quaternion: [w, x, y, z] body-to-ENU rotation
        accel_bias: accelerometer bias in body frame (m/s^2)
        gyro_bias: gyroscope bias in body frame (rad/s)
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> 'NominalState':
        return NominalState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            accel_bias=self.accel_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
        )


class ESKF:
    """Error-State Kalman Filter for IMU + TERCOM + barometric fusion.

    Usage:
        eskf = ESKF(config)
        eskf.initialize(position, velocity, quaternion, stamp_s)

        # IMU callback (high rate):
        eskf.predict(accel, gyro, stamp_s)

        # TERCOM fix callback:
        result = eskf.update_position_2d(z_xy, R_xy, stamp_s)

        # Barometric callback:
        eskf.update_altitude(z_alt, R_alt, stamp_s)

        # Velocity callback:
        eskf.update_velocity(z_vel, R_vel, stamp_s)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys matching eskf_node ROS parameters:
                imu_rate_hz, accel_noise, gyro_noise, accel_bias_noise,
                gyro_bias_noise, bias_time_constant,
                init_pos_std, init_vel_std, init_att_std,
                init_abias_std, init_wbias_std,
                world_origin_alt (for baro conversion)
        """
        self.config = config
        self.state = NominalState()
        self.dx = np.zeros(15)       # Error state
        self.P = np.eye(15)          # Error-state covariance
        self.initialized = False
        self.last_stamp_s: Optional[float] = None

    def initialize(self, position: np.ndarray, velocity: np.ndarray,
                   quaternion: np.ndarray, stamp_s: float) -> None:
        """Set initial state from GPS fix and IMU readings.

        Args:
            position: [x, y, z] in local ENU (meters)
            velocity: [vx, vy, vz] in local ENU (m/s)
            quaternion: [w, x, y, z] body-to-ENU
            stamp_s: Initialization timestamp in seconds
        """
        self.state.position = position.copy()
        self.state.velocity = velocity.copy()
        self.state.quaternion = normalize_quaternion(quaternion.copy())
        self.state.accel_bias = np.zeros(3)
        self.state.gyro_bias = np.zeros(3)
        self.dx = np.zeros(15)

        # Build initial covariance from config
        p = self.config.get('init_pos_std', 5.0)
        v = self.config.get('init_vel_std', 1.0)
        a = self.config.get('init_att_std', 0.05)
        ab = self.config.get('init_abias_std', 0.5)
        wb = self.config.get('init_wbias_std', 0.01)

        self.P = np.diag(np.concatenate([
            [p**2]*3, [v**2]*3, [a**2]*3, [ab**2]*3, [wb**2]*3
        ]))

        self.last_stamp_s = stamp_s
        self.initialized = True

    def predict(self, accel: np.ndarray, gyro: np.ndarray, stamp_s: float) -> None:
        """IMU-driven prediction step.

        Propagates nominal state by direct integration and updates
        error-state covariance via the linearized transition matrix.

        Args:
            accel: Raw accelerometer [ax, ay, az] in body frame (m/s^2)
            gyro: Raw gyroscope [wx, wy, wz] in body frame (rad/s)
            stamp_s: IMU message timestamp in seconds
        """
        if not self.initialized:
            return

        # Compute dt from timestamps
        if self.last_stamp_s is None:
            dt = 1.0 / self.config.get('imu_rate_hz', 50.0)
        else:
            dt = stamp_s - self.last_stamp_s
            if dt <= 0 or dt > 1.0:
                dt = 1.0 / self.config.get('imu_rate_hz', 50.0)
        self.last_stamp_s = stamp_s

        # Bias-corrected measurements
        a_corr = accel - self.state.accel_bias
        w_corr = gyro - self.state.gyro_bias

        # Rotation matrix body -> ENU
        R = quaternion_to_rotation_matrix(self.state.quaternion)

        # Acceleration in ENU (rotate body accel, add gravity)
        a_enu = R @ a_corr + GRAVITY_ENU

        # Nominal state propagation
        self.state.position += self.state.velocity * dt + 0.5 * a_enu * dt**2
        self.state.velocity += a_enu * dt
        q_delta = small_angle_quaternion(w_corr * dt)
        self.state.quaternion = normalize_quaternion(
            quaternion_multiply(self.state.quaternion, q_delta)
        )

        # Error-state transition matrix F = I + Fx*dt
        tau = self.config.get('bias_time_constant', 300.0)
        Fx = np.zeros((15, 15))
        Fx[0:3, 3:6] = np.eye(3)
        Fx[3:6, 6:9] = -R @ skew_symmetric(a_corr)
        Fx[3:6, 9:12] = -R
        Fx[6:9, 6:9] = -skew_symmetric(w_corr)
        Fx[6:9, 12:15] = -np.eye(3)
        Fx[9:12, 9:12] = -(1.0 / tau) * np.eye(3)
        Fx[12:15, 12:15] = -(1.0 / tau) * np.eye(3)

        F = np.eye(15) + Fx * dt

        # Process noise covariance Q
        sa = self.config.get('accel_noise', 0.1)
        sw = self.config.get('gyro_noise', 0.01)
        sab = self.config.get('accel_bias_noise', 0.001)
        swb = self.config.get('gyro_bias_noise', 0.0001)

        Q = np.zeros((15, 15))
        Q[3:6, 3:6] = (sa * dt)**2 * np.eye(3)
        Q[6:9, 6:9] = (sw * dt)**2 * np.eye(3)
        Q[9:12, 9:12] = sab**2 * dt * np.eye(3)
        Q[12:15, 12:15] = swb**2 * dt * np.eye(3)

        # Covariance propagation
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)  # enforce symmetry

    def update_position_2d(self, z_xy: np.ndarray, R_xy: np.ndarray,
                           stamp_s: float) -> dict:
        """TERCOM 2D horizontal position measurement update.

        Args:
            z_xy: [x, y] measured position in local ENU (meters)
            R_xy: 2x2 measurement noise covariance (meters^2)
            stamp_s: Timestamp of TERCOM fix in seconds

        Returns:
            dict with 'innovation': [dx, dy], 'NIS': float, 'accepted': bool
        """
        if not self.initialized:
            return {'innovation': np.zeros(2), 'NIS': 0.0, 'accepted': False}

        # Measurement Jacobian: selects dp_x, dp_y from error state
        H = np.zeros((2, 15))
        H[0, 0] = 1.0  # dp_x
        H[1, 1] = 1.0  # dp_y

        # Innovation: measurement - prediction
        y = z_xy - self.state.position[:2]

        return self._apply_update(H, y, R_xy, stamp_s)

    def update_altitude(self, z_alt_msl: float, R_alt: float,
                        stamp_s: float) -> dict:
        """Barometric altitude measurement update.

        Args:
            z_alt_msl: Altitude above MSL in meters
            R_alt: Measurement noise variance (meters^2)
            stamp_s: Timestamp in seconds

        Returns:
            dict with 'innovation': float, 'NIS': float, 'accepted': bool
        """
        if not self.initialized:
            return {'innovation': np.zeros(1), 'NIS': 0.0, 'accepted': False}

        # Convert MSL altitude to local ENU z
        world_origin_alt = self.config.get('world_origin_alt', 0.0)
        z_local = z_alt_msl - world_origin_alt

        # Measurement Jacobian: selects dp_z
        H = np.zeros((1, 15))
        H[0, 2] = 1.0

        y = np.array([z_local - self.state.position[2]])
        R = np.array([[R_alt]])

        return self._apply_update(H, y, R, stamp_s)

    def update_velocity(self, z_vel: np.ndarray, R_vel: np.ndarray,
                        stamp_s: float) -> dict:
        """3D velocity measurement update.

        Args:
            z_vel: [vx, vy, vz] in local ENU (m/s)
            R_vel: 3x3 measurement noise covariance (m/s)^2
            stamp_s: Timestamp in seconds

        Returns:
            dict with 'innovation', 'NIS', 'accepted'
        """
        if not self.initialized:
            return {'innovation': np.zeros(3), 'NIS': 0.0, 'accepted': False}

        # Measurement Jacobian: selects dv
        H = np.zeros((3, 15))
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        H[2, 5] = 1.0

        y = z_vel - self.state.velocity

        return self._apply_update(H, y, R_vel, stamp_s)

    def _apply_update(self, H: np.ndarray, y: np.ndarray,
                      R: np.ndarray, stamp_s: float) -> dict:
        """Generic Kalman update with Joseph-form covariance.

        Args:
            H: Measurement Jacobian (m x 15)
            y: Innovation vector (m,)
            R: Measurement noise covariance (m x m)
            stamp_s: Timestamp in seconds

        Returns:
            dict with 'innovation', 'NIS', 'accepted'
        """
        # Innovation covariance
        S = H @ self.P @ H.T + R

        # NIS (Normalized Innovation Squared)
        try:
            nis = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            nis = float('inf')

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return {'innovation': y, 'NIS': nis, 'accepted': False}

        # Error state update
        self.dx = K @ y

        # Nominal state injection
        self.state.position += self.dx[0:3]
        self.state.velocity += self.dx[3:6]
        dtheta = self.dx[6:9]
        q_err = small_angle_quaternion(dtheta)
        self.state.quaternion = normalize_quaternion(
            quaternion_multiply(self.state.quaternion, q_err)
        )
        self.state.accel_bias += self.dx[9:12]
        self.state.gyro_bias += self.dx[12:15]

        # Joseph-form covariance update (numerically stable)
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # enforce symmetry

        # Reset error state
        self.dx = np.zeros(15)

        return {
            'innovation': y,
            'NIS': nis,
            'accepted': True,
            'S': S,
        }

    def get_position_std(self) -> np.ndarray:
        """Return [std_x, std_y, std_z] from covariance diagonal."""
        return np.sqrt(np.maximum(np.diag(self.P)[:3], 0.0))

    def reset_covariance(self) -> None:
        """Reset covariance to initial values and zero biases.

        Used for divergence recovery (soft reset). Keeps current position/velocity.
        """
        self.state.accel_bias = np.zeros(3)
        self.state.gyro_bias = np.zeros(3)
        self.dx = np.zeros(15)

        p = self.config.get('init_pos_std', 5.0)
        v = self.config.get('init_vel_std', 1.0)
        a = self.config.get('init_att_std', 0.05)
        ab = self.config.get('init_abias_std', 0.5)
        wb = self.config.get('init_wbias_std', 0.01)

        # Inflate position uncertainty more for divergence recovery
        self.P = np.diag(np.concatenate([
            [(p * 10)**2]*3, [v**2]*3, [a**2]*3, [ab**2]*3, [wb**2]*3
        ]))
