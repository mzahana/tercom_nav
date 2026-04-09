"""ESKF Navigation Filter Node.

State machine: WAITING_GPS -> INITIALIZING -> RUNNING <-> DIVERGED

GPS is used ONCE for initialization, then disabled.
IMU drives prediction at imu_rate_hz (decimated from raw ~250 Hz).
TERCOM fixes, barometric altitude, and velocity updates are applied
when available, rate-limited to their respective parameters.

Topic remappings (set in launch file):
  imu_data       -> /target/mavros/imu/data
  gps_global     -> /target/mavros/global_position/global
  altitude       -> /target/mavros/altitude
  velocity_local -> /target/mavros/local_position/velocity_local
  local_odom     -> /target/mavros/local_position/odom
  tercom_fix     -> /tercom/position_fix
  tercom_quality -> /tercom/match_quality

Publications:
  ~/odom         (nav_msgs/Odometry)                 at imu_rate_hz
  ~/pose         (geometry_msgs/PoseWithCovStamped)  at imu_rate_hz
  ~/global       (sensor_msgs/NavSatFix)             at 1 Hz
  ~/bias_accel   (geometry_msgs/Vector3Stamped)      at 1 Hz
  ~/bias_gyro    (geometry_msgs/Vector3Stamped)      at 1 Hz
  ~/state        (std_msgs/String)                   on change
  ~/health       (std_msgs/Float32MultiArray)        at 1 Hz
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import (
    Vector3Stamped, PoseWithCovarianceStamped, PointStamped, TwistStamped
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix
from std_srvs.srv import Trigger

from mavros_msgs.msg import Altitude

from tercom_nav.core.eskf import ESKF
from tercom_nav.core.health_monitor import HealthMonitor
from tercom_nav.core.coordinate_utils import (
    compute_utm_origin, utm_to_local_enu, local_enu_to_utm, utm_to_latlon, latlon_to_utm
)


class ESKFNode(Node):

    # State machine states
    STATE_WAITING_GPS = 'WAITING_GPS'
    STATE_INITIALIZING = 'INITIALIZING'
    STATE_RUNNING = 'RUNNING'
    STATE_DIVERGED = 'DIVERGED'
    STATE_RESETTING = 'RESETTING'

    def __init__(self):
        super().__init__('eskf_node')

        # Declare all parameters
        self.declare_parameter('imu_rate_hz', 50.0)
        self.declare_parameter('accel_noise', 0.1)
        self.declare_parameter('gyro_noise', 0.01)
        self.declare_parameter('accel_bias_noise', 0.001)
        self.declare_parameter('gyro_bias_noise', 0.0001)
        self.declare_parameter('bias_time_constant', 300.0)
        self.declare_parameter('tercom_noise_base', -1.0)
        self.declare_parameter('baro_noise', 3.0)
        self.declare_parameter('baro_update_rate_hz', 1.0)
        self.declare_parameter('velocity_noise', 0.5)
        self.declare_parameter('enable_velocity_updates', True)
        self.declare_parameter('velocity_update_rate_hz', 5.0)
        self.declare_parameter('init_pos_std', 5.0)
        self.declare_parameter('init_vel_std', 1.0)
        self.declare_parameter('init_att_std', 0.05)
        self.declare_parameter('init_abias_std', 0.5)
        self.declare_parameter('init_wbias_std', 0.01)
        self.declare_parameter('gps_init_samples', 20)
        self.declare_parameter('gps_init_timeout_s', 30.0)
        self.declare_parameter('world_origin_lat', 0.0)
        self.declare_parameter('world_origin_lon', 0.0)
        self.declare_parameter('world_origin_alt', 0.0)
        self.declare_parameter('nis_threshold', 15.0)
        self.declare_parameter('nis_window_size', 10)
        self.declare_parameter('max_position_std', 500.0)
        self.declare_parameter('max_innovation_m', 200.0)
        self.declare_parameter('consecutive_reject_limit', 5)
        self.declare_parameter('divergence_action', 'reset')

        # World origin
        lat0 = self.get_parameter('world_origin_lat').value
        lon0 = self.get_parameter('world_origin_lon').value
        alt0 = self.get_parameter('world_origin_alt').value
        if lat0 != 0.0 or lon0 != 0.0:
            self._origin = compute_utm_origin(lat0, lon0, alt0)
        else:
            self._origin = {'easting': 0.0, 'northing': 0.0, 'alt': alt0,
                            'zone_number': 37, 'zone_letter': 'N', 'epsg': 32637}
            self.get_logger().warning('world_origin_lat/lon not set. UTM conversion may be incorrect.')

        # ESKF config dict
        eskf_config = {
            'imu_rate_hz': self.get_parameter('imu_rate_hz').value,
            'accel_noise': self.get_parameter('accel_noise').value,
            'gyro_noise': self.get_parameter('gyro_noise').value,
            'accel_bias_noise': self.get_parameter('accel_bias_noise').value,
            'gyro_bias_noise': self.get_parameter('gyro_bias_noise').value,
            'bias_time_constant': self.get_parameter('bias_time_constant').value,
            'init_pos_std': self.get_parameter('init_pos_std').value,
            'init_vel_std': self.get_parameter('init_vel_std').value,
            'init_att_std': self.get_parameter('init_att_std').value,
            'init_abias_std': self.get_parameter('init_abias_std').value,
            'init_wbias_std': self.get_parameter('init_wbias_std').value,
            'world_origin_alt': alt0,
        }

        self._eskf = ESKF(eskf_config)
        self._health_monitor = HealthMonitor(
            nis_threshold=self.get_parameter('nis_threshold').value,
            nis_window=self.get_parameter('nis_window_size').value,
            max_position_std=self.get_parameter('max_position_std').value,
            max_innovation=self.get_parameter('max_innovation_m').value,
            consecutive_reject_limit=self.get_parameter('consecutive_reject_limit').value,
        )

        # State machine
        self._state = self.STATE_WAITING_GPS
        self._gps_samples = []  # (lat, lon, alt, vx, vy, vz, quat)
        self._init_odom = None
        self._gps_init_start_s = None
        self._last_innov_norm = 0.0   # last TERCOM innovation ||z - Hx|| in metres

        # IMU decimation
        imu_rate = self.get_parameter('imu_rate_hz').value
        raw_imu_rate = 250.0  # PX4 SITL IMU rate
        self._imu_decimation = max(1, round(raw_imu_rate / imu_rate))
        self._imu_counter = 0

        # Rate limiting for baro/velocity updates
        self._last_baro_update_s = 0.0
        self._last_vel_update_s = 0.0
        self._baro_interval_s = 1.0 / max(self.get_parameter('baro_update_rate_hz').value, 0.01)
        self._vel_interval_s = 1.0 / max(self.get_parameter('velocity_update_rate_hz').value, 0.01)

        # Latest tercom quality (for adaptive noise)
        self._last_tercom_quality = [0.0, 1.5, 5.0, 0.0]

        # Publishers
        self._pub_odom = self.create_publisher(Odometry, '~/odom', 10)
        self._pub_pose = self.create_publisher(PoseWithCovarianceStamped, '~/pose', 10)
        self._pub_global = self.create_publisher(NavSatFix, '~/global', 10)
        self._pub_bias_a = self.create_publisher(Vector3Stamped, '~/bias_accel', 10)
        self._pub_bias_w = self.create_publisher(Vector3Stamped, '~/bias_gyro', 10)
        self._pub_state = self.create_publisher(String, '~/state', 10)
        self._pub_health = self.create_publisher(Float32MultiArray, '~/health', 10)

        # Subscriptions — MAVROS publishes with BEST_EFFORT; use SENSOR_DATA QoS to match.
        # Internal topics (tercom_fix, tercom_quality) are from our own nodes (RELIABLE).
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Imu, 'imu_data', self._cb_imu, sensor_qos)
        self.create_subscription(NavSatFix, 'gps_global', self._cb_gps, sensor_qos)
        self.create_subscription(Altitude, 'altitude', self._cb_altitude, sensor_qos)
        self.create_subscription(TwistStamped, 'velocity_local', self._cb_velocity, sensor_qos)
        self.create_subscription(Odometry, 'local_odom', self._cb_odom, sensor_qos)
        self.create_subscription(PointStamped, 'tercom_fix', self._cb_tercom_fix, 10)
        self.create_subscription(Float32MultiArray, 'tercom_quality', self._cb_tercom_quality, 10)

        # Services
        self.create_service(Trigger, '~/reset_filter', self._srv_reset)

        # 1 Hz timer for slow publishers and health
        self.create_timer(1.0, self._timer_1hz)

        # GPS init timeout timer
        self.create_timer(5.0, self._check_gps_timeout)

        self._publish_state(self.STATE_WAITING_GPS)
        self.get_logger().info(
            f'ESKF node started. IMU decimation: 1/{self._imu_decimation} '
            f'(~{250/self._imu_decimation:.0f} Hz)'
        )

    # ---- State Machine Helpers ----

    def _publish_state(self, new_state: str):
        if new_state != self._state:
            self.get_logger().info(f'ESKF state: {self._state} -> {new_state}')
        self._state = new_state
        msg = String()
        msg.data = new_state
        self._pub_state.publish(msg)

    # ---- GPS Initialization ----

    def _cb_gps(self, msg: NavSatFix):
        """GPS callback - only used during WAITING_GPS and INITIALIZING states."""
        if self._state not in (self.STATE_WAITING_GPS, self.STATE_INITIALIZING):
            return

        # Check fix quality
        if msg.status.status < 0:  # NavSatStatus.STATUS_NO_FIX = -1
            return

        if self._state == self.STATE_WAITING_GPS:
            self._state = self.STATE_INITIALIZING
            self._gps_init_start_s = (
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            )
            self._publish_state(self.STATE_INITIALIZING)
            self.get_logger().info(
                f'GPS fix acquired at ({msg.latitude:.6f}, {msg.longitude:.6f}). '
                f'Collecting {self.get_parameter("gps_init_samples").value} samples...'
            )

        # Collect GPS samples for averaging
        if self._state == self.STATE_INITIALIZING:
            sample = {
                'lat': msg.latitude, 'lon': msg.longitude,
                'alt': msg.altitude,
            }
            self._gps_samples.append(sample)

            required = self.get_parameter('gps_init_samples').value
            if len(self._gps_samples) >= required:
                self._finalize_initialization(msg.header.stamp)

    def _finalize_initialization(self, stamp):
        """Average GPS samples and initialize the ESKF."""
        avg_lat = np.mean([s['lat'] for s in self._gps_samples])
        avg_lon = np.mean([s['lon'] for s in self._gps_samples])
        avg_alt = np.mean([s['alt'] for s in self._gps_samples])

        # Convert GPS to local ENU
        easting, northing, _, _ = latlon_to_utm(avg_lat, avg_lon)
        pos_enu = utm_to_local_enu(
            easting, northing, avg_alt,
            self._origin['easting'], self._origin['northing'], self._origin['alt'],
        )

        # Initial velocity from latest odometry
        vel_enu = np.zeros(3)
        if self._init_odom is not None:
            v = self._init_odom.twist.twist.linear
            vel_enu = np.array([v.x, v.y, v.z])

        # Identity quaternion (level flight assumption)
        quat = np.array([1.0, 0.0, 0.0, 0.0])

        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        self._eskf.initialize(pos_enu, vel_enu, quat, stamp_s)

        self._health_monitor.reset()
        self._publish_state(self.STATE_RUNNING)
        self.get_logger().info(
            f'ESKF initialized: pos=[{pos_enu[0]:.1f}, {pos_enu[1]:.1f}, {pos_enu[2]:.1f}] m ENU, '
            f'from {len(self._gps_samples)} GPS samples'
        )

    def _cb_odom(self, msg: Odometry):
        """Store latest odometry for initialization velocity."""
        self._init_odom = msg

    # ---- IMU Prediction ----

    def _cb_imu(self, msg: Imu):
        """IMU callback - drives ESKF prediction at decimated rate."""
        if self._state != self.STATE_RUNNING:
            return

        self._imu_counter += 1
        if self._imu_counter % self._imu_decimation != 0:
            return

        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self._eskf.predict(accel, gyro, stamp_s)

        # Publish odometry and pose at IMU rate
        self._publish_odom(msg.header)

    # ---- Sensor Updates ----

    def _cb_altitude(self, msg: Altitude):
        """Barometric altitude update - rate-limited."""
        if self._state != self.STATE_RUNNING:
            return

        ts_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if ts_s - self._last_baro_update_s < self._baro_interval_s:
            return
        self._last_baro_update_s = ts_s

        R_baro = self.get_parameter('baro_noise').value ** 2
        self._eskf.update_altitude(msg.amsl, R_baro, ts_s)

    def _cb_velocity(self, msg: TwistStamped):
        """Velocity aiding update - rate-limited."""
        if self._state != self.STATE_RUNNING:
            return
        if not self.get_parameter('enable_velocity_updates').value:
            return

        ts_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if ts_s - self._last_vel_update_s < self._vel_interval_s:
            return
        self._last_vel_update_s = ts_s

        sv = self.get_parameter('velocity_noise').value
        R_vel = np.eye(3) * sv**2
        z_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
        ])
        self._eskf.update_velocity(z_vel, R_vel, ts_s)

    def _cb_tercom_quality(self, msg: Float32MultiArray):
        self._last_tercom_quality = list(msg.data)

    def _cb_tercom_fix(self, msg: PointStamped):
        """TERCOM position fix - apply as 2D position update."""
        if self._state != self.STATE_RUNNING:
            return

        ts_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Convert UTM fix to local ENU
        utm_e = msg.point.x
        utm_n = msg.point.y
        pos_enu = utm_to_local_enu(
            utm_e, utm_n, 0.0,
            self._origin['easting'], self._origin['northing'], self._origin['alt'],
        )
        z_xy = pos_enu[:2]

        # Compute adaptive measurement noise from quality metrics
        tercom_noise_base = self.get_parameter('tercom_noise_base').value
        if tercom_noise_base < 0:
            tercom_noise_base = 5.93  # fallback; ideally from DEM info

        q = self._last_tercom_quality
        mad = q[0] if len(q) > 0 else 5.0
        disc = q[1] if len(q) > 1 else 1.5
        rough = q[2] if len(q) > 2 else 5.0

        # Adaptive noise: higher noise for poor matches
        from tercom_nav.core.terrain_quality import compute_adaptive_noise
        noise_var = compute_adaptive_noise(mad, disc, rough, tercom_noise_base, tercom_noise_base)
        R_xy = np.eye(2) * noise_var

        # Check innovation before applying
        innovation = z_xy - self._eskf.state.position[:2]
        H = np.zeros((2, 15))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        S = H @ self._eskf.P @ H.T + R_xy

        health_result = self._health_monitor.check_innovation(innovation, S)
        self._last_innov_norm = float(np.linalg.norm(innovation))

        if health_result['accept']:
            self._eskf.update_position_2d(z_xy, R_xy, ts_s)

            self.get_logger().debug(
                f'TERCOM update: innovation=[{innovation[0]:.1f}, {innovation[1]:.1f}] m, '
                f'NIS={health_result["nis"]:.2f}'
            )
        else:
            self.get_logger().warning(
                f'TERCOM fix GATED: ||innovation||={np.linalg.norm(innovation):.1f} m > '
                f'{self.get_parameter("max_innovation_m").value} m'
            )

        # Check divergence after update
        cov_result = self._health_monitor.check_covariance(self._eskf.P)
        diverged, reason = self._health_monitor.is_diverged()
        if not cov_result['healthy']:
            diverged = True
            reason = f'Covariance too large: max_std={cov_result["max_pos_std"]:.1f} m'

        if diverged:
            self._handle_divergence(reason)

    # ---- Divergence Recovery ----

    def _handle_divergence(self, reason: str):
        action = self.get_parameter('divergence_action').value
        self.get_logger().error(f'Filter DIVERGED: {reason}. Action: {action}')
        self._publish_state(self.STATE_DIVERGED)

        if action == 'warn':
            pass  # Stay in RUNNING, just log

        elif action == 'reset':
            self._publish_state(self.STATE_RESETTING)
            self._eskf.reset_covariance()
            self._health_monitor.reset()
            self.get_logger().info('Filter soft-reset: covariance inflated, biases zeroed')
            self._publish_state(self.STATE_RUNNING)

        elif action == 'reset_with_gps':
            self._publish_state(self.STATE_RESETTING)
            self._gps_samples.clear()
            self._health_monitor.reset()
            self._eskf.initialized = False
            self._publish_state(self.STATE_WAITING_GPS)
            self.get_logger().info('Filter hard-reset: re-acquiring GPS')

    # ---- Publishing ----

    def _publish_odom(self, header):
        """Publish Odometry and PoseWithCovarianceStamped at IMU rate."""
        if not self._eskf.initialized:
            return

        s = self._eskf.state
        P = self._eskf.P

        # Odometry
        odom = Odometry()
        odom.header = header
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = s.position[0]
        odom.pose.pose.position.y = s.position[1]
        odom.pose.pose.position.z = s.position[2]
        odom.pose.pose.orientation.w = s.quaternion[0]
        odom.pose.pose.orientation.x = s.quaternion[1]
        odom.pose.pose.orientation.y = s.quaternion[2]
        odom.pose.pose.orientation.z = s.quaternion[3]
        odom.twist.twist.linear.x = s.velocity[0]
        odom.twist.twist.linear.y = s.velocity[1]
        odom.twist.twist.linear.z = s.velocity[2]

        # 6x6 pose covariance (row-major, pos+rot)
        cov6 = np.zeros((6, 6))
        cov6[:3, :3] = P[:3, :3]     # position covariance
        cov6[3:, 3:] = P[6:9, 6:9]  # orientation covariance (approximate)
        odom.pose.covariance = cov6.flatten().tolist()

        self._pub_odom.publish(odom)

        # PoseWithCovariance (for TERCOM dynamic search radius)
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = odom.header
        pose_msg.pose = odom.pose
        self._pub_pose.publish(pose_msg)

    def _timer_1hz(self):
        """1 Hz slow publishers: global position, biases, health."""
        if not self._eskf.initialized:
            return

        s = self._eskf.state
        stamp = self.get_clock().now().to_msg()

        # Global position (lat/lon)
        try:
            e, n, _ = local_enu_to_utm(
                s.position[0], s.position[1], s.position[2],
                self._origin['easting'], self._origin['northing'], self._origin['alt'],
            )
            lat, lon = utm_to_latlon(e, n,
                                     self._origin.get('zone_number', 37),
                                     self._origin.get('zone_letter', 'N'))
            alt_msl = self._origin['alt'] + s.position[2]

            global_msg = NavSatFix()
            global_msg.header.stamp = stamp
            global_msg.header.frame_id = 'map'
            global_msg.latitude = lat
            global_msg.longitude = lon
            global_msg.altitude = alt_msl
            global_msg.status.status = 0  # STATUS_FIX
            self._pub_global.publish(global_msg)
        except Exception as e:
            self.get_logger().debug(f'Global publish failed: {e}')

        # Accelerometer bias
        ba_msg = Vector3Stamped()
        ba_msg.header.stamp = stamp
        ba_msg.vector.x = s.accel_bias[0]
        ba_msg.vector.y = s.accel_bias[1]
        ba_msg.vector.z = s.accel_bias[2]
        self._pub_bias_a.publish(ba_msg)

        # Gyroscope bias
        bw_msg = Vector3Stamped()
        bw_msg.header.stamp = stamp
        bw_msg.vector.x = s.gyro_bias[0]
        bw_msg.vector.y = s.gyro_bias[1]
        bw_msg.vector.z = s.gyro_bias[2]
        self._pub_bias_w.publish(bw_msg)

        # Health metrics
        pos_std = self._eskf.get_position_std()
        avg_nis = self._health_monitor.get_avg_nis()
        innov_norm = self._last_innov_norm
        is_healthy = 1.0 if self._state == self.STATE_RUNNING else 0.0

        health_msg = Float32MultiArray()
        health_msg.data = [
            float(avg_nis),
            float(np.max(pos_std)),
            float(innov_norm),
            float(is_healthy),
        ]
        self._pub_health.publish(health_msg)

    def _check_gps_timeout(self):
        """Check if GPS initialization has timed out."""
        if self._state != self.STATE_INITIALIZING:
            return
        if self._gps_init_start_s is None:
            return

        timeout = self.get_parameter('gps_init_timeout_s').value
        now_s = self.get_clock().now().nanoseconds / 1e9
        elapsed = now_s - self._gps_init_start_s

        if elapsed > timeout:
            n = len(self._gps_samples)
            if n >= 3:
                self.get_logger().warning(
                    f'GPS init timeout ({timeout}s). Using {n} samples collected so far.'
                )
                stamp = self.get_clock().now().to_msg()
                self._finalize_initialization(stamp)
            else:
                self.get_logger().error(
                    f'GPS init timeout with only {n} samples. Retrying...'
                )
                self._gps_samples.clear()
                self._publish_state(self.STATE_WAITING_GPS)

    # ---- Service ----

    def _srv_reset(self, request, response):
        """Force filter reset back to GPS initialization."""
        self.get_logger().info('Filter reset requested via service')
        self._gps_samples.clear()
        self._health_monitor.reset()
        self._eskf.initialized = False
        self._publish_state(self.STATE_WAITING_GPS)
        response.success = True
        response.message = 'Filter reset. Re-acquiring GPS...'
        return response


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ESKFNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        print(f'ESKF node error: {e}')
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
