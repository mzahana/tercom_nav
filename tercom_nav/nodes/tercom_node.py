"""TERCOM Matcher Node.

Collects terrain elevation profiles from synchronized barometer + rangefinder
and performs vectorized TERCOM correlation matching against a DEM.

Topic remappings (applied in launch file):
  altitude        -> /target/mavros/altitude
  eskf_odom       -> /tercom/eskf_node/odom
  distance_sensor -> /scan  (sensor_msgs/LaserScan, single-beam lidar)
  imu_data        -> /target/mavros/imu/data
  eskf_covariance -> /tercom/eskf/pose

Publications:
  ~/position_fix    (geometry_msgs/PointStamped)    - UTM fix (x=E, y=N, z=elev)
  ~/match_quality   (std_msgs/Float32MultiArray)    - [MAD, discrim, rough, noise]
  ~/profile_path    (nav_msgs/Path)                 - profile sample positions
  ~/search_region   (visualization_msgs/Marker)     - search window wireframe
  ~/status          (std_msgs/String)               - COLLECTING|MATCHING|WAITING_SENSORS
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSPresetProfiles

import message_filters
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu, LaserScan
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger

# mavros_msgs - available in the Docker environment when workspace is sourced
from mavros_msgs.msg import Altitude

from tercom_nav.core.dem_manager import DEMManager
from tercom_nav.core.tercom_matcher import ProfileCollector, match_profile
from tercom_nav.core.adaptive_sampler import AdaptiveSampler
from tercom_nav.core.terrain_quality import compute_adaptive_noise
from tercom_nav.core.coordinate_utils import compute_utm_origin, local_enu_to_utm


class TERCOMNode(Node):
    def __init__(self):
        super().__init__('tercom_node')

        # Declare all parameters
        self.declare_parameter('dem_file', '')
        self.declare_parameter('profile_min_spacing_m', -1.0)
        self.declare_parameter('profile_max_samples', 20)
        self.declare_parameter('profile_min_distance_m', -1.0)
        self.declare_parameter('search_radius_pixels', 50)
        self.declare_parameter('search_radius_dynamic', True)
        self.declare_parameter('search_radius_sigma_mult', 3.0)
        self.declare_parameter('search_radius_min_pixels', 10)
        self.declare_parameter('search_radius_max_pixels', 100)
        self.declare_parameter('mad_reject_threshold', 30.0)
        self.declare_parameter('discrimination_min', 1.5)
        self.declare_parameter('discrimination_exclusion_radius', 3)
        self.declare_parameter('roughness_min', 3.0)
        self.declare_parameter('enable_adaptive_sampling', True)
        self.declare_parameter('adaptive_min_interval_s', 0.5)
        self.declare_parameter('adaptive_max_interval_s', 5.0)
        self.declare_parameter('adaptive_pixels_per_sample', 1.5)
        self.declare_parameter('sync_slop_s', 0.05)
        self.declare_parameter('world_origin_lat', 0.0)
        self.declare_parameter('world_origin_lon', 0.0)
        self.declare_parameter('world_origin_alt', 0.0)
        # Rangefinder mounting: True = gimbal keeps beam vertical (no trig needed);
        # False = body-fixed sensor tilts with the UAV (apply cos(roll)*cos(pitch)).
        self.declare_parameter('rangefinder_is_gimbaled', True)

        # Load DEM
        dem_file = self.get_parameter('dem_file').value
        if not dem_file:
            self.get_logger().error('"dem_file" parameter is required!')
            raise RuntimeError('dem_file parameter is required')

        self._dem = DEMManager(dem_file)
        self.get_logger().info(
            f'DEM loaded: pixel_size={self._dem.pixel_size_x:.2f}m, '
            f'{self._dem.width}x{self._dem.height} px'
        )

        # Compute UTM origin from world origin parameters
        lat0 = self.get_parameter('world_origin_lat').value
        lon0 = self.get_parameter('world_origin_lon').value
        alt0 = self.get_parameter('world_origin_alt').value
        if lat0 != 0.0 or lon0 != 0.0:
            self._origin = compute_utm_origin(lat0, lon0, alt0)
        else:
            # Fallback: use center of DEM as origin
            center_e = (self._dem.bounds['west'] + self._dem.bounds['east']) / 2
            center_n = (self._dem.bounds['south'] + self._dem.bounds['north']) / 2
            self._origin = {
                'easting': center_e, 'northing': center_n, 'alt': alt0,
            }
            self.get_logger().warning(
                'world_origin_lat/lon not set; using DEM center as ENU origin'
            )

        # Resolve auto-parameters from DEM pixel size
        px = self._dem.pixel_size_x
        min_spacing = self.get_parameter('profile_min_spacing_m').value
        if min_spacing < 0:
            min_spacing = 1.0 * px

        min_dist = self.get_parameter('profile_min_distance_m').value
        if min_dist < 0:
            min_dist = 15.0 * px

        max_samples = self.get_parameter('profile_max_samples').value
        self._profile_min_distance = min_dist

        # Adaptive sampler
        self._sampler = AdaptiveSampler(
            pixel_size_m=px,
            pixels_per_sample=self.get_parameter('adaptive_pixels_per_sample').value,
            min_interval_s=self.get_parameter('adaptive_min_interval_s').value,
            max_interval_s=self.get_parameter('adaptive_max_interval_s').value,
        )
        self._adaptive_enabled = self.get_parameter('enable_adaptive_sampling').value

        # Profile collector
        self._collector = ProfileCollector(
            min_spacing_m=min_spacing,
            max_samples=max_samples,
        )

        self._rangefinder_is_gimbaled = self.get_parameter('rangefinder_is_gimbaled').value

        # State variables
        self._latest_imu = None
        self._latest_odom = None
        self._eskf_covariance = None
        self._status = 'WAITING_SENSORS'
        self._match_count = 0
        self._reject_count = 0
        self._diag_logged = False  # one-shot sensor diagnostic

        # Publishers
        self._pub_fix = self.create_publisher(PointStamped, '~/position_fix', 10)
        self._pub_quality = self.create_publisher(Float32MultiArray, '~/match_quality', 10)
        self._pub_path = self.create_publisher(Path, '~/profile_path', 10)
        self._pub_search = self.create_publisher(Marker, '~/search_region', 10)
        self._pub_status = self.create_publisher(String, '~/status', 10)
        self._pub_rejected_fix = self.create_publisher(
            PointStamped, '~/rejected_fix', 10)
        self._pub_rejection_reason = self.create_publisher(
            String, '~/rejection_reason', 10)

        # Message filter synchronizer for altitude + rangefinder
        # MAVROS publishes with BEST_EFFORT reliability — match it.
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        sync_slop = self.get_parameter('sync_slop_s').value
        self._sub_alt = message_filters.Subscriber(self, Altitude, 'altitude',
                                                   qos_profile=sensor_qos)
        self._sub_range = message_filters.Subscriber(self, LaserScan, 'distance_sensor',
                                                     qos_profile=sensor_qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_alt, self._sub_range],
            queue_size=20,
            slop=sync_slop,
        )
        self._sync.registerCallback(self._cb_synced)

        # Other subscriptions
        self.create_subscription(Imu, 'imu_data', self._cb_imu, sensor_qos)
        self.create_subscription(Odometry, 'eskf_odom', self._cb_odom, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, 'eskf_covariance', self._cb_eskf_cov, 10
        )

        # Services
        self.create_service(Trigger, '~/trigger_match', self._srv_trigger_match)
        self.create_service(Trigger, '~/reset_profile', self._srv_reset_profile)

        # Status timer at 1 Hz
        self.create_timer(1.0, self._timer_status)

        self.get_logger().info('TERCOM node initialized and waiting for sensors')

    # ---- Sensor Callbacks ----

    def _cb_imu(self, msg: Imu):
        self._latest_imu = msg

    def _cb_odom(self, msg: Odometry):
        self._latest_odom = msg

    def _cb_eskf_cov(self, msg: PoseWithCovarianceStamped):
        self._eskf_covariance = msg

    def _cb_synced(self, alt_msg: Altitude, range_msg: LaserScan):
        """Called when barometer + rangefinder messages are time-synchronized."""
        if self._latest_imu is None or self._latest_odom is None:
            self._status = 'WAITING_SENSORS'
            return

        # Extract range from LaserScan (single-beam lidar: ranges[0])
        if not range_msg.ranges or not math.isfinite(range_msg.ranges[0]):
            return
        raw_range = range_msg.ranges[0]
        if raw_range < range_msg.range_min or raw_range > range_msg.range_max:
            return

        self._status = 'COLLECTING'

        # Compute h_AGL from rangefinder.
        # Gimbaled sensor: beam is always vertical — range equals vertical AGL directly.
        # Body-fixed sensor: beam tilts with the UAV — project slant range onto vertical
        #   axis using vehicle roll and pitch from the IMU: h_agl = range*cos(roll)*cos(pitch).
        if self._rangefinder_is_gimbaled:
            h_agl = raw_range
        else:
            q = self._latest_imu.orientation
            sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            pitch = math.asin(max(-1.0, min(1.0, sinp)))

            h_agl = raw_range * math.cos(roll) * math.cos(pitch)

        # Compute terrain elevation from synchronized measurements
        # h_terrain = h_baro_MSL - h_AGL
        h_terrain = alt_msg.amsl - h_agl

        if not self._diag_logged:
            self._diag_logged = True
            dem_elev_range = (self._dem.elevation[self._dem.elevation > -9000].min(),
                              self._dem.elevation[self._dem.elevation > -9000].max())
            pos = self._latest_odom.pose.pose.position
            self.get_logger().info(
                f'[DIAG] baro_amsl={alt_msg.amsl:.1f}m  raw_range={raw_range:.1f}m  '
                f'h_agl={h_agl:.1f}m  h_terrain={h_terrain:.1f}m  '
                f'DEM_elev=[{dem_elev_range[0]:.1f}, {dem_elev_range[1]:.1f}]m  '
                f'ENU_pos=({pos.x:.1f}, {pos.y:.1f})m'
            )

        # Current position in local ENU
        pos = self._latest_odom.pose.pose.position
        position_enu = np.array([pos.x, pos.y, pos.z])

        # Timestamp from synchronized messages (use altitude message stamp)
        ts_s = alt_msg.header.stamp.sec + alt_msg.header.stamp.nanosec * 1e-9

        # Decide whether to sample (adaptive or always)
        if self._adaptive_enabled:
            should = self._sampler.should_sample(position_enu, ts_s)
        else:
            should = True

        if not should:
            return

        # Record sample in adaptive sampler
        self._sampler.record_sample(position_enu, ts_s)

        # Add sample to profile collector
        profile_ready = self._collector.try_add_sample(h_terrain, position_enu, ts_s)

        if profile_ready:
            # Check minimum distance criterion
            if self._collector.total_distance_m >= self._profile_min_distance:
                self._run_matching(alt_msg.header)

    def _run_matching(self, trigger_header=None):
        """Perform TERCOM correlation matching with the current profile."""
        self._status = 'MATCHING'

        terrain_h, dx_m, dy_m, timestamps = self._collector.get_profile_arrays()
        if len(terrain_h) < 5:
            self.get_logger().warning('Not enough profile samples for matching (< 5)')
            return

        # Predicted UTM position (first sample position)
        if self._latest_odom is not None:
            pos = self._latest_odom.pose.pose.position
            pred_e, pred_n, _ = local_enu_to_utm(
                pos.x - dx_m[0], pos.y - dy_m[0], 0.0,
                self._origin['easting'], self._origin['northing'], 0.0,
            )
        else:
            # Use DEM center as fallback
            pred_e = (self._dem.bounds['west'] + self._dem.bounds['east']) / 2
            pred_n = (self._dem.bounds['south'] + self._dem.bounds['north']) / 2

        # Compute search radius
        search_radius = self._compute_search_radius()

        # Log predicted position vs DEM bounds on every match attempt
        b = self._dem.bounds
        in_bounds = (b['west'] <= pred_e <= b['east'] and b['south'] <= pred_n <= b['north'])
        self.get_logger().info(
            f'[MATCH] pred_utm=({pred_e:.0f}E, {pred_n:.0f}N)  '
            f'DEM_bounds=W{b["west"]:.0f} E{b["east"]:.0f} S{b["south"]:.0f} N{b["north"]:.0f}  '
            f'in_bounds={in_bounds}  search_r={search_radius}px  '
            f'terrain_h=[{min(terrain_h):.1f}, {max(terrain_h):.1f}]m  '
            f'n_samples={len(terrain_h)}'
        )

        # Run vectorized matching
        try:
            result = match_profile(
                dem_array=self._dem.elevation,
                transform=self._dem.transform,
                pixel_size_x=self._dem.pixel_size_x,
                pixel_size_y=self._dem.pixel_size_y,
                terrain_h=terrain_h,
                dx_m=dx_m,
                dy_m=dy_m,
                predicted_utm=(pred_e, pred_n),
                search_radius_px=search_radius,
                discrimination_exclusion_radius=self.get_parameter(
                    'discrimination_exclusion_radius').value,
            )
        except Exception as e:
            self.get_logger().error(f'match_profile failed: {e}')
            self._collector.slide_window()
            return

        # Apply quality thresholds
        mad_thresh = self.get_parameter('mad_reject_threshold').value
        disc_min = self.get_parameter('discrimination_min').value
        rough_min = self.get_parameter('roughness_min').value

        accepted = (
            result['valid'] and
            result['mad'] < mad_thresh and
            result['discrimination'] >= disc_min and
            result['roughness'] >= rough_min
        )

        if accepted:
            self._match_count += 1
            accept_rate = self._match_count / max(self._match_count + self._reject_count, 1)

            # Compute adaptive noise
            noise = compute_adaptive_noise(
                mad=result['mad'],
                discrimination=result['discrimination'],
                roughness=result['roughness'],
                pixel_size=self._dem.pixel_size_x,
                base_noise=-1.0,
            )

            # Publish position fix
            mid_ts_s = float(np.mean(timestamps))
            fix_msg = PointStamped()
            fix_msg.header.frame_id = 'utm'
            fix_msg.header.stamp.sec = int(mid_ts_s)
            fix_msg.header.stamp.nanosec = int((mid_ts_s % 1.0) * 1e9)
            fix_msg.point.x = result['utm'][0]  # UTM easting
            fix_msg.point.y = result['utm'][1]  # UTM northing
            fix_msg.point.z = float(np.mean(terrain_h))  # avg terrain elevation
            self._pub_fix.publish(fix_msg)

            # Publish quality metrics
            quality_msg = Float32MultiArray()
            quality_msg.data = [
                float(result['mad']),
                float(result['discrimination']),
                float(result['roughness']),
                float(noise),
            ]
            self._pub_quality.publish(quality_msg)

            # Publish profile path visualization (terrain_h in MSL → converted to ENU z)
            self._publish_profile_path(terrain_h, dx_m, dy_m, timestamps)

            self.get_logger().info(
                f'TERCOM match #{self._match_count}: '
                f'MAD={result["mad"]:.1f}m disc={result["discrimination"]:.2f} '
                f'rough={result["roughness"]:.1f}m '
                f'accept_rate={accept_rate:.0%}'
            )
        else:
            self._reject_count += 1
            reject_rate = self._reject_count / max(self._match_count + self._reject_count, 1)
            self.get_logger().warning(
                f'TERCOM match REJECTED (#{self._reject_count}, {reject_rate:.0%} rejection rate): '
                f'MAD={result["mad"]:.1f}m (thresh={mad_thresh}) '
                f'disc={result["discrimination"]:.2f} (min={disc_min}) '
                f'rough={result["roughness"]:.1f}m (min={rough_min})'
            )

            # Publish rejected fix location and reason (for diagnostics visualization)
            if result.get('valid', False) and result.get('utm') is not None:
                rej_fix = PointStamped()
                rej_fix.header.stamp = self.get_clock().now().to_msg()
                rej_fix.header.frame_id = 'utm'
                rej_fix.point.x = float(result['utm'][0])
                rej_fix.point.y = float(result['utm'][1])
                rej_fix.point.z = 0.0
                self._pub_rejected_fix.publish(rej_fix)

                # Determine primary rejection reason
                if result['mad'] >= mad_thresh:
                    reason = f'MAD={result["mad"]:.1f}m'
                elif result['discrimination'] < disc_min:
                    reason = f'disc={result["discrimination"]:.2f}'
                else:
                    reason = f'rough={result["roughness"]:.1f}m'
                rej_reason = String()
                rej_reason.data = reason
                self._pub_rejection_reason.publish(rej_reason)

        # Slide window for continuity
        self._collector.slide_window()
        self._status = 'COLLECTING'

    def _compute_search_radius(self) -> int:
        """Compute search radius in pixels, optionally from ESKF covariance."""
        min_px = self.get_parameter('search_radius_min_pixels').value
        max_px = self.get_parameter('search_radius_max_pixels').value
        static_r = self.get_parameter('search_radius_pixels').value

        if (self.get_parameter('search_radius_dynamic').value and
                self._eskf_covariance is not None):
            cov = self._eskf_covariance.pose.covariance
            std_x = math.sqrt(max(cov[0], 0.0))
            std_y = math.sqrt(max(cov[7], 0.0))
            sigma_pos = max(std_x, std_y)
            mult = self.get_parameter('search_radius_sigma_mult').value
            dynamic_r = int(mult * sigma_pos / self._dem.pixel_size_x)
            return max(min_px, min(max_px, dynamic_r))

        return max(min_px, min(max_px, static_r))

    def _publish_profile_path(self, terrain_h, dx_m, dy_m, timestamps):
        """Publish profile sample positions as a Path message.

        Points are placed at the DEM surface elevation so the path lies on the
        terrain in RViz2.  z is in local ENU (map frame): terrain_h_MSL - origin_alt.
        """
        if self._latest_odom is None:
            return

        path_msg = Path()
        path_msg.header.frame_id = 'map'
        if timestamps is not None and len(timestamps) > 0:
            mid_ts = float(np.mean(timestamps))
            path_msg.header.stamp.sec = int(mid_ts)
            path_msg.header.stamp.nanosec = int((mid_ts % 1.0) * 1e9)

        base_pos = self._latest_odom.pose.pose.position
        base_x = base_pos.x - dx_m[-1]  # walk back to profile start
        base_y = base_pos.y - dy_m[-1]
        origin_alt = self._origin.get('alt', 0.0)

        for h, dx, dy, ts in zip(terrain_h, dx_m, dy_m, timestamps):
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp.sec = int(ts)
            ps.header.stamp.nanosec = int((ts % 1.0) * 1e9)
            ps.pose.position.x = base_x + dx
            ps.pose.position.y = base_y + dy
            ps.pose.position.z = float(h) - origin_alt   # MSL → ENU z
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self._pub_path.publish(path_msg)

    # ---- Services ----

    def _srv_trigger_match(self, request, response):
        """Force a TERCOM match with current samples (minimum 5 required)."""
        if self._collector.num_samples < 5:
            response.success = False
            response.message = (
                f'Not enough samples: {self._collector.num_samples} < 5'
            )
            return response

        self.get_logger().info('Manual match triggered via service')
        self._run_matching()
        response.success = True
        response.message = (
            f'Match triggered with {self._collector.num_samples} samples. '
            f'Total accepted: {self._match_count}'
        )
        return response

    def _srv_reset_profile(self, request, response):
        """Clear all collected profile samples."""
        n = self._collector.num_samples
        self._collector.reset()
        self._sampler.reset()
        self._status = 'COLLECTING'
        response.success = True
        response.message = f'Profile reset. Cleared {n} samples.'
        self.get_logger().info(f'Profile reset by service call (had {n} samples)')
        return response

    # ---- Status Timer ----

    def _timer_status(self):
        msg = String()
        if self._status == 'COLLECTING':
            n = self._collector.num_samples
            m = int(self.get_parameter('profile_max_samples').value)
            msg.data = f'COLLECTING {n}/{m}'
        else:
            msg.data = self._status
        self._pub_status.publish(msg)
        total = self._match_count + self._reject_count
        if self._status == 'COLLECTING':
            min_dist = self._profile_min_distance
            self.get_logger().info(
                f'TERCOM: COLLECTING [{self._collector.num_samples}/'
                f'{self.get_parameter("profile_max_samples").value} samples, '
                f'{self._collector.total_distance_m:.0f}/{min_dist:.0f}m] '
                f'matches={self._match_count} rejected={self._reject_count}'
            )
        elif self._status == 'WAITING_SENSORS':
            self.get_logger().warning(
                'TERCOM: WAITING_SENSORS — check imu_data, altitude, '
                'distance_sensor, and local_odom topics'
            )
        elif total > 0:
            accept_rate = self._match_count / total
            self.get_logger().info(
                f'TERCOM: {self._status} | '
                f'matches={self._match_count} rejected={self._reject_count} '
                f'accept_rate={accept_rate:.0%}'
            )


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TERCOMNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        print(f'TERCOM node error: {e}')
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
