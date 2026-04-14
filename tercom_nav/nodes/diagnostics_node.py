"""Diagnostics Node - evaluation and visualization for TERCOM navigation.

Computes position errors against ground truth (PX4 SITL via MAVROS).
Publishes RViz-ready markers, paths, and DEM point cloud.
Optionally logs everything to CSV for post-flight analysis.

Topic remappings (applied in launch file):
  eskf_odom         -> /tercom/eskf/odom
  ground_truth_odom -> /target/mavros/local_position/odom
  ground_truth_global -> /target/mavros/global_position/global
  tercom_fix        -> /tercom/position_fix
  tercom_quality    -> /tercom/match_quality
  eskf_state        -> /tercom/eskf/state
  eskf_health       -> /tercom/eskf/health
"""
import os
import csv
import math
import json
import collections
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from tercom_nav.core.timing import ComponentTimer
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, QoSPresetProfiles

from std_msgs.msg import String, Float32MultiArray, Float64
from std_msgs.msg import String as StringMsg
from geometry_msgs.msg import Vector3Stamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2, NavSatFix
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from tercom_nav.core.terrain_quality import classify_terrain_quality
from tercom_nav.core.coordinate_utils import compute_utm_origin, utm_to_local_enu


class DiagnosticsNode(Node):
    def __init__(self):
        super().__init__('diagnostics_node')

        # Declare all parameters
        self.declare_parameter('log_to_csv', True)
        self.declare_parameter('csv_path', '/tmp/tercom_logs/')
        self.declare_parameter('publish_dem_pointcloud', True)
        self.declare_parameter('dem_pointcloud_decimation', 4)
        self.declare_parameter('dem_file', '')
        self.declare_parameter('error_publish_rate_hz', 10.0)
        self.declare_parameter('path_publish_rate_hz', 2.0)
        self.declare_parameter('world_origin_lat', 0.0)
        self.declare_parameter('world_origin_lon', 0.0)
        self.declare_parameter('world_origin_alt', 0.0)

        # UTM origin for coordinate conversion (used to place markers in local ENU frame)
        _origin = compute_utm_origin(
            self.get_parameter('world_origin_lat').value,
            self.get_parameter('world_origin_lon').value,
            self.get_parameter('world_origin_alt').value,
        )
        self._utm_origin_e   = _origin['easting']
        self._utm_origin_n   = _origin['northing']
        self._utm_origin_alt = _origin['alt']

        # State
        self._est_odom = None
        self._gt_odom = None
        self._gt_global = None
        self._last_tercom_quality = [0.0, 1.0, 0.0, 0.0]  # [MAD, disc, rough, noise]
        self._filter_state = 'UNKNOWN'
        self._health = [0.0, 0.0, 0.0, 0.0]

        self._est_path = Path()
        self._gt_path = Path()
        self._est_path.header.frame_id = 'map'
        self._gt_path.header.frame_id = 'map'

        # Frame alignment: MAVROS local_position/odom uses PX4's NED origin (wherever
        # the drone booted), while ESKF uses our UTM-derived ENU origin. They are
        # different frames and cannot be compared directly. We align them once on the
        # first sample where both are available — storing the constant offset to add to
        # every ground truth position before computing errors.
        self._frame_offset = None      # np.array([dx, dy, dz]) to add to gt position
        self._pending_alignment = False  # set True when filter enters RUNNING

        self._h_errors = []      # horizontal error history
        self._accepted_count = 0
        self._tercom_marker_id = 0
        self._tercom_markers = MarkerArray()

        # Rolling history for new diagnostic publishers
        self._nis_deque = collections.deque(maxlen=60)       # 1 Hz → 60 s window
        self._err_history = collections.deque(maxlen=120)    # 2 Hz → 60 s window
        self._est_path_errors: list = []                     # error per est_path pose
        self._rejected_marker_id = 0
        self._rejected_markers = MarkerArray()
        self._last_rejection_reason = ''
        self._pending_rejected_fix = None

        # CSV setup
        self._csv_file = None
        self._csv_writer = None
        self._csv_t0_ns = None
        if self.get_parameter('log_to_csv').value:
            self._setup_csv()

        # Publishers
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._pub_pos_err = self.create_publisher(Vector3Stamped, '~/position_error', 10)
        self._pub_err_norm = self.create_publisher(Float64, '~/error_norm', 10)
        self._pub_err_stats = self.create_publisher(Float32MultiArray, '~/error_stats', 10)
        self._pub_est_path = self.create_publisher(Path, '~/estimated_path', 10)
        self._pub_gt_path = self.create_publisher(Path, '~/ground_truth_path', 10)
        self._pub_tc_fixes = self.create_publisher(MarkerArray, '~/tercom_fixes_viz', 10)
        self._pub_cov_ellipse = self.create_publisher(Marker, '~/covariance_ellipse', 10)
        self._pub_dem_cloud = self.create_publisher(PointCloud2, '~/dem_surface', latched_qos)
        self._pub_error_arrow      = self.create_publisher(MarkerArray,        '~/error_arrow', 10)
        self._pub_error_col_path   = self.create_publisher(MarkerArray,        '~/error_colored_path', 10)
        self._pub_rejected_viz     = self.create_publisher(MarkerArray,        '~/rejected_fixes_viz', 10)
        self._pub_nis_history      = self.create_publisher(Float32MultiArray,  '~/nis_history', 10)
        self._pub_err_history_chart= self.create_publisher(Float32MultiArray,  '~/error_history_chart', 10)
        # Profiling: 16 floats — [exec_ms, hz] × 8 components across all 3 nodes
        self._pub_profiling = self.create_publisher(Float32MultiArray, '~/profiling', 10)

        # Cached timing arrays received from other nodes
        self._tercom_timing = [0.0] * 4  # [synced_ms, synced_hz, match_ms, match_hz]
        self._eskf_timing   = [0.0] * 6  # [imu_ms, imu_hz, tc_ms, tc_hz, baro_ms, baro_hz]

        # Own component timers
        self._t_timer_error = ComponentTimer()
        self._t_timer_paths = ComponentTimer()
        self._t_timer_stats = ComponentTimer()

        # Subscriptions — MAVROS publishes with BEST_EFFORT; use SENSOR_DATA QoS to match.
        # Internal topics (eskf_odom, tercom_fix/quality, eskf_state/health) are RELIABLE.
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Odometry, 'eskf_odom', self._cb_eskf_odom, 10)
        self.create_subscription(Odometry, 'ground_truth_odom', self._cb_gt_odom, sensor_qos)
        self.create_subscription(NavSatFix, 'ground_truth_global', self._cb_gt_global, sensor_qos)
        self.create_subscription(PointStamped, 'tercom_fix', self._cb_tercom_fix, 10)
        self.create_subscription(Float32MultiArray, 'tercom_quality', self._cb_tercom_quality, 10)
        self.create_subscription(String, 'eskf_state', self._cb_eskf_state, 10)
        self.create_subscription(Float32MultiArray, 'eskf_health', self._cb_eskf_health, 10)
        self.create_subscription(PointStamped, 'rejected_fix',      self._cb_rejected_fix, 10)
        self.create_subscription(StringMsg,    'rejection_reason',  self._cb_rejection_reason, 10)
        self.create_subscription(Float32MultiArray, 'tercom_timing', self._cb_tercom_timing, 10)
        self.create_subscription(Float32MultiArray, 'eskf_timing',   self._cb_eskf_timing, 10)

        # Timers
        err_rate = self.get_parameter('error_publish_rate_hz').value
        path_rate = self.get_parameter('path_publish_rate_hz').value
        self.create_timer(1.0 / err_rate, self._timer_error)
        self.create_timer(1.0 / path_rate, self._timer_paths)
        self.create_timer(1.0, self._timer_stats)

        # Publish DEM point cloud once at startup
        if self.get_parameter('publish_dem_pointcloud').value:
            dem_file = self.get_parameter('dem_file').value
            if dem_file:
                self.create_timer(2.0, self._publish_dem_pointcloud_once)

        self.get_logger().info('Diagnostics node started')

    # ---- Subscriptions ----

    def _cb_eskf_odom(self, msg: Odometry):
        self._est_odom = msg
        # Perform deferred frame alignment on the first fresh odom after entering RUNNING.
        # _align_frames() can't run from _cb_eskf_state because eskf_odom is only
        # published during RUNNING — so self._est_odom would be None or stale at that moment.
        if self._pending_alignment:
            self._pending_alignment = False
            self._align_frames()

        # Append to estimated path
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self._est_path.poses.append(ps)
        self._est_path.header.stamp = msg.header.stamp

        # Track error per pose (used for error-colored path)
        if self._gt_odom is not None and self._frame_offset is not None:
            gt = self._gt_odom.pose.pose.position
            ep = msg.pose.pose.position
            dx = ep.x - (gt.x + self._frame_offset[0])
            dy = ep.y - (gt.y + self._frame_offset[1])
            self._est_path_errors.append(math.hypot(dx, dy))
        else:
            self._est_path_errors.append(0.0)
        # Limit path history to last 2000 poses
        if len(self._est_path.poses) > 2000:
            self._est_path.poses = self._est_path.poses[-2000:]
            self._est_path_errors = self._est_path_errors[-2000:]

        # CSV logging
        if self._csv_writer and self._gt_odom:
            self._write_csv_row(msg)

    def _cb_gt_odom(self, msg: Odometry):
        self._gt_odom = msg
        # Store poses in raw MAVROS frame always. Offset is applied at publish time
        # so that frame re-alignment (on filter re-init) never double-shifts history.
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self._gt_path.poses.append(ps)
        self._gt_path.header.stamp = msg.header.stamp

    def _cb_gt_global(self, msg: NavSatFix):
        self._gt_global = msg

    def _cb_tercom_fix(self, msg: PointStamped):
        """Visualize each accepted TERCOM fix as a sphere + label at drone's ENU position.

        Markers are placed at the drone's current ESKF-estimated position so they
        appear along the traveled path in RViz2, consistent with the rejected-fix markers.
        Color: green (good quality) or yellow (marginal quality).
        """
        if self._est_odom is None:
            return  # no drone position yet

        mad  = self._last_tercom_quality[0] if len(self._last_tercom_quality) > 0 else 30.0
        disc = self._last_tercom_quality[1] if len(self._last_tercom_quality) > 1 else 1.5
        rough = self._last_tercom_quality[2] if len(self._last_tercom_quality) > 2 else 5.0

        # Use drone's ENU position (map frame) so markers appear along the travel path
        px = self._est_odom.pose.pose.position.x
        py = self._est_odom.pose.pose.position.y
        pz = self._est_odom.pose.pose.position.z

        self._accepted_count += 1
        base_id = self._tercom_marker_id
        self._tercom_marker_id += 2  # sphere + label

        # Quality classification: good (green) or marginal (yellow)
        quality = classify_terrain_quality(
            roughness=rough, roughness_min=5.0,
            discrimination=disc, discrimination_min=1.02,
            mad=mad, mad_threshold=30.0,
        )

        sphere = Marker()
        sphere.header.frame_id = 'map'
        sphere.header.stamp = msg.header.stamp
        sphere.ns = 'tercom_fixes'
        sphere.id = base_id
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = px
        sphere.pose.position.y = py
        sphere.pose.position.z = pz
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 12.0
        sphere.scale.y = 12.0
        sphere.scale.z = 12.0
        sphere.lifetime = Duration(sec=0)
        if quality == 'good':
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 0.0, 1.0, 0.0, 0.9
        else:  # marginal
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1.0, 1.0, 0.0, 0.9
        self._tercom_markers.markers.append(sphere)

        lbl = Marker()
        lbl.header.frame_id = 'map'
        lbl.header.stamp = msg.header.stamp
        lbl.ns = 'tercom_fixes'
        lbl.id = base_id + 1
        lbl.type = Marker.TEXT_VIEW_FACING
        lbl.action = Marker.ADD
        lbl.pose.position.x = px
        lbl.pose.position.y = py
        lbl.pose.position.z = pz + 15.0
        lbl.scale.z = 7.0
        lbl.color.r, lbl.color.g, lbl.color.b, lbl.color.a = 0.2, 1.0, 0.2, 1.0
        lbl.text = f'#{self._accepted_count} MAD:{mad:.1f}m disc:{disc:.2f}'
        lbl.lifetime = Duration(sec=0)
        self._tercom_markers.markers.append(lbl)

        self._pub_tc_fixes.publish(self._tercom_markers)

    def _cb_tercom_quality(self, msg: Float32MultiArray):
        self._last_tercom_quality = list(msg.data)

    def _cb_eskf_state(self, msg: String):
        prev = self._filter_state
        self._filter_state = msg.data
        # Schedule frame alignment for the next eskf_odom callback.
        # We cannot align here because eskf_odom is only published during RUNNING,
        # so self._est_odom is None or stale at the moment this state message arrives.
        if prev != 'RUNNING' and msg.data == 'RUNNING':
            self._pending_alignment = True

    def _cb_eskf_health(self, msg: Float32MultiArray):
        self._health = list(msg.data)
        if self._health:
            self._nis_deque.append(float(self._health[0]))

    def _cb_rejected_fix(self, msg: PointStamped):
        """Cache most-recent rejected fix; marker built when reason arrives."""
        self._pending_rejected_fix = msg

    def _cb_rejection_reason(self, msg: StringMsg):
        """Build an X-cross + label marker for a rejected TERCOM fix."""
        self._last_rejection_reason = msg.data
        if self._pending_rejected_fix is None:
            return
        fix = self._pending_rejected_fix
        self._pending_rejected_fix = None

        if self._est_odom is None:
            return  # no position to anchor the marker

        px = self._est_odom.pose.pose.position.x
        py = self._est_odom.pose.pose.position.y
        pz = self._est_odom.pose.pose.position.z

        base_id = self._rejected_marker_id
        self._rejected_marker_id += 3  # consumes ids: bar1, bar2, text

        cross_size = 15.0  # metres
        for i, angle_offset in enumerate([math.pi / 4, -math.pi / 4]):
            bar = Marker()
            bar.header.frame_id = 'map'
            bar.header.stamp = fix.header.stamp
            bar.ns = 'rejected_fixes'
            bar.id = base_id + i
            bar.type = Marker.CYLINDER
            bar.action = Marker.ADD
            bar.pose.position.x = px
            bar.pose.position.y = py
            bar.pose.position.z = pz
            bar.pose.orientation.z = math.sin(angle_offset / 2)
            bar.pose.orientation.w = math.cos(angle_offset / 2)
            bar.scale.x = 2.0
            bar.scale.y = 2.0
            bar.scale.z = cross_size
            bar.color.r = 1.0
            bar.color.g = 0.0
            bar.color.b = 0.0
            bar.color.a = 0.85
            bar.lifetime = Duration(sec=0)
            self._rejected_markers.markers.append(bar)

        lbl = Marker()
        lbl.header.frame_id = 'map'
        lbl.header.stamp = fix.header.stamp
        lbl.ns = 'rejected_fixes'
        lbl.id = base_id + 2
        lbl.type = Marker.TEXT_VIEW_FACING
        lbl.action = Marker.ADD
        lbl.pose.position.x = px
        lbl.pose.position.y = py
        lbl.pose.position.z = pz + 15.0
        lbl.scale.z = 8.0
        lbl.color.r = 1.0
        lbl.color.g = 0.3
        lbl.color.b = 0.3
        lbl.color.a = 1.0
        lbl.text = self._last_rejection_reason
        lbl.lifetime = Duration(sec=0)
        self._rejected_markers.markers.append(lbl)

        self._pub_rejected_viz.publish(self._rejected_markers)

    def _cb_tercom_timing(self, msg: Float32MultiArray):
        """Cache timing data published by tercom_node."""
        if len(msg.data) >= 4:
            self._tercom_timing = list(msg.data[:4])

    def _cb_eskf_timing(self, msg: Float32MultiArray):
        """Cache timing data published by eskf_node."""
        if len(msg.data) >= 6:
            self._eskf_timing = list(msg.data[:6])

    @staticmethod
    def _error_to_color(h_err: float):
        """Return (r, g, b) float tuple for a horizontal error value in metres."""
        if h_err < 10.0:
            return (0.0, 1.0, 0.0)
        elif h_err < 25.0:
            return (1.0, 0.8, 0.0)
        elif h_err < 50.0:
            return (1.0, 0.4, 0.0)
        else:
            return (1.0, 0.0, 0.0)

    # ---- Timers ----

    def _align_frames(self):
        """Compute the constant offset from MAVROS odom frame to ESKF ENU frame.

        Called on the first eskf_odom received after the filter enters RUNNING,
        so self._est_odom is guaranteed to hold the freshly-initialized ESKF position.
        The offset = est_pos - gt_pos, so gt_aligned = gt_pos + offset ≈ true ENU pos.
        """
        if self._est_odom is None or self._gt_odom is None:
            return
        est = self._est_odom.pose.pose.position
        gt = self._gt_odom.pose.pose.position
        self._frame_offset = np.array([
            est.x - gt.x,
            est.y - gt.y,
            est.z - gt.z,
        ])
        self.get_logger().info(
            f'Frame alignment: MAVROS→ENU offset = '
            f'({self._frame_offset[0]:.1f}, {self._frame_offset[1]:.1f}, '
            f'{self._frame_offset[2]:.1f}) m'
        )

    def _timer_error(self):
        """Publish current position error against ground truth."""
        _t0 = self._t_timer_error.start()
        if self._est_odom is None or self._gt_odom is None or self._frame_offset is None:
            self._t_timer_error.stop(_t0)
            return

        est = self._est_odom.pose.pose.position
        gt = self._gt_odom.pose.pose.position

        dx = est.x - (gt.x + self._frame_offset[0])
        dy = est.y - (gt.y + self._frame_offset[1])
        dz = est.z - (gt.z + self._frame_offset[2])
        h_err = math.hypot(dx, dy)

        self._h_errors.append(h_err)
        if len(self._h_errors) > 1000:
            self._h_errors = self._h_errors[-1000:]

        # Publish error vector
        err_vec = Vector3Stamped()
        err_vec.header = self._est_odom.header
        err_vec.vector.x = dx
        err_vec.vector.y = dy
        err_vec.vector.z = dz
        self._pub_pos_err.publish(err_vec)

        # Publish error norm
        err_norm_msg = Float64()
        err_norm_msg.data = h_err
        self._pub_err_norm.publish(err_norm_msg)

        # Error arrow: tail at GT position, tip at estimated position
        from geometry_msgs.msg import Point
        gt_px = gt.x + self._frame_offset[0]
        gt_py = gt.y + self._frame_offset[1]
        gt_pz = gt.z + self._frame_offset[2]
        r, g, b = self._error_to_color(h_err)
        shaft_diam = max(2.0, min(15.0, h_err / 5.0))

        arrow_ma = MarkerArray()
        arrow = Marker()
        arrow.header = self._est_odom.header
        arrow.header.frame_id = 'map'
        arrow.ns = 'error_arrow'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        tail = Point(); tail.x = gt_px; tail.y = gt_py; tail.z = gt_pz
        tip  = Point(); tip.x  = est.x;  tip.y  = est.y;  tip.z  = est.z
        arrow.points = [tail, tip]
        arrow.scale.x = shaft_diam
        arrow.scale.y = shaft_diam * 2.0
        arrow.scale.z = shaft_diam * 2.5
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = r, g, b, 0.9
        arrow.lifetime = Duration(sec=0, nanosec=300_000_000)
        arrow_ma.markers.append(arrow)

        lbl = Marker()
        lbl.header = arrow.header
        lbl.ns = 'error_arrow'
        lbl.id = 1
        lbl.type = Marker.TEXT_VIEW_FACING
        lbl.action = Marker.ADD
        lbl.pose.position.x = (est.x + gt_px) / 2
        lbl.pose.position.y = (est.y + gt_py) / 2
        lbl.pose.position.z = est.z + 20.0
        lbl.scale.z = 8.0
        lbl.color.r, lbl.color.g, lbl.color.b, lbl.color.a = r, g, b, 1.0
        lbl.text = f'err={h_err:.1f}m'
        lbl.lifetime = Duration(sec=0, nanosec=300_000_000)
        arrow_ma.markers.append(lbl)

        self._pub_error_arrow.publish(arrow_ma)

        # Covariance ellipse from ESKF pose covariance
        cov = self._est_odom.pose.covariance
        if any(c != 0 for c in cov):
            self._publish_covariance_ellipse(cov, self._est_odom.header)

        self._t_timer_error.stop(_t0)

    def _timer_paths(self):
        """Publish accumulated path visualizations."""
        _t0 = self._t_timer_paths.start()
        if self._est_path.poses:
            self._pub_est_path.publish(self._est_path)
        if self._gt_path.poses:
            # Apply current frame offset at publish time (poses stored raw).
            # Before alignment is known use zero offset so the path is visible
            # immediately; once the filter enters RUNNING the whole path shifts
            # once into the correct ENU frame with no internal discontinuity.
            ox, oy, oz = self._frame_offset if self._frame_offset is not None else (0.0, 0.0, 0.0)
            aligned = Path()
            aligned.header = self._gt_path.header
            aligned.header.frame_id = 'map'
            for raw_ps in self._gt_path.poses:
                ps = PoseStamped()
                ps.header = raw_ps.header
                ps.pose.orientation = raw_ps.pose.orientation
                ps.pose.position.x = raw_ps.pose.position.x + ox
                ps.pose.position.y = raw_ps.pose.position.y + oy
                ps.pose.position.z = raw_ps.pose.position.z + oz
                aligned.poses.append(ps)
            self._pub_gt_path.publish(aligned)

        # Error-colored path: one LINE_LIST marker per path segment
        from geometry_msgs.msg import Point
        poses = self._est_path.poses
        errors = self._est_path_errors
        if len(poses) >= 2 and len(errors) == len(poses):
            col_ma = MarkerArray()
            for i in range(len(poses) - 1):
                seg = Marker()
                seg.header.frame_id = 'map'
                seg.header.stamp = self.get_clock().now().to_msg()
                seg.ns = 'error_colored_path'
                seg.id = i
                seg.type = Marker.LINE_LIST
                seg.action = Marker.ADD
                p0 = Point()
                p0.x = poses[i].pose.position.x
                p0.y = poses[i].pose.position.y
                p0.z = poses[i].pose.position.z
                p1 = Point()
                p1.x = poses[i + 1].pose.position.x
                p1.y = poses[i + 1].pose.position.y
                p1.z = poses[i + 1].pose.position.z
                seg.points = [p0, p1]
                seg.scale.x = 3.0
                r2, g2, b2 = self._error_to_color(errors[i])
                seg.color.r, seg.color.g, seg.color.b, seg.color.a = r2, g2, b2, 0.9
                seg.lifetime = Duration(sec=0)
                col_ma.markers.append(seg)
            self._pub_error_col_path.publish(col_ma)

        # Error history chart (stride-4, up to 120 samples)
        if (self._h_errors and self._est_odom is not None and
                self._gt_odom is not None and self._frame_offset is not None):
            h_now = self._h_errors[-1]
            errors_arr = np.array(self._h_errors)
            rms_h = float(np.sqrt(np.mean(errors_arr ** 2)))
            max_h = float(np.max(errors_arr))
            v_err = abs(self._est_odom.pose.pose.position.z -
                        (self._gt_odom.pose.pose.position.z + self._frame_offset[2]))
            self._err_history.append((h_now, rms_h, max_h, v_err))

            chart_msg = Float32MultiArray()
            flat = []
            for (h, rm, mx, v) in self._err_history:
                flat += [h, rm, mx, v]
            chart_msg.data = flat
            self._pub_err_history_chart.publish(chart_msg)

        self._t_timer_paths.stop(_t0)

    def _timer_stats(self):
        """Publish 1 Hz running error statistics."""
        _t0 = self._t_timer_stats.start()
        if not self._h_errors:
            self._t_timer_stats.stop(_t0)
            return

        errors = np.array(self._h_errors)
        rms_h = float(np.sqrt(np.mean(errors**2)))
        max_h = float(np.max(errors))
        mean_h = float(np.mean(errors))

        # Vertical error
        v_err = 0.0
        if self._est_odom and self._gt_odom:
            v_err = abs(self._est_odom.pose.pose.position.z -
                        self._gt_odom.pose.pose.position.z)

        stats = Float32MultiArray()
        stats.data = [rms_h, max_h, mean_h, v_err,
                      self._h_errors[-1] if self._h_errors else 0.0, v_err]
        self._pub_err_stats.publish(stats)

        # NIS history (60 values at 1 Hz)
        if self._nis_deque:
            nis_msg = Float32MultiArray()
            nis_msg.data = list(self._nis_deque)
            self._pub_nis_history.publish(nis_msg)

        self.get_logger().info(
            f'State:{self._filter_state} | H_err: rms={rms_h:.1f}m max={max_h:.1f}m'
        )

        self._t_timer_stats.stop(_t0)

        # Publish aggregated profiling for all three nodes.
        # Layout (16 floats): [exec_ms, hz] × 8 components:
        #   [0-1]  tercom: cb_synced
        #   [2-3]  tercom: run_matching
        #   [4-5]  eskf: cb_imu
        #   [6-7]  eskf: cb_tercom_fix
        #   [8-9]  eskf: cb_altitude
        #   [10-11] diag: timer_error
        #   [12-13] diag: timer_paths
        #   [14-15] diag: timer_stats
        prof_msg = Float32MultiArray()
        prof_msg.data = (
            self._tercom_timing[:4]
            + self._eskf_timing[:6]
            + [
                float(self._t_timer_error.avg_exec_ms()),
                float(self._t_timer_error.avg_hz()),
                float(self._t_timer_paths.avg_exec_ms()),
                float(self._t_timer_paths.avg_hz()),
                float(self._t_timer_stats.avg_exec_ms()),
                float(self._t_timer_stats.avg_hz()),
            ]
        )
        self._pub_profiling.publish(prof_msg)

    def _publish_covariance_ellipse(self, cov, header):
        """Publish 2D covariance ellipse as a flat cylinder marker."""
        P2d = np.array([[cov[0], cov[1]],
                        [cov[6], cov[7]]])
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(P2d)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            a = 2.0 * math.sqrt(eigenvalues[1])  # 2-sigma
            b = 2.0 * math.sqrt(eigenvalues[0])
            angle = math.atan2(eigenvectors[1, 1], eigenvectors[0, 1])
        except Exception:
            return

        marker = Marker()
        marker.header = header
        marker.header.frame_id = 'map'
        marker.ns = 'covariance_ellipse'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        if self._est_odom:
            marker.pose.position = self._est_odom.pose.pose.position
        marker.pose.position.z -= 2.0  # slightly below drone for visibility

        # Orientation from eigenvector angle
        marker.pose.orientation.w = math.cos(angle / 2)
        marker.pose.orientation.z = math.sin(angle / 2)

        marker.scale.x = max(a, 1.0)
        marker.scale.y = max(b, 1.0)
        marker.scale.z = 0.5
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.3
        self._pub_cov_ellipse.publish(marker)

    def _publish_dem_pointcloud_once(self):
        """Generate DEM as PointCloud2 and publish with transient local QoS."""
        dem_file = self.get_parameter('dem_file').value
        if not dem_file:
            return

        try:
            from tercom_nav.core.dem_manager import DEMManager
            from sensor_msgs.msg import PointField
            import struct

            dem = DEMManager(dem_file)
            decimation = self.get_parameter('dem_pointcloud_decimation').value

            world_lat = self.get_parameter('world_origin_lat').value
            world_lon = self.get_parameter('world_origin_lon').value
            world_alt = self.get_parameter('world_origin_alt').value

            origin = compute_utm_origin(world_lat, world_lon, world_alt) if world_lat != 0.0 else None

            points_data = []
            elev_min, elev_max = dem.elevation_range
            elev_range = max(elev_max - elev_min, 1.0)

            for row in range(0, dem.height, decimation):
                for col in range(0, dem.width, decimation):
                    elev = float(dem.elevation[row, col])
                    if elev <= dem.nodata_value:
                        continue

                    easting = dem.transform.c + col * dem.transform.a
                    northing = dem.transform.f + row * dem.transform.e

                    if origin:
                        enu = utm_to_local_enu(
                            easting, northing, elev,
                            origin['easting'], origin['northing'], origin['alt']
                        )
                        x, y, z = float(enu[0]), float(enu[1]), float(enu[2])
                    else:
                        x, y, z = easting, northing, elev

                    # Jet colormap approximation
                    t = (elev - elev_min) / elev_range
                    r = max(0.0, min(1.0, 1.5 - abs(4 * t - 3)))
                    g = max(0.0, min(1.0, 1.5 - abs(4 * t - 2)))
                    b = max(0.0, min(1.0, 1.5 - abs(4 * t - 1)))
                    rgb = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
                    rgb_float = struct.unpack('f', struct.pack('I', rgb))[0]

                    points_data.append(struct.pack('ffff', x, y, z, rgb_float))

            if not points_data:
                return

            pc2 = PointCloud2()
            pc2.header.frame_id = 'map'
            pc2.header.stamp = self.get_clock().now().to_msg()
            pc2.height = 1
            pc2.width = len(points_data)
            pc2.is_dense = True
            pc2.is_bigendian = False
            pc2.point_step = 16
            pc2.row_step = pc2.point_step * pc2.width

            fields = []
            for name, offset, datatype in [
                ('x', 0, PointField.FLOAT32),
                ('y', 4, PointField.FLOAT32),
                ('z', 8, PointField.FLOAT32),
                ('rgb', 12, PointField.FLOAT32),
            ]:
                f = PointField()
                f.name = name
                f.offset = offset
                f.datatype = datatype
                f.count = 1
                fields.append(f)
            pc2.fields = fields

            import array as arr
            all_data = b''.join(points_data)
            pc2.data = arr.array('B', all_data).tolist()

            self._pub_dem_cloud.publish(pc2)
            self.get_logger().info(
                f'DEM PointCloud2 published: {len(points_data)} points '
                f'(decimation={decimation})'
            )
        except Exception as e:
            self.get_logger().error(f'DEM point cloud generation failed: {e}')

    # ---- CSV Logging ----

    def _setup_csv(self):
        csv_dir = self.get_parameter('csv_path').value
        os.makedirs(csv_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file_path = os.path.join(csv_dir, f'tercom_log_{timestamp}.csv')
        self._csv_file = open(csv_file_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            # Timing
            'ros_timestamp_ns', 'time_s',
            # Estimated state
            'est_x', 'est_y', 'est_z',
            'est_vx', 'est_vy', 'est_vz', 'est_speed',
            # Ground truth position
            'true_x', 'true_y', 'true_z',
            'true_vx', 'true_vy', 'true_vz', 'true_speed',
            # GPS ground truth (when available)
            'gt_lat', 'gt_lon', 'gt_alt',
            # Position errors
            'err_x', 'err_y', 'err_z', 'err_h_norm', 'err_v_abs', 'err_3d_norm',
            # ESKF covariance (position diagonal + xy cross-term + velocity diagonal)
            'cov_xx', 'cov_yy', 'cov_zz', 'cov_xy',
            'cov_vx', 'cov_vy', 'cov_vz',
            # TERCOM match quality
            'tercom_mad', 'tercom_disc', 'tercom_roughness', 'tercom_noise',
            # Accepted fix count
            'tercom_accepted_count',
            # Last rejection reason (empty string if last event was an accept)
            'last_rejection_reason',
            # Filter state and health
            'filter_state',
            'nis', 'health_max_pos_std', 'health_innov_norm', 'health_is_healthy', 'health_soft_reset_count', 'health_hard_reset_count',
        ])
        self._csv_t0_ns = None  # will be set on first row
        self.get_logger().info(f'Logging to CSV: {csv_file_path}')

    def _write_csv_row(self, est_msg: Odometry):
        if not self._gt_odom or not self._csv_writer or self._frame_offset is None:
            return

        try:
            ep = est_msg.pose.pose.position
            ev = est_msg.twist.twist.linear
            gp = self._gt_odom.pose.pose.position
            gv = self._gt_odom.twist.twist.linear
            cov = est_msg.pose.covariance

            # Apply frame alignment offset before computing error
            dx = ep.x - (gp.x + self._frame_offset[0])
            dy = ep.y - (gp.y + self._frame_offset[1])
            dz = ep.z - (gp.z + self._frame_offset[2])

            # TERCOM quality metrics
            q = self._last_tercom_quality
            mad   = q[0] if len(q) > 0 else 0.0
            disc  = q[1] if len(q) > 1 else 0.0
            rough = q[2] if len(q) > 2 else 0.0
            noise = q[3] if len(q) > 3 else 0.0

            # Filter health metrics
            nis            = self._health[0] if len(self._health) > 0 else 0.0
            max_pos_std    = self._health[1] if len(self._health) > 1 else 0.0
            innov_norm     = self._health[2] if len(self._health) > 2 else 0.0
            is_healthy     = self._health[3] if len(self._health) > 3 else 1.0
            soft_reset_count = self._health[4] if len(self._health) > 4 else 0.0
            hard_reset_count = self._health[5] if len(self._health) > 5 else 0.0

            # Timing
            ts_ns = est_msg.header.stamp.sec * 10**9 + est_msg.header.stamp.nanosec
            if self._csv_t0_ns is None:
                self._csv_t0_ns = ts_ns
            time_s = (ts_ns - self._csv_t0_ns) * 1e-9

            # Speed magnitudes
            est_speed  = math.sqrt(ev.x**2 + ev.y**2 + ev.z**2)
            true_speed = math.sqrt(gv.x**2 + gv.y**2 + gv.z**2)

            # GPS ground truth (lat/lon/alt) when available
            gt_lat = self._gt_global.latitude  if self._gt_global else float('nan')
            gt_lon = self._gt_global.longitude if self._gt_global else float('nan')
            gt_alt = self._gt_global.altitude  if self._gt_global else float('nan')

            self._csv_writer.writerow([
                # Timing
                ts_ns, time_s,
                # Estimated state
                ep.x, ep.y, ep.z,
                ev.x, ev.y, ev.z, est_speed,
                # Ground truth state
                gp.x, gp.y, gp.z,
                gv.x, gv.y, gv.z, true_speed,
                # GPS ground truth
                gt_lat, gt_lon, gt_alt,
                # Errors
                dx, dy, dz,
                math.hypot(dx, dy),          # err_h_norm
                abs(dz),                     # err_v_abs
                math.sqrt(dx**2 + dy**2 + dz**2),  # err_3d_norm
                # Covariance (position diagonal + xy cross-term + velocity diagonal)
                cov[0], cov[7], cov[14],     # cov_xx, cov_yy, cov_zz
                cov[1],                      # cov_xy (x–y cross-covariance)
                cov[21], cov[28], cov[35],   # cov_vx, cov_vy, cov_vz
                # TERCOM quality
                mad, disc, rough, noise,
                # Fix accounting
                self._accepted_count,
                self._last_rejection_reason,
                # Filter state and health
                self._filter_state,
                nis, max_pos_std, innov_norm, is_healthy, soft_reset_count, hard_reset_count,
            ])
        except Exception as e:
            self.get_logger().warning(f'CSV write error: {e}')

    def destroy_node(self):
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = DiagnosticsNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Diagnostics node error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
