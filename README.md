# tercom_nav

GPS-denied navigation for fixed-wing UAVs using **TERCOM** (Terrain Contour Matching) fused with an **Error-State Kalman Filter** (ESKF). The package operates entirely in the `tercom` ROS 2 namespace and interfaces with PX4 SITL via MAVROS.

---

## Overview

The system estimates UAV position without GPS by correlating a real-time terrain elevation profile (barometer + rangefinder) against a pre-loaded Digital Elevation Model (DEM). GPS is used once at startup to initialize the filter; after that, the ESKF runs on IMU prediction with periodic TERCOM, barometric, and velocity updates.

### Architecture

```
MAVROS topics
    │
    ├── imu/data ──────────────────────────────────────► eskf_node (prediction)
    ├── altitude ──────────────────────────► tercom_node ──► eskf_node (baro update)
    ├── distance_sensor/rangefinder_pub ───► tercom_node
    ├── local_position/odom ───────────────► tercom_node
    ├── global_position/global ────────────────────────► eskf_node (init only)
    └── local_position/velocity_local ─────────────────► eskf_node (velocity update)

tercom_node ──► /tercom/tercom_node/position_fix ──────► eskf_node (TERCOM update)
tercom_node ◄── /tercom/eskf_node/pose (covariance feedback for dynamic search radius)

eskf_node ──► /tercom/eskf_node/odom ──────────────────► diagnostics_node
```

**Node responsibilities:**

| Node | Responsibility |
|---|---|
| `dem_server_node` | Loads GeoTIFF DEM at startup; exposes metadata via latched topic and ROS services |
| `tercom_node` | Collects synchronized baro + rangefinder profiles; runs vectorized TERCOM correlation matching |
| `eskf_node` | 15-state ESKF (position / velocity / attitude / accel-bias / gyro-bias); fuses IMU + TERCOM + baro + velocity |
| `diagnostics_node` | Computes ground-truth error metrics, publishes RViz visualizations, and logs CSV |

---

## Requirements

### ROS 2

- **ROS 2 Humble** or later (tested on Humble)
- Build type: `ament_python`

### ROS dependencies

```
rclpy
std_msgs
geometry_msgs
nav_msgs
sensor_msgs
visualization_msgs
mavros_msgs
tf2_ros
diagnostic_msgs
std_srvs
message_filters
```

### Python dependencies

| Package | Purpose |
|---|---|
| `numpy` | Vectorized TERCOM matching, ESKF matrices |
| `scipy` | Quaternion operations in ESKF |
| `rasterio` | GeoTIFF DEM loading and CRS auto-detection |
| `pyproj` | Coordinate reference system transformations (UTM ↔ WGS84) |

Install Python dependencies:

```bash
pip install numpy scipy rasterio pyproj
```

### Hardware / Simulation

- A GeoTIFF DEM covering the flight area (geographic or projected CRS; auto-reprojected to UTM if needed)
- A rangefinder (AGL distance) + barometer (AMSL altitude) — or their MAVROS equivalents in simulation
- Tested against **PX4 SITL + Gazebo** using the `taif_test4` terrain model

---

## Installation

```bash
# Clone into your ROS 2 workspace
cd ~/ros2_ws/src
git clone <repo-url> tercom_nav

# Install ROS dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --packages-select tercom_nav

# Source
source install/setup.bash
```

---

## Quick Start

### Full system launch (all 4 nodes)

```bash
ros2 launch tercom_nav tercom_nav.launch.py \
    dem_file:=/path/to/terrain.tif \
    world_origin_lat:=<lat> \
    world_origin_lon:=<lon> \
    world_origin_alt:=<alt_msl>
```

### Minimal launch (TERCOM + ESKF only, no diagnostics)

```bash
ros2 launch tercom_nav tercom_standalone.launch.py \
    dem_file:=/path/to/terrain.tif \
    world_origin_lat:=<lat> \
    world_origin_lon:=<lon> \
    world_origin_alt:=<alt_msl>
```

### taif_test4 example (PX4 SITL)

```bash
ros2 launch tercom_nav tercom_nav.launch.py \
    dem_file:=/home/user/shared_volume/PX4-Autopilot/Tools/simulation/gz/models/taif_test4/textures/taif_test4_tercom_dem.tif \
    world_origin_lat:=21.2651 \
    world_origin_lon:=40.3542 \
    world_origin_alt:=1859.7
```

Or use the pre-configured override file:

```bash
ros2 launch tercom_nav tercom_nav.launch.py \
    params_file:=$(ros2 pkg prefix tercom_nav)/share/tercom_nav/config/taif_test4_params.yaml
```

---

## Launch Arguments

### `tercom_nav.launch.py`

| Argument | Default | Description |
|---|---|---|
| `dem_file` | `""` | **Required.** Absolute path to GeoTIFF DEM |
| `dem_metadata_file` | `""` | Optional JSON sidecar with DEM metadata |
| `mavros_ns` | `target/mavros` | MAVROS namespace prefix |
| `world_origin_lat` | `0.0` | Gazebo world origin latitude (degrees) |
| `world_origin_lon` | `0.0` | Gazebo world origin longitude (degrees) |
| `world_origin_alt` | `0.0` | Gazebo world origin altitude MSL (m) |
| `use_sim_time` | `true` | Use `/clock` from simulation |
| `params_file` | `<pkg>/config/tercom_params.yaml` | Parameter override YAML |

### `tercom_standalone.launch.py`

Same as above except `dem_metadata_file` and `params_file` are not exposed (always uses package defaults). Use for lightweight deployments where diagnostics are not needed.

---

## Parameters

All parameters are set via the params YAML. The defaults are in `config/tercom_params.yaml`.

### dem_server_node

| Parameter | Default | Description |
|---|---|---|
| `dem_file` | `""` | Absolute path to GeoTIFF DEM |
| `dem_metadata_file` | `""` | Optional JSON sidecar |
| `nodata_value` | `-9999.0` | Sentinel value for missing cells |
| `interpolation_method` | `"bilinear"` | `"nearest"` or `"bilinear"` |

### tercom_node

| Parameter | Default | Description |
|---|---|---|
| `dem_file` | `""` | Absolute path to GeoTIFF DEM |
| `profile_min_spacing_m` | `-1.0` | Min distance between profile samples (m); `-1` = auto = `1.0 × pixel_size` |
| `profile_max_samples` | `20` | Max samples per profile before triggering a match |
| `profile_min_distance_m` | `-1.0` | Min total profile length (m); `-1` = auto = `15 × pixel_size` |
| `search_radius_pixels` | `50` | Static search half-size in pixels |
| `search_radius_dynamic` | `true` | Adapt search radius from ESKF covariance |
| `search_radius_sigma_mult` | `3.0` | Dynamic radius = `mult × sigma_pos / pixel_size` |
| `search_radius_min_pixels` | `10` | Lower bound for dynamic search radius |
| `search_radius_max_pixels` | `100` | Upper bound for dynamic search radius |
| `mad_reject_threshold` | `30.0` | Reject match if MAD > this (m) |
| `discrimination_min` | `1.5` | Reject match if `2nd_best / best < this` |
| `discrimination_exclusion_radius` | `3` | Pixel exclusion zone around best for 2nd-best search |
| `roughness_min` | `3.0` | Reject match if terrain std < this (m) — flat terrain |
| `enable_adaptive_sampling` | `true` | Enable velocity-based adaptive sampling |
| `adaptive_min_interval_s` | `0.5` | Min time between adaptive samples (s) |
| `adaptive_max_interval_s` | `5.0` | Max time between adaptive samples (s) |
| `adaptive_pixels_per_sample` | `1.5` | Target spacing in DEM pixels per sample |
| `sync_slop_s` | `0.05` | Timestamp tolerance for baro/rangefinder sync (s) |
| `world_origin_lat` | `0.0` | Gazebo world origin latitude |
| `world_origin_lon` | `0.0` | Gazebo world origin longitude |
| `world_origin_alt` | `0.0` | Gazebo world origin altitude MSL (m) |

### eskf_node

| Parameter | Default | Description |
|---|---|---|
| `imu_rate_hz` | `50.0` | Decimated IMU processing rate (raw input ~250 Hz) |
| `accel_noise` | `0.1` | Accelerometer noise density (m/s²/√Hz) |
| `gyro_noise` | `0.01` | Gyroscope noise density (rad/s/√Hz) |
| `accel_bias_noise` | `0.001` | Accelerometer bias random walk (m/s³) |
| `gyro_bias_noise` | `0.0001` | Gyroscope bias random walk (rad/s²) |
| `bias_time_constant` | `300.0` | Gauss-Markov bias time constant (s) |
| `tercom_noise_base` | `-1.0` | Base TERCOM position noise (m); `-1` = auto from DEM pixel size |
| `baro_noise` | `3.0` | Barometric altitude noise std (m) |
| `baro_update_rate_hz` | `1.0` | Rate limit for barometric updates (Hz) |
| `velocity_noise` | `0.5` | Velocity measurement noise std (m/s) |
| `enable_velocity_updates` | `true` | Enable MAVROS velocity aiding |
| `velocity_update_rate_hz` | `5.0` | Rate limit for velocity updates (Hz) |
| `init_pos_std` | `5.0` | Initial position uncertainty (m) |
| `init_vel_std` | `1.0` | Initial velocity uncertainty (m/s) |
| `init_att_std` | `0.05` | Initial attitude uncertainty (rad) |
| `init_abias_std` | `0.5` | Initial accelerometer bias uncertainty (m/s²) |
| `init_wbias_std` | `0.01` | Initial gyroscope bias uncertainty (rad/s) |
| `gps_init_samples` | `20` | GPS fixes averaged for initialization |
| `gps_init_timeout_s` | `30.0` | Max wait time for GPS initialization (s) |
| `nis_threshold` | `15.0` | NIS window average above this → divergence (χ² 2DOF 95% = 5.99) |
| `nis_window_size` | `10` | Window size for NIS averaging |
| `max_position_std` | `500.0` | Position std above this → divergence (m) |
| `max_innovation_m` | `200.0` | Innovation gate: reject TERCOM updates beyond this (m) |
| `consecutive_reject_limit` | `5` | Consecutive rejections before divergence |
| `divergence_action` | `"reset"` | Response to divergence: `"warn"`, `"reset"`, or `"reset_with_gps"` |
| `world_origin_lat/lon/alt` | `0.0` | Must match `tercom_node` and `diagnostics_node` |

### diagnostics_node

| Parameter | Default | Description |
|---|---|---|
| `log_to_csv` | `true` | Enable CSV logging |
| `csv_path` | `"/tmp/tercom_logs/"` | Output directory for CSV files |
| `publish_dem_pointcloud` | `true` | Publish DEM as PointCloud2 at startup |
| `dem_pointcloud_decimation` | `4` | Publish every Nth DEM pixel |
| `dem_file` | `""` | DEM file for point cloud generation |
| `error_publish_rate_hz` | `10.0` | Rate for error metric topics (Hz) |
| `path_publish_rate_hz` | `2.0` | Rate for path visualization topics (Hz) |
| `world_origin_lat/lon/alt` | `0.0` | Must match other nodes |

---

## Topics

### tercom_node

**Subscribes:**

| Topic (remapped from) | Type | Description |
|---|---|---|
| `altitude` | `mavros_msgs/Altitude` | Barometric AMSL altitude (synchronized with rangefinder) |
| `distance_sensor` | `sensor_msgs/Range` | Rangefinder AGL distance |
| `imu_data` | `sensor_msgs/Imu` | Roll/pitch correction for rangefinder tilt |
| `local_odom` | `nav_msgs/Odometry` | Local ENU position for profile collection |
| `eskf_covariance` | `geometry_msgs/PoseWithCovarianceStamped` | ESKF covariance for dynamic search radius |

**Publishes:**

| Topic | Type | Description |
|---|---|---|
| `/tercom/tercom_node/position_fix` | `geometry_msgs/PointStamped` | UTM position fix (`x`=easting, `y`=northing, `z`=terrain elevation); `frame_id=utm` |
| `/tercom/tercom_node/match_quality` | `std_msgs/Float32MultiArray` | `[MAD (m), discrimination ratio, roughness (m), adaptive_noise_var]` |
| `/tercom/tercom_node/profile_path` | `nav_msgs/Path` | Profile sample positions for RViz |
| `/tercom/tercom_node/search_region` | `visualization_msgs/Marker` | Search window wireframe |
| `/tercom/tercom_node/status` | `std_msgs/String` | `COLLECTING`, `MATCHING`, or `WAITING_SENSORS` at 1 Hz |

**Services:**

| Service | Type | Description |
|---|---|---|
| `/tercom/tercom_node/trigger_match` | `std_srvs/Trigger` | Force a match with current samples (≥ 5 required) |
| `/tercom/tercom_node/reset_profile` | `std_srvs/Trigger` | Clear all collected samples |

### eskf_node

**Subscribes:**

| Topic (remapped from) | Type | Description |
|---|---|---|
| `imu_data` | `sensor_msgs/Imu` | High-rate IMU (decimated to `imu_rate_hz`) |
| `gps_global` | `sensor_msgs/NavSatFix` | GPS fix — used only during initialization |
| `altitude` | `mavros_msgs/Altitude` | Barometric AMSL altitude |
| `velocity_local` | `geometry_msgs/TwistStamped` | ENU velocity aiding |
| `local_odom` | `nav_msgs/Odometry` | Stored for initialization velocity |
| `tercom_fix` | `geometry_msgs/PointStamped` | TERCOM UTM fix |
| `tercom_quality` | `std_msgs/Float32MultiArray` | Match quality for adaptive measurement noise |

**Publishes:**

| Topic | Type | Rate | Description |
|---|---|---|---|
| `/tercom/eskf_node/odom` | `nav_msgs/Odometry` | `imu_rate_hz` | Full odometry with 6×6 covariance |
| `/tercom/eskf_node/pose` | `geometry_msgs/PoseWithCovarianceStamped` | `imu_rate_hz` | Pose + covariance (fed back to tercom_node) |
| `/tercom/eskf_node/global` | `sensor_msgs/NavSatFix` | 1 Hz | Estimated lat/lon/alt |
| `/tercom/eskf_node/bias_accel` | `geometry_msgs/Vector3Stamped` | 1 Hz | Estimated accelerometer bias |
| `/tercom/eskf_node/bias_gyro` | `geometry_msgs/Vector3Stamped` | 1 Hz | Estimated gyroscope bias |
| `/tercom/eskf_node/state` | `std_msgs/String` | On change | Filter state machine state |
| `/tercom/eskf_node/health` | `std_msgs/Float32MultiArray` | 1 Hz | `[avg_NIS, max_pos_std (m), innov_norm, is_healthy]` |

**Services:**

| Service | Type | Description |
|---|---|---|
| `/tercom/eskf_node/reset_filter` | `std_srvs/Trigger` | Hard reset: re-enters `WAITING_GPS` state |

### dem_server_node

**Publishes:**

| Topic | Type | Description |
|---|---|---|
| `/tercom/dem_server_node/dem_info` | `std_msgs/String` | Latched JSON: CRS, pixel size, bounds, elevation range |

**Services:**

| Service | Type | Description |
|---|---|---|
| `/tercom/dem_server_node/get_dem_info` | `std_srvs/Trigger` | Returns DEM metadata as JSON string |
| `/tercom/dem_server_node/get_elevation` | `std_srvs/Trigger` | Returns DEM bounding box info |

### diagnostics_node

**Publishes:**

| Topic | Type | Description |
|---|---|---|
| `/tercom/diagnostics_node/position_error` | `geometry_msgs/Vector3Stamped` | Per-axis error vs ground truth (m) |
| `/tercom/diagnostics_node/error_norm` | `std_msgs/Float64` | Horizontal error norm (m) |
| `/tercom/diagnostics_node/error_stats` | `std_msgs/Float32MultiArray` | `[rms_h, max_h, mean_h, v_err, latest_h, v_err]` (m) |
| `/tercom/diagnostics_node/estimated_path` | `nav_msgs/Path` | Accumulated ESKF trajectory |
| `/tercom/diagnostics_node/ground_truth_path` | `nav_msgs/Path` | Accumulated ground truth trajectory |
| `/tercom/diagnostics_node/tercom_fixes_viz` | `visualization_msgs/MarkerArray` | Sphere markers colored by match quality |
| `/tercom/diagnostics_node/covariance_ellipse` | `visualization_msgs/Marker` | 2-sigma covariance ellipse |
| `/tercom/diagnostics_node/dem_surface` | `sensor_msgs/PointCloud2` | DEM point cloud with jet colormap (latched, published once) |

---

## ESKF State Machine

The filter transitions through the following states:

```
WAITING_GPS ──► INITIALIZING ──► RUNNING ◄──► DIVERGED ──► RESETTING
                                                  │
                                                  └──► RUNNING (after soft reset)
```

| State | Description |
|---|---|
| `WAITING_GPS` | Waiting for first valid GPS fix |
| `INITIALIZING` | Averaging `gps_init_samples` GPS fixes to compute initial position |
| `RUNNING` | Normal operation: IMU prediction + TERCOM/baro/velocity updates |
| `DIVERGED` | Filter health checks failed; action depends on `divergence_action` |
| `RESETTING` | Hard reset in progress; re-acquires GPS |

**Divergence actions:**

| `divergence_action` | Behavior |
|---|---|
| `"warn"` | Log warning only; stay in `RUNNING` |
| `"reset"` | Inflate covariance, zero biases; stay in `RUNNING` |
| `"reset_with_gps"` | Full hard reset; re-enter `WAITING_GPS` |

---

## TERCOM Matching

Terrain elevation at each sample point is computed as:

```
h_terrain = h_baro_AMSL − h_AGL × cos(roll) × cos(pitch)
```

A match is triggered when:
1. `num_samples ≥ profile_max_samples`, **and**
2. `total_distance ≥ profile_min_distance_m`

The matcher evaluates all candidate positions in the search window simultaneously using vectorized NumPy operations. A candidate match is **rejected** if any of these fail:

| Check | Threshold |
|---|---|
| MAD (mean absolute deviation) | `> mad_reject_threshold` |
| Discrimination ratio (2nd-best / best) | `< discrimination_min` |
| Terrain roughness (std of DEM window) | `< roughness_min` |

After a successful match, a sliding window retains the last `max_samples / 2` samples for profile continuity into the next match.

**Fix marker colors** in RViz (diagnostics_node):
- Green — roughness > 6 m AND discrimination > 2.25 AND MAD < 15 m
- Yellow — all thresholds pass but not by the above margin
- Red — any rejection threshold failed

---

## CSV Logging

When `log_to_csv: true`, `diagnostics_node` creates a timestamped CSV at `csv_path/tercom_log_YYYYMMDD_HHMMSS.csv`.

**Columns:** `ros_timestamp_ns`, `est_x`, `est_y`, `est_z`, `est_vx`, `est_vy`, `est_vz`, `true_x`, `true_y`, `true_z`, `true_vx`, `true_vy`, `true_vz`, `err_x`, `err_y`, `err_z`, `err_h_norm`, `err_3d_norm`, `cov_xx`, `cov_yy`, `cov_zz`, `tercom_mad`, `tercom_disc`, `tercom_roughness`, `filter_state`, `nis`

---

## RViz

A pre-configured RViz layout is provided:

```bash
rviz2 -d $(ros2 pkg prefix tercom_nav)/share/tercom_nav/config/rviz_tercom.rviz
```

Key displays:
- DEM surface point cloud (jet colormap by elevation)
- Estimated path and ground truth path
- TERCOM fix markers (color-coded by quality)
- Profile collection path
- 2-sigma covariance ellipse

---

## Core Modules

| Module | Description |
|---|---|
| `core/dem_manager.py` | Loads any GeoTIFF DEM; auto-reprojects geographic → UTM if needed; bilinear or nearest-neighbor elevation lookups (single-point and vectorized batch) |
| `core/tercom_matcher.py` | `ProfileCollector` sliding-window buffer with min-spacing enforcement; `match_profile()` fully vectorized TERCOM correlation returning best UTM position, MAD, discrimination ratio, and roughness |
| `core/eskf.py` | 15-state ESKF (position / velocity / attitude / accel-bias / gyro-bias) with IMU-driven linearized prediction and separate Joseph-form updates for 2D TERCOM position, barometric altitude, and 3D velocity |
| `core/adaptive_sampler.py` | Distance-based trigger: fires when the drone travels `pixels_per_sample × pixel_size` meters since the last sample, with min/max time interval clamps |
| `core/coordinate_utils.py` | WGS84 ↔ UTM ↔ local ENU ↔ DEM pixel coordinate conversions; caches `pyproj.Transformer` objects for efficiency |
| `core/health_monitor.py` | Three independent ESKF health checks: NIS windowed average, position covariance bound, and consecutive innovation rejection count |
| `core/terrain_quality.py` | Terrain roughness computation; `good` / `marginal` / `poor` classification; adaptive TERCOM measurement noise variance scaling with MAD, roughness, and discrimination |

---

## Tests

Run all tests:

```bash
cd ~/ros2_ws
colcon test --packages-select tercom_nav
colcon test-result --verbose
```

Or run directly with pytest (no ROS runtime required):

```bash
cd src/tercom_nav
pytest test/ -v
```

**46 tests across 5 files**, all pure unit tests against core library code:

| File | Tests | What is covered |
|---|---|---|
| `test_adaptive_sampler.py` | 9 | Adaptive sampler triggers, health monitor divergence and reset |
| `test_coordinate_utils.py` | 7 | UTM conversion, ENU round-trips, pixel coordinate transforms |
| `test_dem_manager.py` | 7 | DEM load, elevation lookup, bounds check, batch vs single |
| `test_eskf.py` | 15 | ESKF init, prediction, all update types, covariance symmetry, divergence |
| `test_tercom_matcher.py` | 8 | Profile collection, spacing enforcement, match recovery, flat terrain, error handling |

---

## Package Structure

```
tercom_nav/
├── package.xml
├── setup.py
├── setup.cfg
├── README.md
├── resource/
│   └── tercom_nav
├── config/
│   ├── tercom_params.yaml       # Default parameters for all nodes
│   ├── taif_test4_params.yaml   # Override config for taif_test4 DEM (PX4 SITL)
│   └── rviz_tercom.rviz         # RViz layout
├── launch/
│   ├── tercom_nav.launch.py         # Full system: all 4 nodes
│   └── tercom_standalone.launch.py  # Minimal: tercom_node + eskf_node only
├── tercom_nav/
│   ├── nodes/
│   │   ├── dem_server_node.py
│   │   ├── tercom_node.py
│   │   ├── eskf_node.py
│   │   └── diagnostics_node.py
│   └── core/
│       ├── dem_manager.py
│       ├── tercom_matcher.py
│       ├── eskf.py
│       ├── adaptive_sampler.py
│       ├── coordinate_utils.py
│       ├── health_monitor.py
│       └── terrain_quality.py
└── test/
    ├── test_adaptive_sampler.py
    ├── test_coordinate_utils.py
    ├── test_dem_manager.py
    ├── test_eskf.py
    └── test_tercom_matcher.py
```
