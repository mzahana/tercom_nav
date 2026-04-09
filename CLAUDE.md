# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build
cd ~/ros2_ws  # or /home/mabdelkader/gpsdnav_shared_volume/ros2_ws
colcon build --packages-select tercom_nav
source install/setup.bash

# Run all unit tests (no ROS runtime needed)
pytest test/ -v

# Run a single test file
pytest test/test_eskf.py -v

# Full ROS test suite
colcon test --packages-select tercom_nav
colcon test-result --verbose

# Lint (run via colcon test; these are configured as test_depend)
# ament_copyright, ament_flake8, ament_pep257
```

## Launch

```bash
# Full system (all 4 nodes)
ros2 launch tercom_nav tercom_nav.launch.py \
    params_file:=$(ros2 pkg prefix tercom_nav)/share/tercom_nav/config/taif_test4_params.yaml

# Minimal (TERCOM + ESKF only, no diagnostics)
ros2 launch tercom_nav tercom_standalone.launch.py \
    dem_file:=/path/to/dem.tif \
    world_origin_lat:=<lat> world_origin_lon:=<lon> world_origin_alt:=<alt>

# RViz
rviz2 -d $(ros2 pkg prefix tercom_nav)/share/tercom_nav/config/rviz_tercom.rviz
```

## Diagnostic Commands

```bash
ros2 topic echo /tercom/tercom_node/status --once       # COLLECTING|MATCHING|WAITING_SENSORS
ros2 topic hz /tercom/tercom_node/position_fix          # Fix frequency
ros2 topic echo /tercom/eskf_node/health --once         # [avg_NIS, max_pos_std, innov_norm, is_healthy]
ros2 service call /tercom/tercom_node/trigger_match std_srvs/srv/Trigger {}
ros2 param list /tercom/tercom_node                     # Verify parameter loading
```

## Architecture Overview

**tercom_nav** is a ROS 2 package for GPS-denied UAV navigation using TERCOM (Terrain Contour Matching) fused with a 15-state Error-State Kalman Filter (ESKF). GPS is used once for initialization only; subsequent navigation relies on IMU prediction + TERCOM terrain correlation.

### Node Topology

```
MAVROS (PX4 SITL)
  imu/data ──────────────► eskf_node  (prediction at imu_rate_hz=50 Hz)
  altitude ───────────────► tercom_node + eskf_node (barometric altitude update)
  distance_sensor ────────► tercom_node (AGL rangefinder, LaserScan type, element [0])
  local_position/odom ────► tercom_node + diagnostics_node
  global_position/global ─► eskf_node (init only) + diagnostics_node
  velocity_local ─────────► eskf_node (velocity aiding)

  tercom_node ──position_fix──► eskf_node (2D TERCOM position update)
              ──match_quality──► eskf_node (adaptive measurement noise)
  eskf_node ──pose──────────► tercom_node (covariance → dynamic search radius)
```

All nodes run under the `/tercom` namespace. Internal topic names are `/tercom/<node_name>/<topic>`.

### Core Modules (`tercom_nav/core/`) — No ROS Dependencies

| Module | Responsibility |
|--------|---------------|
| `dem_manager.py` | Load GeoTIFF, auto-detect CRS, reproject geographic→UTM, batch elevation lookup |
| `tercom_matcher.py` | `ProfileCollector` (sliding-window buffer + spacing enforcement) + vectorized TERCOM correlation via NumPy broadcasting (no Python loops) |
| `eskf.py` | 15-state ESKF: `[Δpos(3), Δvel(3), Δatt(3), Δaccel_bias(3), Δgyro_bias(3)]`; updates for 2D position, altitude, velocity |
| `coordinate_utils.py` | WGS84↔UTM↔local ENU↔DEM pixel; caches pyproj Transformer at module level |
| `adaptive_sampler.py` | Distance-based terrain sample trigger |
| `health_monitor.py` | Three-check divergence detection: NIS windowed average, covariance bounds, consecutive rejection count |
| `terrain_quality.py` | DEM roughness, match quality classification, adaptive noise scaling |

### ESKF State Machine

```
WAITING_GPS → INITIALIZING → RUNNING ↔ DIVERGED → RESETTING → WAITING_GPS
```

Divergence action (configurable): `"warn"` / `"reset"` (inflate covariance) / `"reset_with_gps"` (full re-init).

### TERCOM Match Quality Rejection

A TERCOM fix is rejected if any condition fails:
- MAD > `mad_reject_threshold` (default 30 m)
- discrimination < `discrimination_min` (default 1.02; low on straight-line flights)
- roughness < `roughness_min` (default 5 m; flat terrain is unresolvable)

Accepted fixes use adaptive noise: poor-quality matches inflate `R_tercom` so the filter trusts them less.

### Key Design Decisions (from TROUBLESHOOT.md)

1. **MAVROS sensor subscriptions** must use `QoSPresetProfiles.SENSOR_DATA` (BEST_EFFORT) — MAVROS publishers are BEST_EFFORT; RELIABLE subscriber will get no data silently.
2. **Rangefinder** is published as `sensor_msgs/LaserScan` (GPU lidar), not `sensor_msgs/Range`. Extract `msg.ranges[0]` with bounds check.
3. **Dynamic search radius** minimum is 80 pixels (~475 m) for `taif_test4`. The default 50 px minimum was too small and caused the true position to fall outside the search window.
4. **Parameter override YAML files** must be complete copies of `tercom_params.yaml`. Partial files cause the launch file defaults to silently overwrite YAML values.
5. **Frame alignment**: MAVROS `local_position` uses PX4 NED boot origin ≠ ESKF ENU (UTM origin). `diagnostics_node` records and applies the offset on the first `RUNNING` sample.
6. **YAML parameter node names** must include the `/tercom/` prefix (e.g., `/tercom/tercom_node:`) to match the namespace.

### Configuration Files

- `config/tercom_params.yaml` — default parameters for all 4 nodes
- `config/taif_test4_params.yaml` — complete override tuned for PX4 SITL `taif_test4` world (origin: 21.2651°N, 40.3542°E, 1859.7 m MSL); **must be a full copy of tercom_params.yaml**

### CSV Logging

`diagnostics_node` writes 25-column CSV to `csv_path` (configured in params): timestamps, estimated/true position+velocity, per-axis errors, covariance diagonals, TERCOM match metrics, filter state, NIS.
