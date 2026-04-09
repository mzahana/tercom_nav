# tercom_nav Troubleshooting Guide

A record of every issue encountered during bring-up of the TERCOM navigation system,
the root cause, and the fix applied. Ordered chronologically.

---

## Issue 1 — Nodes crash at startup: `dem_file parameter is required`

**Symptom**
```
[dem_server-1] [ERROR] Parameter "dem_file" is required but not set!
[tercom_node-2] [ERROR] "dem_file" parameter is required!
```

**Root cause — two bugs:**

**Bug A: YAML node names did not include the namespace.**
The YAML files used bare names (`dem_server_node:`) but all nodes run under the
`/tercom` namespace, so ROS 2 looks for `/tercom/dem_server_node`. The names
did not match, the entire parameter block was silently ignored, and every parameter
fell back to its `declare_parameter` default (empty string for `dem_file`).

**Bug B: Launch file overrides clobbered YAML values.**
The launch file always built override dicts like `{'dem_file': dem_file, 'world_origin_lat': 0.0, ...}`
and placed them *after* `params_file` in the `parameters` list. When only `params_file`
was passed on the CLI, the unset defaults (`dem_file=''`, `world_origin_*=0.0`) silently
overwrote whatever the YAML had set.

**Fix**
- Prefixed all node keys in both YAML files with `/tercom/` (e.g. `/tercom/dem_server_node:`).
- Changed launch arg defaults for `world_origin_*` to sentinel `'__unset__'`; only insert
  overrides into the dict when a value was actually provided on the CLI.

---

## Issue 2 — QoS incompatibility: no messages received from MAVROS

**Symptom**
```
[WARN] New publisher discovered on topic 'altitude', offering incompatible QoS.
       No messages will be received from it. Last incompatible policy: RELIABILITY
```
Same warning for `imu_data`, `gps_global`, `local_odom`, `velocity_local`,
`ground_truth_odom`, `ground_truth_global`.

**Root cause**
MAVROS publishes all sensor topics with `BEST_EFFORT` reliability (standard for
high-rate sensor data). Our nodes subscribed with the default `RELIABLE` QoS.
In ROS 2 a `RELIABLE` subscriber and a `BEST_EFFORT` publisher are incompatible —
no messages flow and no error is raised beyond the warning.

**Fix**
Changed all MAVROS-sourced subscriptions to `QoSPresetProfiles.SENSOR_DATA`
(which is `BEST_EFFORT`, depth 10 — matching MAVROS). Subscriptions to internal
topics published by our own nodes (TERCOM fix, quality, ESKF odom, state, health)
remain `RELIABLE` (`depth=10`).

Files changed: `nodes/tercom_node.py`, `nodes/eskf_node.py`, `nodes/diagnostics_node.py`.

---

## Issue 3 — `/scan` (LaserScan) used instead of `sensor_msgs/Range`

**Symptom**
The rangefinder in the simulation publishes on `/scan` as `sensor_msgs/msg/LaserScan`
(single-beam lidar), not on the expected `sensor_msgs/msg/Range` topic.

Sample message:
```
ranges: [191.82302856445312]   # single element
range_min: 0.07999999821186066
range_max: 1000.0
```

**Root cause**
The original implementation assumed a `sensor_msgs/Range` rangefinder. The actual
sensor in the Gazebo model is a single-beam GPU lidar that publishes `LaserScan`.

**Fix**
- Changed import from `Range` to `LaserScan` in `tercom_node.py`.
- Changed `message_filters.Subscriber` type accordingly.
- Extracted AGL range as `range_msg.ranges[0]` with validity guards:
  - empty list check
  - `math.isfinite()` check
  - `range_min ≤ value ≤ range_max` bounds check
- Updated `distance_sensor` remapping in both launch files from
  `/target/mavros/distance_sensor/rangefinder_pub` → `/scan`.

---

## Issue 4 — TERCOM matching always rejects: `disc=1.01` with huge MAD

**Symptom**
```
TERCOM match REJECTED (#15, 100% rejection rate): MAD=61.6m disc=1.01 rough=10.4m
```
All rejections failing `discrimination_min=1.5`. `search_r=10px` always.

**Root cause — two problems:**

**Problem A: Search radius always clamped to minimum (10px = 59m).**
The dynamic search radius is `3.0 × sigma_pos / pixel_size`. At startup the ESKF
position covariance is small (velocity updates keep it tight after GPS init), so the
computed radius was always below `search_radius_min_pixels=10`. The true position
was 324m away — **the correct location was never inside the 59m search window**.

**Problem B: `discrimination_min=1.5` is unachievable for straight-line flight.**
TERCOM discrimination = `second_best_MAD / best_MAD`. A straight-line profile
perfectly constrains along-track position but leaves cross-track ambiguous — every
candidate perpendicular to the flight path gives the same MAD. So discrimination is
naturally ~1.0 even when the match is excellent, because the second-best candidate
is just one pixel off in the cross-track direction.

**Fix**
- Raised `search_radius_min_pixels` from 10 → 80 (475m floor for taif at 5.93m/px).
- Raised `search_radius_pixels` (static fallback) from 50 → 200 px.
- Lowered `discrimination_min` from 1.5 → 1.02 (filters only true ambiguity where
  best = second-best; `roughness_min` remains the primary flat-terrain guard).
- Raised `roughness_min` from 3.0 → 5.0m to compensate.

---

## Issue 5 — TERCOM matching rejects good matches: `MAD=5m` rejected

**Symptom**
```
TERCOM match REJECTED (#1, 100%): MAD=5.0m (thresh=30.0) disc=1.02 (min=1.5) rough=38.4m
```
MAD and roughness are excellent but discrimination still fails.

**Root cause**
Same as Issue 4 Problem B. The `discrimination_min` threshold had not yet been
lowered at this point in the session. Confirmed the root cause by observing that
MAD grew monotonically across rejections (5m → 14m → 32m → 52m) because each
rejected match allowed the ESKF to drift further, degrading the next predicted
position and making subsequent matches worse.

**Fix**
Applied the `discrimination_min: 1.02` change from Issue 4.

---

## Issue 6 — Constant ~324m offset between estimated and ground truth paths in RViz

**Symptom**
`H_err: rms=311m max=312m` from the very first sample, even at ESKF initialization.
In RViz the two paths run in parallel with a fixed offset.

**Root cause**
The two paths are in different coordinate frames:
- `eskf_node/odom` publishes position in **local ENU** with origin at the Gazebo
  world GPS point (`world_origin_lat/lon/alt` → converted to UTM, used as ENU zero).
- MAVROS `local_position/odom` (ground truth) publishes position in **PX4's local
  NED frame**, whose origin is wherever the drone was when PX4 armed/booted.

These two origins are completely unrelated. Subtracting them directly — as the
original `diagnostics_node` did — produces a constant error equal to the distance
between the two frame origins, which happened to be ~324m in this scenario.

**Fix**
Added one-time frame alignment in `diagnostics_node`:
- When the ESKF first transitions to `RUNNING`, record `offset = est_pos - gt_pos`.
  At this moment the ESKF was just initialized from GPS, so `est ≈ true_pos_in_ENU`.
- Apply the offset to all ground truth positions before computing errors:
  `gt_aligned = gt_pos + offset`.
- Retroactively shift all previously collected ground truth path poses.
- Error computation and CSV logging both use the aligned ground truth.
- The alignment offset is logged once:
  `Frame alignment: MAVROS→ENU offset = (X.X, Y.Y, Z.Z) m`

---

## Issue 7 — `taif_test4_params.yaml` did not contain all parameters

**Symptom**
When launching with `params_file:=taif_test4_params.yaml`, parameters not
explicitly listed in that file fell back to node source defaults
(`declare_parameter(...)`) rather than the tuned values in `tercom_params.yaml`.
This caused unintended behavior (e.g. wrong search radius, wrong discrimination
threshold) when the taif file was sparse.

**Root cause**
The launch file treated `params_file` as a replacement for `tercom_params.yaml`.
Any parameter absent from the user-supplied file was not read from any YAML at all.

**Fix**
Made `taif_test4_params.yaml` a **complete, self-contained copy** of
`tercom_params.yaml` with all taif-specific values changed in place. A scenario
YAML now lists every parameter, so there is no hidden dependency on another file.

The convention is:
- `tercom_params.yaml` — generic baseline, used when no `params_file` is passed.
- `<scenario>_params.yaml` — complete copy, edited for that scenario. Fully
  replaces the default when passed as `params_file`.

---

## Diagnostic Commands

```bash
# Check which state TERCOM is in (WAITING_SENSORS / COLLECTING / MATCHING)
ros2 topic echo /tercom/tercom_node/status --once

# Check if fixes are being published
ros2 topic hz /tercom/tercom_node/position_fix

# See match quality on every attempt
ros2 topic echo /tercom/tercom_node/match_quality

# Check ESKF filter state
ros2 topic echo /tercom/eskf_node/state --once

# Check ESKF health metrics [avg_NIS, max_pos_std, innov_norm, is_healthy]
ros2 topic echo /tercom/eskf_node/health --once

# Verify sensor topics are arriving
ros2 topic hz /target/mavros/imu/data
ros2 topic hz /target/mavros/altitude
ros2 topic hz /scan
ros2 topic hz /target/mavros/local_position/odom

# Check what parameters a node actually received
ros2 param list /tercom/tercom_node
ros2 param get /tercom/tercom_node search_radius_min_pixels
ros2 param get /tercom/tercom_node discrimination_min

# Force a match attempt with current samples
ros2 service call /tercom/tercom_node/trigger_match std_srvs/srv/Trigger {}

# Reset the profile buffer
ros2 service call /tercom/tercom_node/reset_profile std_srvs/srv/Trigger {}

# Hard-reset the ESKF (re-acquires GPS)
ros2 service call /tercom/eskf_node/reset_filter std_srvs/srv/Trigger {}
```

---

## Key Parameter Relationships

| Parameter | Effect | taif_test4 value |
|---|---|---|
| `search_radius_min_pixels` | Floor for dynamic search window. Must be large enough to contain the true position during cold-start ESKF drift. `min_px × pixel_size` ≥ expected cold-start error. | 80 px (≈475m) |
| `search_radius_pixels` | Static radius used when dynamic is off or as initial value. | 200 px (≈1186m) |
| `discrimination_min` | Ratio threshold. Set to 1.02 for straight-line flight where cross-track is inherently ambiguous. Raise to 1.5+ only if the vehicle flies curved paths. | 1.02 |
| `roughness_min` | Primary guard against matching on flat terrain. Compensates for the lowered discrimination threshold. | 5.0 m |
| `tercom_noise_base` | ESKF measurement noise for TERCOM fixes. Set to one DEM pixel size so the filter trusts fixes to within one pixel. | 5.93 m |
| `world_origin_lat/lon/alt` | Must match the `spherical_coordinates` block in the Gazebo world SDF exactly. Wrong values shift the entire ENU frame, placing the predicted UTM position outside the DEM. | 21.2651°N, 40.3542°E, 1859.7m |
