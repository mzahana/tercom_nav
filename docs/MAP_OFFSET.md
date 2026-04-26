## How to find the offset of the map center

To find the offset and elevation information for a DEM world, you can use the `gps_to_enu.py` script.
Gazebo applies a positional offset (`<pos>`) to the heightmap, meaning the physical center of the heightmap image does not sit exactly at the `(0, 0)` origin of the world.

For example, to find the offset of the map center for taif_test4, first installed the missing module pymap3d and gathered the center coordinates of the map from taif_test4.sdf. The taif_test4 center WGS84 coordinates are:

From taif_test4.sdf:
```xml
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>21.265059957990964</latitude_deg>
      <longitude_deg>40.354156494140625</longitude_deg>
      <elevation>1859.7000000000007</elevation>
    </spherical_coordinates>
```

Latitude: 21.265059957990964
Longitude: 40.354156494140625

One can then run the script using the center coordinates as the target to determine the ground elevation and positional offset of the map center. Here is the command and the output it generated:

```bash
cd /home/user/shared_volume/ros2_ws/src/gps_denied_navigation_sim
python3 scripts/gps_to_enu.py \
  --center-lat 21.265059957990964 \
  --center-lon 40.354156494140625 \
  --target-lat 21.265059957990964 \
  --target-lon 40.354156494140625 \
  --tif-file models/taif_test4/textures/taif_test4_height_map.tif
```

Outputs the following crucial information:
```text
Map physical dimensions (from SDF): 2708.27m x 2692.29m x 285.9m
Map pos offset (from SDF): 71.26m, -70.86m, 0.0m
Center Elevation: 125.258 m
Target Elevation: 125.258 m

Corresponding ENU coordinates (Z is absolute elevation):
X (East) : 0.000000
Y (North): 0.000000
Z (Elev) : 125.257661
```

### Understanding the Offsets

There are two separate concepts here:
1. **World Origin (`0, 0`)**: The location of the spherical coordinates defined in the SDF. At `X=0, Y=0`, the ground elevation is `125.26m`.
2. **Heightmap Center (`71.26, -70.86`)**: Gazebo physically shifts the center of the heightmap mesh by `71.26m` in X and `-70.86m` in Y. At this specific location, the ground elevation corresponds to the exact center pixel of the `.tif` image, which happens to be `~75.9m`.

## What to modify based on the offset information

### 1. Spawning the UAV (`gps_denied_navigation_sim/launch/dem.launch.py`)
To spawn the drone safely above the ground, you must match the `zpos` to the ground elevation at your chosen `xpos` and `ypos` (plus a buffer).

**Option A: Spawn at the heightmap center**
Ground elevation here is `75.9m`. Adding a small buffer, we can spawn at `Z=76.0m`.
```python
    elif world_type == 'taif_test4':
        xpos, ypos, zpos = '71.26', '-70.86', '76.0'
```

**Option B: Spawn at the World Origin**
Ground elevation here is `125.26m`. Adding a small buffer, we can spawn at `Z=126m`.
```python
    elif world_type == 'taif_test4':
        xpos, ypos, zpos = '0.0', '0.0', '126.0'
```

### 2. TERCOM Point Cloud Visualization (`tercom_nav/config/taif_test4_params.yaml`)

```yaml
    dem_pos_offset: [-352.895, 72.263, 134.3]
```

`diagnostics_node` synthesises the DEM pointcloud from the `.tif` as

```python
xs = (eas - origin.easting) + dem_pos_offset[0]
ys = (nos - origin.northing) + dem_pos_offset[1]
zs = (els - origin.alt)      + dem_pos_offset[2]
```

where `(eas, nos)` are UTM coordinates from the DEM's georeferencing and `origin` is the `world_origin_{lat,lon,alt}` converted to UTM. For the cloud to land on top of the Gazebo terrain we need

```
dem_pos_offset_xy = heightmap.<pos>_xy  −  (DEM_center_UTM − world_origin_UTM)
dem_pos_offset_z  = world_origin_alt   −  DEM_min_elev
```

For `taif_test4`:

| Quantity                                         | Value                       |
| ------------------------------------------------ | --------------------------- |
| `world_origin_UTM` (EPSG:32637)                  | `(640500.097, 2352089.709)` |
| `DEM_center_UTM` (from `.tif` transform)         | `(640924.251, 2351946.587)` |
| `DEM_center_UTM − world_origin_UTM`              | `(+424.155, −143.123)`      |
| `heightmap.<pos>_xy` (taif_test4/model.sdf)      | `(71.26, −70.86)`           |
| `dem_pos_offset_xy`                              | `(−352.895, +72.263)`       |
| `world_origin_alt − DEM_min_elev = 1859.7 − 1725.4` | `134.3`                  |

> **Caveat — heightmap placement vs. DEM georeferencing.** The Gazebo SDF puts the heightmap mesh at `<pos>71.26 −70.86 0</pos>` (via the world's `<spherical_coordinates>` origin), but the DEM `.tif` is georeferenced so that its centre sits ≈`(+424 m E, −143 m N)` from that same origin. Gazebo ignores the `.tif`'s UTM transform and simply places the mesh at `<pos>`, so the rendered terrain and the georeferenced DEM are physically offset by ≈`(352 m, 72 m)`. `dem_pos_offset_xy` compensates for that mismatch in visualisation. If TERCOM indexes the `.tif` by GPS it will look up the wrong patch of terrain; a permanent fix requires either re‑georeferencing the `.tif` or moving `<spherical_coordinates>` so that the heightmap `<pos>` equals `(DEM_center_UTM − world_origin_UTM)`.

### 3. Static `map → target/odom` TF (`gps_denied_navigation_sim/launch/dem.launch.py`)

The Gazebo GPS plugin uses the world's `<spherical_coordinates><elevation>` as its altitude reference, **not** the heightmap's `DEM_min` shift. Consequently, at spawn:

- GPS altitude = `world_origin_alt + spawn_z_gazebo`.
- PX4 home altitude is set from that first fix, so MAVROS `local_position.z = 0` at spawn.
- The ESKF converts GPS to ENU with `z = gps_alt − world_origin_alt`, so **`ESKF.z ≡ gazebo_z`** (the `world_origin_alt` term cancels).

This means the `map` frame's Z convention is plain Gazebo Z (i.e. `map.z = 0 ⇔ DEM_min_elev MSL`). For `target/base_link` (MAVROS chain) to land at the same `z = gazebo_z` as the ESKF estimate and the DEM pointcloud, the static TF must lift `target/odom` by `spawn_z`:

```python
# gps_denied_navigation_sim/launch/dem.launch.py
map_odom_z = float(zpos)   # spawn_z; for taif_test4 → 76.4 m
map2pose_tf_node = Node(
    ...,
    arguments=[xpos, ypos, str(map_odom_z), '0', '0', '0', 'map', 'target/odom'],
)
```

Verification (taif_test4, drone at spawn):

| Quantity                           | Value in `map` |
| ---------------------------------- | -------------- |
| `target/base_link` via MAVROS TF   | `76.4 m`       |
| `/tercom/eskf_node/odom.pose.z`    | `≈ 76.4 m`     |
| DEM heightmap-center pixel (≈1801 MSL) | `≈ 75.9 m` |

All three agree, and flying the drone preserves the alignment because both MAVROS local_z and ESKF z track `gazebo_z − spawn_z` and `gazebo_z` respectively, differing by a constant `spawn_z` that is absorbed by `map_odom_z`.

## How to find the ground elevation for any arbitrary spawn coordinate

If you want to spawn the UAV at a specific `(X, Y)` location in Gazebo that is neither the world origin nor the heightmap center, you must calculate the ground elevation at that specific coordinate so you don't spawn underground.

We have provided a script `gps_denied_navigation_sim/scripts/get_elevation_at_xy.py` to do exactly this. It reads the `<pos>` and `<size>` from the `model.sdf` and calculates the elevation of any given Gazebo coordinate by sampling the `.tif` file.

**Example usage:**
If you want to spawn the drone at `X=250.0`, `Y=-100.0` in `taif_test4`:
```bash
cd /home/user/shared_volume/ros2_ws/src/gps_denied_navigation_sim
python3 scripts/get_elevation_at_xy.py \
  --x 250.0 \
  --y -100.0 \
  --tif-file models/taif_test4/textures/taif_test4_height_map.tif
```

**Example Output:**
```text
Coordinate: X=250.0, Y=-100.0
Ground Elevation: Z=62.345 meters
Recommended Spawn Z: 62.8 meters (adds 0.5m buffer)
```

You can then update `dem.launch.py` with these exact values:
```python
    elif world_type == 'taif_test4':
        xpos, ypos, zpos = '250.0', '-100.0', '62.8'
```
