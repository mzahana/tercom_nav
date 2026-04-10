"""TERCOM terrain profile collection and vectorized correlation matching.

TERCOM (TERrain COntour Matching) works by:
1. Collecting N terrain elevation measurements along a flight path
2. Searching for the position in the DEM where a synthetic profile
   best matches the measured profile (minimum MAD)
3. The best-match position is fed to the ESKF as a 2D position fix

All operations are vectorized with NumPy. No Python loops over candidates.
"""
import numpy as np
from typing import List, Optional, Tuple


class ProfileCollector:
    """Collects terrain elevation samples with timestamps and relative positions.

    Each sample stores:
    - terrain_h: measured terrain elevation (h_baro - h_AGL) in meters MSL
    - dx, dy: East/North displacement from first sample in local ENU meters
    - timestamp: float seconds from ROS Time (used for adaptive sampling)

    The buffer uses a sliding window: after a successful match,
    the oldest N/2 samples are kept for continuity.
    """

    def __init__(self, min_spacing_m: float, max_samples: int):
        """
        Args:
            min_spacing_m: Minimum distance between consecutive samples (meters)
            max_samples: Maximum number of samples before triggering a match
        """
        self.min_spacing = min_spacing_m
        self.max_samples = max_samples
        # Each entry: (terrain_h, dx, dy, timestamp)
        self._samples: List[Tuple[float, float, float, float]] = []
        self._last_position: Optional[np.ndarray] = None
        self._reference_position: Optional[np.ndarray] = None

    def try_add_sample(self, terrain_h: float, position_enu: np.ndarray,
                       timestamp: float) -> bool:
        """Attempt to add a terrain sample.

        Returns True when the buffer is full (profile ready to match).

        Args:
            terrain_h: Terrain elevation (h_baro - h_AGL) in meters MSL
            position_enu: Current position [x, y, z] in local ENU meters
            timestamp: Current time in seconds (from ROS header stamp)

        Returns:
            True if max_samples reached and profile is ready for matching
        """
        # Enforce minimum spacing
        if self._last_position is not None:
            dist = float(np.linalg.norm(position_enu[:2] - self._last_position[:2]))
            if dist < self.min_spacing:
                return False

        # Set reference position on first sample
        if self._reference_position is None:
            self._reference_position = position_enu[:2].copy()

        dx = float(position_enu[0] - self._reference_position[0])
        dy = float(position_enu[1] - self._reference_position[1])
        self._samples.append((float(terrain_h), dx, dy, float(timestamp)))
        self._last_position = position_enu[:2].copy()

        return len(self._samples) >= self.max_samples

    def get_profile_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract profile as NumPy arrays.

        Returns:
            (terrain_h, dx_m, dy_m, timestamps) - each shape (N,)
            terrain_h: measured terrain elevations (meters MSL)
            dx_m: East displacements from first sample (meters)
            dy_m: North displacements from first sample (meters)
            timestamps: sample times in seconds
        """
        if not self._samples:
            return (np.empty(0), np.empty(0), np.empty(0), np.empty(0))
        arr = np.array(self._samples, dtype=np.float64)
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    def slide_window(self) -> None:
        """Keep the newest N/2 samples after a successful match.

        This provides profile continuity: the next collection starts
        with half the previous samples already in the buffer.
        """
        keep = self.max_samples // 2
        if len(self._samples) > keep:
            self._samples = self._samples[-keep:]
            if self._samples:
                # Recompute displacements relative to the first kept sample
                first_dx = self._samples[0][1]
                first_dy = self._samples[0][2]
                recomputed = []
                for h, dx, dy, ts in self._samples:
                    recomputed.append((h, dx - first_dx, dy - first_dy, ts))
                self._samples = recomputed

                # Restore _reference_position to the absolute ENU position of the
                # first kept sample.  _last_position holds the absolute position of
                # the last sample; the last kept sample's (recomputed) dx/dy is the
                # displacement from first-kept to last-kept, so:
                #   first_kept_pos = last_pos - last_kept_displacement
                # Without this, new samples would compute dx=0 relative to their
                # own position instead of relative to first_kept, corrupting the
                # profile geometry for every subsequent match.
                if self._last_position is not None:
                    last_dx_new = self._samples[-1][1]
                    last_dy_new = self._samples[-1][2]
                    self._reference_position = np.array([
                        self._last_position[0] - last_dx_new,
                        self._last_position[1] - last_dy_new,
                    ])
                else:
                    self._reference_position = None  # safety fallback
                # _last_position remains valid: it is still the absolute position
                # of the last kept sample, used for spacing enforcement.

    def reset(self) -> None:
        """Clear all samples and reset state."""
        self._samples.clear()
        self._last_position = None
        self._reference_position = None

    @property
    def num_samples(self) -> int:
        """Number of samples currently collected."""
        return len(self._samples)

    @property
    def total_distance_m(self) -> float:
        """Total path length from first to last sample (meters)."""
        if len(self._samples) < 2:
            return 0.0
        first = self._samples[0]
        last = self._samples[-1]
        return float(np.hypot(last[1] - first[1], last[2] - first[2]))


def match_profile(dem_array: np.ndarray, transform,
                  pixel_size_x: float, pixel_size_y: float,
                  terrain_h: np.ndarray, dx_m: np.ndarray, dy_m: np.ndarray,
                  predicted_utm: Tuple[float, float], search_radius_px: int,
                  discrimination_exclusion_radius: int = 3,
                  nodata_value: float = -9999.0) -> dict:
    """Perform vectorized TERCOM matching of a terrain profile against a DEM.

    This is the computational core of TERCOM. All candidate positions are
    evaluated simultaneously using NumPy broadcasting - no Python loops.

    Args:
        dem_array: 2D elevation array (rows x cols), float32
        transform: rasterio.Affine pixel-to-UTM transform
        pixel_size_x: DEM pixel width in meters (|transform.a|)
        pixel_size_y: DEM pixel height in meters (|transform.e|)
        terrain_h: Measured terrain elevations, shape (N,), already datum-corrected
            (i.e., subtract dem_elevation_offset before passing here)
        dx_m: East displacements from first sample, shape (N,) in meters
        dy_m: North displacements from first sample, shape (N,) in meters
        predicted_utm: (easting, northing) predicted position of first sample
        search_radius_px: Half-width of search window in pixels
        discrimination_exclusion_radius: Pixels around best match excluded
            when computing second-best MAD (prevents adjacent-pixel bias)
        nodata_value: Sentinel for missing DEM data

    Returns:
        dict with keys:
            'utm': (easting, northing) of best match (first sample position)
            'mad': float - best MAD value in meters
            'discrimination': float - second_best_MAD / best_MAD (> 1 = good)
            'roughness': float - std of DEM in search window (meters)
            'valid': bool - True if enough valid DEM pixels were found

    Raises:
        ValueError: If terrain_h, dx_m, dy_m have mismatched shapes
    """
    if not (terrain_h.shape == dx_m.shape == dy_m.shape):
        raise ValueError(
            f"Shape mismatch: terrain_h={terrain_h.shape}, "
            f"dx_m={dx_m.shape}, dy_m={dy_m.shape}"
        )

    N = len(terrain_h)
    dem_h, dem_w = dem_array.shape

    # Convert meter displacements to pixel offsets
    # North-up raster: east -> col increases right, north -> row decreases
    dx_px = dx_m / pixel_size_x         # East displacement -> column offset
    dy_px = -dy_m / pixel_size_y        # North displacement -> row offset (inverted)

    # Predicted position in DEM pixel coordinates
    # transform.c = west edge, transform.f = north edge
    pred_col = (predicted_utm[0] - transform.c) / transform.a
    pred_row = (predicted_utm[1] - transform.f) / transform.e
    pred_col_int = int(round(pred_col))
    pred_row_int = int(round(pred_row))

    # Generate candidate grid (centered on predicted position)
    r = search_radius_px
    row_start = max(0, pred_row_int - r)
    row_end = min(dem_h, pred_row_int + r + 1)
    col_start = max(0, pred_col_int - r)
    col_end = min(dem_w, pred_col_int + r + 1)

    if row_end <= row_start or col_end <= col_start:
        return {
            'utm': predicted_utm, 'mad': np.inf,
            'discrimination': 1.0, 'roughness': 0.0, 'valid': False,
        }

    rows = np.arange(row_start, row_end)
    cols = np.arange(col_start, col_end)
    n_rows = len(rows)
    n_cols = len(cols)

    # Meshgrid of candidate positions, flattened
    cand_cols_2d, cand_rows_2d = np.meshgrid(cols, rows)
    candidate_rows = cand_rows_2d.ravel()   # (M,)
    candidate_cols = cand_cols_2d.ravel()   # (M,)

    # For each candidate, compute all N sample pixel positions
    # sample_rows[i, j] = candidate_rows[i] + dy_px[j]   shape (M, N)
    sample_rows = candidate_rows[:, None] + dy_px[None, :]   # (M, N)
    sample_cols = candidate_cols[:, None] + dx_px[None, :]   # (M, N)

    # Clip to valid DEM range (we'll mask true out-of-bounds separately)
    sample_rows_int = np.round(sample_rows).astype(np.int32)
    sample_cols_int = np.round(sample_cols).astype(np.int32)
    in_bounds = ((sample_rows_int >= 0) & (sample_rows_int < dem_h) &
                 (sample_cols_int >= 0) & (sample_cols_int < dem_w))

    # Clamp for safe indexing (masked below anyway)
    sample_rows_clipped = np.clip(sample_rows_int, 0, dem_h - 1)
    sample_cols_clipped = np.clip(sample_cols_int, 0, dem_w - 1)

    # Batch DEM lookup (M x N)
    dem_profiles = dem_array[sample_rows_clipped, sample_cols_clipped]   # (M, N)

    # Build validity mask: in-bounds AND not nodata
    valid_mask = in_bounds & (dem_profiles > nodata_value)    # (M, N)
    valid_count = valid_mask.sum(axis=1).astype(np.float32)   # (M,)

    # MAD over valid pixels only
    abs_diff = np.abs(dem_profiles.astype(np.float64) - terrain_h[None, :])
    abs_diff[~valid_mask] = 0.0
    sum_abs_diff = abs_diff.sum(axis=1)
    mad = np.where(
        valid_count > 0,
        sum_abs_diff / np.maximum(valid_count, 1.0),
        np.inf
    )

    # Reject candidates where fewer than 70% of samples have valid DEM data
    min_valid = N * 0.7
    mad[valid_count < min_valid] = np.inf

    # Find best match
    best_idx = int(np.argmin(mad))
    best_mad = float(mad[best_idx])

    # Compute discrimination: 2nd-best MAD / best MAD
    # Exclude neighbors within exclusion_radius of best match
    best_r_local = best_idx // n_cols   # local row index within candidate grid
    best_c_local = best_idx % n_cols    # local col index

    row_offsets = np.abs(np.arange(n_rows) - best_r_local)   # shape (n_rows,)
    col_offsets = np.abs(np.arange(n_cols) - best_c_local)   # shape (n_cols,)
    # meshgrid(col_offsets, row_offsets): first output varies along columns (col offset),
    # second output varies along rows (row offset) — both shape (n_rows, n_cols).
    col_grid, row_grid = np.meshgrid(col_offsets, row_offsets)
    neighbor_mask = (col_grid.ravel() <= discrimination_exclusion_radius) & \
                    (row_grid.ravel() <= discrimination_exclusion_radius)

    mad_for_disc = mad.copy()
    mad_for_disc[neighbor_mask] = np.inf
    second_best_mad = float(np.min(mad_for_disc))
    discrimination = second_best_mad / (best_mad + 1e-6)

    # Terrain roughness in search window
    search_patch = dem_array[row_start:row_end, col_start:col_end]
    valid_patch = search_patch[search_patch > nodata_value]
    roughness = float(np.std(valid_patch)) if len(valid_patch) > 0 else 0.0

    # Convert best pixel back to UTM
    best_col = int(candidate_cols[best_idx])
    best_row = int(candidate_rows[best_idx])
    best_easting = float(transform.c + best_col * transform.a)
    best_northing = float(transform.f + best_row * transform.e)

    return {
        'utm': (best_easting, best_northing),
        'mad': best_mad,
        'discrimination': discrimination,
        'roughness': roughness,
        'valid': best_mad < np.inf,
    }
