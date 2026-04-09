"""Unit tests for TERCOM matcher.

Uses a synthetic sinusoidal DEM for deterministic testing.
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS


def make_synthetic_dem(tmp_path, width=200, height=200):
    """Create a sinusoidal DEM with known structure."""
    dem_path = str(tmp_path / 'synth.tif')
    west, south, east, north = 500000.0, 2300000.0, 501200.0, 2301200.0
    transform = from_bounds(west, south, east, north, width, height)
    pixel_size = (east - west) / width  # 6.0 m

    x = np.linspace(0, 8 * np.pi, width)
    y = np.linspace(0, 8 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    elev = (1800.0 + 80.0 * np.sin(xx) + 60.0 * np.cos(yy)).astype(np.float32)

    with rasterio.open(dem_path, 'w', driver='GTiff',
                       height=height, width=width, count=1,
                       dtype=np.float32, crs=CRS.from_epsg(32637),
                       transform=transform, nodata=-9999.0) as dst:
        dst.write(elev, 1)
    return dem_path, transform, pixel_size, elev


class TestProfileCollector:
    def test_basic_collection(self):
        from tercom_nav.core.tercom_matcher import ProfileCollector
        pc = ProfileCollector(min_spacing_m=5.0, max_samples=5)
        positions = np.array([[i * 10.0, 0.0, 0.0] for i in range(6)])
        full = False
        for i, pos in enumerate(positions):
            full = pc.try_add_sample(1800.0 + i, pos, float(i))
            if full:
                break
        assert full
        assert pc.num_samples == 5

    def test_spacing_enforced(self):
        from tercom_nav.core.tercom_matcher import ProfileCollector
        pc = ProfileCollector(min_spacing_m=10.0, max_samples=5)
        pos = np.array([0.0, 0.0, 0.0])
        # Sample 1 always accepted (first)
        pc.try_add_sample(1800.0, pos, 0.0)
        assert pc.num_samples == 1
        # Sample too close - rejected
        pc.try_add_sample(1801.0, np.array([5.0, 0.0, 0.0]), 1.0)
        assert pc.num_samples == 1
        # Sample far enough - accepted
        pc.try_add_sample(1802.0, np.array([12.0, 0.0, 0.0]), 2.0)
        assert pc.num_samples == 2

    def test_slide_window(self):
        from tercom_nav.core.tercom_matcher import ProfileCollector
        pc = ProfileCollector(min_spacing_m=5.0, max_samples=10)
        for i in range(10):
            pc.try_add_sample(1800.0, np.array([i * 10.0, 0.0, 0.0]), float(i))
        assert pc.num_samples == 10
        pc.slide_window()
        assert pc.num_samples == 5  # kept N/2 = 5

    def test_reset(self):
        from tercom_nav.core.tercom_matcher import ProfileCollector
        pc = ProfileCollector(min_spacing_m=5.0, max_samples=5)
        pc.try_add_sample(1800.0, np.array([0.0, 0.0, 0.0]), 0.0)
        pc.reset()
        assert pc.num_samples == 0

    def test_get_profile_arrays_shape(self):
        from tercom_nav.core.tercom_matcher import ProfileCollector
        pc = ProfileCollector(min_spacing_m=5.0, max_samples=5)
        for i in range(3):
            pc.try_add_sample(1800.0 + i, np.array([i * 10.0, 0.0, 0.0]), float(i))
        h, dx, dy, ts = pc.get_profile_arrays()
        assert h.shape == (3,)
        assert dx.shape == (3,)
        assert dy.shape == (3,)
        assert ts.shape == (3,)
        # First sample displacement should be ~0
        assert abs(dx[0]) < 1e-6
        assert abs(dy[0]) < 1e-6


class TestMatchProfile:
    def test_recovers_known_position(self, tmp_path):
        """Plant a known position, generate noiseless profile, verify recovery."""
        from tercom_nav.core.tercom_matcher import match_profile

        dem_path, transform, pixel_size, elev = make_synthetic_dem(tmp_path)

        # True position: row=100, col=100 in the DEM
        true_row, true_col = 100, 100
        true_easting = float(transform.c + true_col * transform.a)
        true_northing = float(transform.f + true_row * transform.e)

        # Generate 10 sample profile along east direction
        N = 10
        step_m = pixel_size * 1.5  # 1.5 pixels per sample
        dx_m = np.arange(N) * step_m
        dy_m = np.zeros(N)

        # DEM elevations at true positions (noiseless)
        sample_cols = true_col + (dx_m / pixel_size).astype(int)
        sample_rows = true_row + np.zeros(N, dtype=int)
        sample_cols = np.clip(sample_cols, 0, elev.shape[1] - 1)
        terrain_h = elev[sample_rows, sample_cols].astype(np.float64)

        result = match_profile(
            dem_array=elev,
            transform=transform,
            pixel_size_x=pixel_size,
            pixel_size_y=pixel_size,
            terrain_h=terrain_h,
            dx_m=dx_m,
            dy_m=dy_m,
            predicted_utm=(true_easting, true_northing),
            search_radius_px=15,
        )

        assert result['valid'] is True
        assert result['mad'] < 2.0  # noiseless -> very low MAD

        # Position error should be < 1.5 pixels
        err_e = abs(result['utm'][0] - true_easting)
        err_n = abs(result['utm'][1] - true_northing)
        assert err_e < pixel_size * 1.5, f"Easting error {err_e:.1f} m > 1.5 pixels"
        assert err_n < pixel_size * 1.5, f"Northing error {err_n:.1f} m > 1.5 pixels"

    def test_flat_terrain_low_discrimination(self, tmp_path):
        """Flat terrain should produce discrimination ~1.0 (ambiguous)."""
        from tercom_nav.core.tercom_matcher import match_profile

        flat_path = str(tmp_path / 'flat.tif')
        width, height = 100, 100
        west, south, east, north = 500000.0, 2300000.0, 500600.0, 2300600.0
        transform = from_bounds(west, south, east, north, width, height)
        flat_elev = np.full((height, width), 1800.0, dtype=np.float32)

        with rasterio.open(flat_path, 'w', driver='GTiff',
                           height=height, width=width, count=1,
                           dtype=np.float32, crs=CRS.from_epsg(32637),
                           transform=transform, nodata=-9999.0) as dst:
            dst.write(flat_elev, 1)

        pixel_size = 6.0
        N = 10
        terrain_h = np.full(N, 1800.0)
        dx_m = np.arange(N) * pixel_size
        dy_m = np.zeros(N)
        pred_utm = (500300.0, 2300300.0)

        result = match_profile(
            dem_array=flat_elev, transform=transform,
            pixel_size_x=pixel_size, pixel_size_y=pixel_size,
            terrain_h=terrain_h, dx_m=dx_m, dy_m=dy_m,
            predicted_utm=pred_utm, search_radius_px=20,
        )
        # Flat terrain: discrimination should be near 1.0 (no distinctive peak)
        assert result['discrimination'] < 2.0, (
            f"Expected low discrimination on flat terrain, got {result['discrimination']:.2f}"
        )

    def test_shape_mismatch_raises(self, tmp_path):
        from tercom_nav.core.tercom_matcher import match_profile
        dem_path, transform, pixel_size, elev = make_synthetic_dem(tmp_path)
        with pytest.raises(ValueError):
            match_profile(
                dem_array=elev, transform=transform,
                pixel_size_x=pixel_size, pixel_size_y=pixel_size,
                terrain_h=np.ones(5), dx_m=np.ones(3), dy_m=np.ones(5),
                predicted_utm=(500600.0, 2300600.0), search_radius_px=10,
            )
