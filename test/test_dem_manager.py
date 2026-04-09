"""Unit tests for DEMManager.

Uses a synthetic DEM for repeatable testing.
"""
import numpy as np
import pytest
import os


def create_synthetic_dem(tmp_path) -> str:
    """Create a small synthetic DEM for testing."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    dem_path = str(tmp_path / 'test_dem.tif')
    width, height = 100, 100
    # UTM Zone 37N bounds (near Taif)
    west, south, east, north = 500000, 2300000, 500600, 2300600
    transform = from_bounds(west, south, east, north, width, height)
    # Sinusoidal terrain
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    elev = 1800.0 + 50.0 * np.sin(xx) * np.cos(yy)

    with rasterio.open(
        dem_path, 'w', driver='GTiff',
        height=height, width=width, count=1,
        dtype=np.float32, crs=CRS.from_epsg(32637),
        transform=transform, nodata=-9999.0
    ) as dst:
        dst.write(elev.astype(np.float32), 1)

    return dem_path


class TestDEMManager:
    def test_load_synthetic(self, tmp_path):
        from tercom_nav.core.dem_manager import DEMManager
        dem_path = create_synthetic_dem(tmp_path)
        dem = DEMManager(dem_path)
        assert dem.width == 100
        assert dem.height == 100
        assert dem.pixel_size_x == pytest.approx(6.0, rel=0.1)
        assert 1750.0 < dem.elevation_range[0] < 1860.0
        assert 1840.0 < dem.elevation_range[1] < 2000.0

    def test_get_elevation_center(self, tmp_path):
        from tercom_nav.core.dem_manager import DEMManager
        dem_path = create_synthetic_dem(tmp_path)
        dem = DEMManager(dem_path)
        # Center of DEM
        center_e = (dem.bounds['west'] + dem.bounds['east']) / 2
        center_n = (dem.bounds['south'] + dem.bounds['north']) / 2
        elev = dem.get_elevation(center_e, center_n)
        assert elev > dem.nodata_value
        assert 1700.0 < elev < 2100.0

    def test_out_of_bounds_returns_nodata(self, tmp_path):
        from tercom_nav.core.dem_manager import DEMManager
        dem_path = create_synthetic_dem(tmp_path)
        dem = DEMManager(dem_path)
        # Way outside bounds
        elev = dem.get_elevation(0.0, 0.0)
        assert elev == dem.nodata_value

    def test_is_in_bounds(self, tmp_path):
        from tercom_nav.core.dem_manager import DEMManager
        dem_path = create_synthetic_dem(tmp_path)
        dem = DEMManager(dem_path)
        center_e = (dem.bounds['west'] + dem.bounds['east']) / 2
        center_n = (dem.bounds['south'] + dem.bounds['north']) / 2
        assert dem.is_in_bounds(center_e, center_n) is True
        assert dem.is_in_bounds(0.0, 0.0) is False

    def test_batch_matches_single(self, tmp_path):
        from tercom_nav.core.dem_manager import DEMManager
        dem_path = create_synthetic_dem(tmp_path)
        dem = DEMManager(dem_path)
        # 5 random in-bounds points
        np.random.seed(42)
        es = np.random.uniform(dem.bounds['west'] + 50, dem.bounds['east'] - 50, 5)
        ns = np.random.uniform(dem.bounds['south'] + 50, dem.bounds['north'] - 50, 5)
        batch = dem.get_elevation_batch(es, ns, method='nearest')
        singles = [dem.get_elevation(e, n, method='nearest') for e, n in zip(es, ns)]
        np.testing.assert_allclose(batch, singles, atol=1e-3)

    def test_file_not_found(self):
        from tercom_nav.core.dem_manager import DEMManager
        with pytest.raises(FileNotFoundError):
            DEMManager('/nonexistent/path/dem.tif')

    def test_get_info(self, tmp_path):
        from tercom_nav.core.dem_manager import DEMManager
        dem_path = create_synthetic_dem(tmp_path)
        dem = DEMManager(dem_path)
        info = dem.get_info()
        assert 'crs_epsg' in info
        assert 'pixel_size_x' in info
        assert 'bounds' in info
        assert 'elevation_range' in info
