"""Coordinate frame conversion utilities for TERCOM navigation.

Supported frames:
- WGS84 geographic (lat/lon degrees)
- UTM projected (easting/northing meters, auto-detected zone)
- Local ENU (East-North-Up, origin at Gazebo world origin)
- DEM pixel (row/col indices, float for sub-pixel precision)

All pyproj.Transformer objects are cached at module level for efficiency.
"""
import numpy as np
from pyproj import CRS, Transformer
from typing import Tuple

# Cache for Transformer objects — creation is expensive, reuse is cheap
_transformer_cache: dict = {}


def _get_transformer(from_crs: str, to_crs: str) -> Transformer:
    """Get or create a cached pyproj Transformer."""
    key = (from_crs, to_crs)
    if key not in _transformer_cache:
        _transformer_cache[key] = Transformer.from_crs(
            from_crs, to_crs, always_xy=True
        )
    return _transformer_cache[key]


def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float, int, str]:
    """Convert WGS84 lat/lon to UTM easting/northing.

    Auto-detects UTM zone from longitude. Uses pyproj for accurate conversion.

    Args:
        lat: Latitude in decimal degrees (positive = North)
        lon: Longitude in decimal degrees (positive = East)

    Returns:
        (easting, northing, zone_number, zone_letter)
        easting/northing in meters
    """
    zone_number = int((lon + 180) / 6) + 1
    zone_letter = 'N' if lat >= 0 else 'S'
    epsg = 32600 + zone_number if lat >= 0 else 32700 + zone_number
    transformer = _get_transformer('EPSG:4326', f'EPSG:{epsg}')
    easting, northing = transformer.transform(lon, lat)
    return easting, northing, zone_number, zone_letter


def utm_to_latlon(easting: float, northing: float,
                  zone_number: int, zone_letter: str) -> Tuple[float, float]:
    """Convert UTM easting/northing to WGS84 lat/lon.

    Args:
        easting: UTM easting in meters
        northing: UTM northing in meters
        zone_number: UTM zone number (1-60)
        zone_letter: 'N' for northern hemisphere, 'S' for southern

    Returns:
        (latitude, longitude) in decimal degrees
    """
    epsg = 32600 + zone_number if zone_letter == 'N' else 32700 + zone_number
    transformer = _get_transformer(f'EPSG:{epsg}', 'EPSG:4326')
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


def utm_to_local_enu(easting: float, northing: float, alt_msl: float,
                     origin_easting: float, origin_northing: float,
                     origin_alt: float) -> np.ndarray:
    """Convert UTM + MSL altitude to local ENU coordinates.

    ENU origin is at (origin_easting, origin_northing, origin_alt).

    Args:
        easting: UTM easting in meters
        northing: UTM northing in meters
        alt_msl: Altitude above MSL in meters
        origin_easting: UTM easting of ENU origin
        origin_northing: UTM northing of ENU origin
        origin_alt: MSL altitude of ENU origin in meters

    Returns:
        np.ndarray([x_east, y_north, z_up]) in meters
    """
    x = easting - origin_easting
    y = northing - origin_northing
    z = alt_msl - origin_alt
    return np.array([x, y, z], dtype=np.float64)


def local_enu_to_utm(x: float, y: float, z: float,
                     origin_easting: float, origin_northing: float,
                     origin_alt: float) -> Tuple[float, float, float]:
    """Convert local ENU to UTM + MSL altitude.

    Args:
        x: East displacement from ENU origin (meters)
        y: North displacement from ENU origin (meters)
        z: Up displacement from ENU origin (meters)
        origin_easting: UTM easting of ENU origin
        origin_northing: UTM northing of ENU origin
        origin_alt: MSL altitude of ENU origin in meters

    Returns:
        (easting, northing, alt_msl) in meters
    """
    easting = origin_easting + x
    northing = origin_northing + y
    alt_msl = origin_alt + z
    return easting, northing, alt_msl


def utm_to_pixel(easting: float, northing: float,
                 transform) -> Tuple[float, float]:
    """Convert UTM coordinates to DEM pixel (col, row).

    Uses the rasterio Affine transform. For a north-up raster:
      col = (easting  - transform.c) / transform.a
      row = (northing - transform.f) / transform.e  (transform.e is negative)

    Args:
        easting: UTM easting in meters
        northing: UTM northing in meters
        transform: rasterio.transform.Affine object

    Returns:
        (col, row) as floats with sub-pixel precision
    """
    col = (easting - transform.c) / transform.a
    row = (northing - transform.f) / transform.e
    return col, row


def pixel_to_utm(col: float, row: float, transform) -> Tuple[float, float]:
    """Convert DEM pixel (col, row) to UTM coordinates.

    For a north-up raster:
      easting  = transform.c + col * transform.a
      northing = transform.f + row * transform.e

    Args:
        col: Column index (float for sub-pixel)
        row: Row index (float for sub-pixel)
        transform: rasterio.transform.Affine object

    Returns:
        (easting, northing) in UTM meters
    """
    easting = transform.c + col * transform.a
    northing = transform.f + row * transform.e
    return easting, northing


def compute_utm_origin(lat: float, lon: float, alt: float) -> dict:
    """Convert a geographic world origin to UTM for use as ENU origin.

    Args:
        lat: World origin latitude (degrees)
        lon: World origin longitude (degrees)
        alt: World origin altitude MSL (meters)

    Returns:
        dict with keys: 'easting', 'northing', 'alt', 'zone_number', 'zone_letter', 'epsg'
    """
    easting, northing, zone_num, zone_let = latlon_to_utm(lat, lon)
    epsg = 32600 + zone_num if zone_let == 'N' else 32700 + zone_num
    return {
        'easting': easting,
        'northing': northing,
        'alt': alt,
        'zone_number': zone_num,
        'zone_letter': zone_let,
        'epsg': epsg,
    }
