"""DEM Manager - loads any GeoTIFF and provides elevation lookups.

The DEMManager auto-detects CRS and pixel size from the GeoTIFF affine
transform. If the DEM is in a geographic CRS (EPSG:4326), it reprojects
to the appropriate UTM zone automatically using rasterio.warp.

Usage:
    dem = DEMManager('/path/to/dem.tif')
    elev = dem.get_elevation(easting=500000, northing=2350000)
    batch = dem.get_elevation_batch(eastings, northings)
"""
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pyproj import CRS
from typing import Optional

logger = logging.getLogger(__name__)


class DEMManager:
    """Loads a GeoTIFF DEM and provides fast elevation lookups.

    All elevation lookups are in UTM coordinates. If the source DEM
    is in a geographic CRS, it is reprojected to UTM at load time.

    Attributes:
        elevation: np.ndarray - 2D float32 elevation array [rows, cols]
        transform: rasterio.Affine - pixel-to-UTM affine transform
        crs: rasterio.CRS - projected UTM CRS
        pixel_size_x: float - pixel width in meters (transform.a)
        pixel_size_y: float - pixel height in meters (|transform.e|)
        bounds: dict - {'west', 'east', 'south', 'north'} in UTM meters
        elevation_range: tuple - (min_elev, max_elev) in meters
        nodata_value: float - sentinel for missing data
        width: int - number of DEM columns
        height: int - number of DEM rows
    """

    def __init__(self, dem_path: str, nodata_value: float = -9999.0):
        """Load DEM from file path.

        Args:
            dem_path: Absolute path to the .tif DEM file
            nodata_value: Sentinel for missing elevation data

        Raises:
            FileNotFoundError: If dem_path does not exist
            ValueError: If the file cannot be read as a valid DEM
        """
        import os
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"DEM file not found: {dem_path}")

        self.nodata_value = nodata_value
        self._load(dem_path)

    def _load(self, dem_path: str) -> None:
        """Internal: load DEM, reproject if needed, extract metadata."""
        with rasterio.open(dem_path) as src:
            src_crs = src.crs
            src_transform = src.transform
            src_data = src.read(1).astype(np.float32)
            src_nodata = src.nodata

            # Replace source nodata with our sentinel
            if src_nodata is not None:
                src_data[src_data == src_nodata] = self.nodata_value

            # Check if we need to reproject (geographic -> UTM)
            if src_crs.is_geographic:
                logger.info(
                    f"DEM CRS is geographic ({src_crs.to_epsg()}), reprojecting to UTM"
                )
                # Find UTM zone from center of DEM
                bounds = src.bounds
                center_lon = (bounds.left + bounds.right) / 2
                center_lat = (bounds.bottom + bounds.top) / 2
                zone_number = int((center_lon + 180) / 6) + 1
                epsg = 32600 + zone_number if center_lat >= 0 else 32700 + zone_number
                dst_crs = CRS.from_epsg(epsg)

                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, *src.bounds
                )
                dst_data = np.full(
                    (dst_height, dst_width), self.nodata_value, dtype=np.float32
                )

                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=self.nodata_value,
                    dst_nodata=self.nodata_value,
                )

                self.elevation = dst_data
                self.transform = dst_transform
                self.crs = dst_crs
                self.height, self.width = dst_data.shape
            else:
                self.elevation = src_data
                self.transform = src_transform
                self.crs = src_crs
                self.height, self.width = src_data.shape

        # Extract pixel size from affine transform
        # For north-up rasters: transform.a = pixel_width (positive)
        #                        transform.e = -pixel_height (negative)
        self.pixel_size_x = abs(float(self.transform.a))
        self.pixel_size_y = abs(float(self.transform.e))

        # Compute bounds in UTM meters
        # Top-left corner is at (transform.c, transform.f)
        west = float(self.transform.c)
        north = float(self.transform.f)
        east = west + self.width * float(self.transform.a)
        south = north + self.height * float(self.transform.e)  # e is negative
        self.bounds = {
            'west': west, 'east': east,
            'south': south, 'north': north,
        }

        # Elevation statistics (excluding nodata)
        valid_mask = self.elevation > self.nodata_value
        if valid_mask.any():
            valid_elevs = self.elevation[valid_mask]
            self.elevation_range = (float(valid_elevs.min()), float(valid_elevs.max()))
        else:
            self.elevation_range = (0.0, 0.0)

        logger.info(
            f"DEM loaded: CRS={self.crs.to_epsg()}, "
            f"size={self.width}x{self.height}, "
            f"pixel={self.pixel_size_x:.2f}x{self.pixel_size_y:.2f} m, "
            f"elev=[{self.elevation_range[0]:.1f}, {self.elevation_range[1]:.1f}] m"
        )

    def get_elevation(self, easting: float, northing: float,
                      method: str = 'bilinear') -> float:
        """Get elevation at a single UTM point.

        Args:
            easting: UTM easting in meters
            northing: UTM northing in meters
            method: 'nearest' or 'bilinear'

        Returns:
            Elevation in meters MSL. Returns nodata_value if out of bounds.
        """
        # Convert UTM to pixel coordinates
        col = (easting - self.transform.c) / self.transform.a
        row = (northing - self.transform.f) / self.transform.e

        if method == 'nearest':
            r = int(round(row))
            c = int(round(col))
            if not (0 <= r < self.height and 0 <= c < self.width):
                return self.nodata_value
            val = self.elevation[r, c]
            return self.nodata_value if val <= self.nodata_value else float(val)

        else:  # bilinear
            r0 = int(np.floor(row))
            c0 = int(np.floor(col))
            r1, c1 = r0 + 1, c0 + 1

            # Check bounds
            if not (0 <= r0 < self.height and 0 <= c0 < self.width):
                return self.nodata_value

            # Clamp r1, c1 to valid range for edge pixels
            r1 = min(r1, self.height - 1)
            c1 = min(c1, self.width - 1)

            dr = row - r0
            dc = col - c0

            q00 = float(self.elevation[r0, c0])
            q01 = float(self.elevation[r0, c1])
            q10 = float(self.elevation[r1, c0])
            q11 = float(self.elevation[r1, c1])

            # Invalidate if any corner is nodata
            if any(q <= self.nodata_value for q in [q00, q01, q10, q11]):
                return self.nodata_value

            # Bilinear interpolation
            val = (q00 * (1 - dc) * (1 - dr) +
                   q01 * dc * (1 - dr) +
                   q10 * (1 - dc) * dr +
                   q11 * dc * dr)
            return float(val)

    def get_elevation_batch(self, eastings: np.ndarray, northings: np.ndarray,
                            method: str = 'nearest') -> np.ndarray:
        """Get elevations at multiple UTM points. Fully vectorized.

        Args:
            eastings: 1D array of UTM eastings (meters)
            northings: 1D array of UTM northings (meters)
            method: 'nearest' (fast, for TERCOM hot path) or 'bilinear'

        Returns:
            1D array of elevations (meters). Out-of-bounds -> nodata_value.
        """
        eastings = np.asarray(eastings, dtype=np.float64)
        northings = np.asarray(northings, dtype=np.float64)

        cols = (eastings - self.transform.c) / self.transform.a
        rows = (northings - self.transform.f) / self.transform.e

        result = np.full(len(eastings), self.nodata_value, dtype=np.float32)

        if method == 'nearest':
            r_int = np.round(rows).astype(np.int32)
            c_int = np.round(cols).astype(np.int32)
            valid = (r_int >= 0) & (r_int < self.height) & \
                    (c_int >= 0) & (c_int < self.width)
            r_int_v = np.clip(r_int, 0, self.height - 1)
            c_int_v = np.clip(c_int, 0, self.width - 1)
            vals = self.elevation[r_int_v, c_int_v]
            vals[~valid] = self.nodata_value
            result = vals.astype(np.float32)

        else:  # bilinear (slower but more accurate)
            for i in range(len(eastings)):
                result[i] = self.get_elevation(eastings[i], northings[i], method='bilinear')

        return result

    def is_in_bounds(self, easting: float, northing: float) -> bool:
        """Check if a UTM point falls within the DEM extent.

        Args:
            easting: UTM easting in meters
            northing: UTM northing in meters

        Returns:
            True if the point is within the DEM bounding box
        """
        return (self.bounds['west'] <= easting <= self.bounds['east'] and
                self.bounds['south'] <= northing <= self.bounds['north'])

    def get_info(self) -> dict:
        """Return DEM metadata as a dictionary.

        Returns:
            dict with keys: crs_epsg, pixel_size_x, pixel_size_y, bounds,
            width, height, elevation_range, nodata_value
        """
        return {
            'crs_epsg': self.crs.to_epsg(),
            'pixel_size_x': self.pixel_size_x,
            'pixel_size_y': self.pixel_size_y,
            'bounds': self.bounds,
            'width': self.width,
            'height': self.height,
            'elevation_range': self.elevation_range,
            'nodata_value': self.nodata_value,
        }
