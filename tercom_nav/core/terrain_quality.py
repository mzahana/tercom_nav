"""Terrain quality assessment for TERCOM matching.

Roughness = std of DEM elevations in a search window.
Low roughness means flat terrain -> unreliable matches.
"""
import numpy as np


def compute_roughness(dem_array: np.ndarray, center_row: int, center_col: int,
                      radius_px: int, nodata_value: float = -9999.0) -> float:
    """Compute terrain roughness (std of elevation) in a window.

    Args:
        dem_array: 2D elevation array (rows x cols), float32
        center_row: Center pixel row index
        center_col: Center pixel column index
        radius_px: Half-width of window in pixels
        nodata_value: Sentinel value for missing data

    Returns:
        Standard deviation of valid elevations in window.
        Returns 0.0 if fewer than 4 valid pixels found.
    """
    h, w = dem_array.shape
    r0 = max(0, center_row - radius_px)
    r1 = min(h, center_row + radius_px + 1)
    c0 = max(0, center_col - radius_px)
    c1 = min(w, center_col + radius_px + 1)

    window = dem_array[r0:r1, c0:c1]
    valid = window[window > nodata_value]

    if len(valid) < 4:
        return 0.0
    return float(np.std(valid))


def classify_terrain_quality(roughness: float, roughness_min: float,
                              discrimination: float, discrimination_min: float,
                              mad: float, mad_threshold: float) -> str:
    """Classify terrain match quality.

    Args:
        roughness: Terrain std in search window (meters)
        roughness_min: Minimum roughness to consider (meters)
        discrimination: Ratio of 2nd-best to best MAD (> 1 is good)
        discrimination_min: Minimum acceptable discrimination ratio
        mad: Best MAD value (meters)
        mad_threshold: Maximum acceptable MAD (meters)

    Returns:
        'good'     - all metrics pass with margin
        'marginal' - borderline; some metrics barely pass
        'poor'     - one or more metrics fail
    """
    if roughness < roughness_min or discrimination < discrimination_min or mad > mad_threshold:
        return 'poor'

    # Good = comfortably above thresholds
    good_roughness = roughness > roughness_min * 2.0
    good_disc = discrimination > discrimination_min * 1.5
    good_mad = mad < mad_threshold * 0.5

    if good_roughness and good_disc and good_mad:
        return 'good'
    return 'marginal'


def compute_adaptive_noise(mad: float, discrimination: float, roughness: float,
                           pixel_size: float, base_noise: float) -> float:
    """Compute adaptive TERCOM measurement noise (sigma^2) for the ESKF.

    Higher quality matches produce lower noise (more trust).
    Formula from Section 2.3 of the implementation plan:
      R_base = pixel_size^2
      roughness_factor  = clamp(roughness / roughness_nominal, 0.5, 3.0)
      discrimination_factor = clamp(discrimination / disc_nominal, 0.5, 3.0)
      mad_factor = mad / mad_nominal
      R_tercom = R_base * max(1.0, mad_factor) / (roughness_factor * discrimination_factor)

    Args:
        mad: Best MAD value (meters)
        discrimination: 2nd_best / best MAD ratio
        roughness: Terrain std in search window (meters)
        pixel_size: DEM pixel size in meters (auto-detected)
        base_noise: Base sigma in meters (-1 = use pixel_size)

    Returns:
        Measurement noise variance (meters^2) for ESKF R matrix diagonal
    """
    if base_noise < 0:
        base_noise = pixel_size

    R_base = base_noise ** 2

    roughness_nominal = 10.0   # meters - expected typical roughness
    disc_nominal = 2.0         # expected typical discrimination
    mad_nominal = 5.0          # meters - expected typical MAD

    roughness_factor = float(np.clip(roughness / roughness_nominal, 0.5, 3.0))
    disc_factor = float(np.clip(discrimination / disc_nominal, 0.5, 3.0))
    mad_factor = mad / mad_nominal

    noise_var = R_base * max(1.0, mad_factor) / (roughness_factor * disc_factor)
    return float(noise_var)
