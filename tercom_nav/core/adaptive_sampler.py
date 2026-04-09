"""Adaptive terrain sampling based on distance traveled.

Instead of fixed-rate sampling, triggers when the drone has moved
a configurable number of DEM pixels since the last sample.

Speed behavior (pixel_size=5.93m, pixels_per_sample=1.5, target=8.9m):
  2 m/s  -> sample every ~4.4 s  (~23 samples over 200m)
  10 m/s -> sample every ~0.9 s  (~23 samples over 200m)
  20 m/s -> sample every 0.5 s   (min_interval clamp, ~23 over 200m)
"""
import numpy as np
from typing import Optional


class AdaptiveSampler:
    def __init__(self, pixel_size_m: float, pixels_per_sample: float = 1.5,
                 min_interval_s: float = 0.5, max_interval_s: float = 5.0):
        """
        Args:
            pixel_size_m: DEM pixel size in meters (auto-detected from DEM)
            pixels_per_sample: Target spacing in DEM pixels per sample
            min_interval_s: Minimum time between samples (seconds)
            max_interval_s: Maximum time between samples (seconds)
        """
        self.target_distance = pixel_size_m * pixels_per_sample
        self.min_interval = min_interval_s
        self.max_interval = max_interval_s
        self.last_sample_position: Optional[np.ndarray] = None
        self.last_sample_time: Optional[float] = None

    def should_sample(self, position_enu: np.ndarray, current_time_s: float) -> bool:
        """Check if enough distance has been traveled to warrant a new sample.

        Args:
            position_enu: Current [x, y, z] in local ENU meters
            current_time_s: Current time in seconds (from ROS header stamp)

        Returns:
            True if a terrain sample should be collected now
        """
        if self.last_sample_position is None or self.last_sample_time is None:
            return True  # Always collect the first sample

        dt = current_time_s - self.last_sample_time
        dist = np.linalg.norm(position_enu[:2] - self.last_sample_position[:2])

        if dt < self.min_interval:
            return False  # Too soon

        if dt >= self.max_interval:
            return True  # Collect even if stationary (max time exceeded)

        return bool(dist >= self.target_distance)  # Distance-based trigger

    def record_sample(self, position_enu: np.ndarray, current_time_s: float) -> None:
        """Record that a sample was just taken at this position and time."""
        self.last_sample_position = position_enu[:2].copy()
        self.last_sample_time = current_time_s

    def reset(self) -> None:
        """Clear state for re-initialization."""
        self.last_sample_position = None
        self.last_sample_time = None

    @property
    def target_distance_m(self) -> float:
        """Target spacing in meters."""
        return self.target_distance
