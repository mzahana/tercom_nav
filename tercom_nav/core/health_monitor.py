"""ESKF Filter Health Monitor.

Three independent health checks:
1. NIS (Normalized Innovation Squared): tracks filter consistency
   - NIS = y^T * S^{-1} * y should follow chi-squared(dof=2) for 2D TERCOM updates
   - 95th percentile threshold = 5.99; use 15.0 as conservative default
2. Covariance bound: position std above max_position_std -> lost
3. Innovation gate: ||innovation|| > max_innovation -> outlier/diverged
"""
import numpy as np
import collections
from typing import Tuple


class HealthMonitor:
    def __init__(self, nis_threshold: float = 15.0, nis_window: int = 10,
                 max_position_std: float = 500.0, max_innovation: float = 200.0,
                 consecutive_reject_limit: int = 5):
        """
        Args:
            nis_threshold: Avg NIS above this signals divergence
            nis_window: Number of recent NIS values to average
            max_position_std: Position std threshold in meters
            max_innovation: Innovation magnitude threshold in meters
            consecutive_reject_limit: Consecutive rejections before divergence
        """
        self.nis_history: collections.deque = collections.deque(maxlen=nis_window)
        self.nis_threshold = nis_threshold
        self.max_position_std = max_position_std
        self.max_innovation = max_innovation
        self.consecutive_rejects = 0
        self.consecutive_reject_limit = consecutive_reject_limit

    def check_innovation(self, innovation: np.ndarray, S: np.ndarray) -> dict:
        """Check a TERCOM measurement before applying it.

        Args:
            innovation: [dx, dy] innovation vector (meters)
            S: 2x2 innovation covariance matrix

        Returns:
            dict with keys:
              'accept': bool - whether to apply this measurement
              'nis': float - Normalized Innovation Squared value
              'gated': bool - True if innovation magnitude was too large
        """
        innov_norm = float(np.linalg.norm(innovation))
        # NIS = y^T * S^{-1} * y
        try:
            nis = float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            nis = float('inf')

        self.nis_history.append(nis)

        gated = innov_norm > self.max_innovation
        if gated:
            self.consecutive_rejects += 1
        else:
            self.consecutive_rejects = 0

        return {
            'accept': not gated,
            'nis': nis,
            'gated': gated,
        }

    def check_covariance(self, P: np.ndarray) -> dict:
        """Check if position covariance has grown beyond safe bounds.

        Args:
            P: 15x15 error-state covariance matrix

        Returns:
            dict with 'healthy': bool, 'max_pos_std': float
        """
        pos_std = np.sqrt(np.maximum(np.diag(P)[:3], 0.0))
        max_std = float(np.max(pos_std))
        return {
            'healthy': max_std < self.max_position_std,
            'max_pos_std': max_std,
        }

    def is_diverged(self) -> Tuple[bool, str]:
        """Overall divergence check combining NIS and consecutive rejection.

        Returns:
            (diverged: bool, reason: str)
        """
        if len(self.nis_history) >= self.nis_history.maxlen:
            avg_nis = float(np.mean(self.nis_history))
            if avg_nis > self.nis_threshold:
                return True, f"Average NIS {avg_nis:.1f} > threshold {self.nis_threshold}"

        if self.consecutive_rejects >= self.consecutive_reject_limit:
            return True, (
                f"Consecutive innovation rejections: {self.consecutive_rejects} "
                f">= limit {self.consecutive_reject_limit}"
            )

        return False, ""

    def get_avg_nis(self) -> float:
        """Return current average NIS (0.0 if no history yet)."""
        if not self.nis_history:
            return 0.0
        return float(np.mean(self.nis_history))

    def reset(self) -> None:
        """Clear all history after filter reset."""
        self.nis_history.clear()
        self.consecutive_rejects = 0
