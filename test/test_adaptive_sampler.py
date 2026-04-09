"""Unit tests for adaptive sampler and health monitor."""
import numpy as np
import pytest


class TestAdaptiveSampler:
    def test_first_sample_always_accepted(self):
        from tercom_nav.core.adaptive_sampler import AdaptiveSampler
        s = AdaptiveSampler(pixel_size_m=5.93)
        assert s.should_sample(np.array([0., 0., 0.]), 0.0) is True

    def test_min_interval_enforced(self):
        from tercom_nav.core.adaptive_sampler import AdaptiveSampler
        s = AdaptiveSampler(pixel_size_m=5.93, min_interval_s=0.5)
        s.record_sample(np.array([0., 0., 0.]), 0.0)
        # Too soon (0.1 s < 0.5 s min_interval)
        assert s.should_sample(np.array([100., 0., 0.]), 0.1) is False

    def test_max_interval_triggers_even_stationary(self):
        from tercom_nav.core.adaptive_sampler import AdaptiveSampler
        s = AdaptiveSampler(pixel_size_m=5.93, max_interval_s=5.0)
        s.record_sample(np.array([0., 0., 0.]), 0.0)
        # After max_interval, should sample even if stationary
        assert s.should_sample(np.array([0., 0., 0.]), 5.1) is True

    def test_distance_trigger(self):
        from tercom_nav.core.adaptive_sampler import AdaptiveSampler
        pixel_size = 5.93
        pixels_per_sample = 1.5
        s = AdaptiveSampler(pixel_size_m=pixel_size, pixels_per_sample=pixels_per_sample,
                            min_interval_s=0.01)
        target_dist = pixel_size * pixels_per_sample
        s.record_sample(np.array([0., 0., 0.]), 0.0)
        # Just under target: no sample
        assert s.should_sample(np.array([target_dist - 0.1, 0., 0.]), 1.0) is False
        # At or over target: sample
        assert s.should_sample(np.array([target_dist + 0.1, 0., 0.]), 1.0) is True

    def test_reset_clears_state(self):
        from tercom_nav.core.adaptive_sampler import AdaptiveSampler
        s = AdaptiveSampler(pixel_size_m=5.93)
        s.record_sample(np.array([0., 0., 0.]), 0.0)
        s.reset()
        # After reset, first sample is always accepted
        assert s.should_sample(np.array([0., 0., 0.]), 0.5) is True


class TestHealthMonitor:
    def test_no_divergence_with_small_innovations(self):
        from tercom_nav.core.health_monitor import HealthMonitor
        hm = HealthMonitor(nis_threshold=15.0, nis_window=10)
        S = np.eye(2) * 100.0
        for _ in range(10):
            hm.check_innovation(np.array([1.0, 1.0]), S)
        diverged, _ = hm.is_diverged()
        assert diverged is False

    def test_divergence_after_consecutive_large_innovations(self):
        from tercom_nav.core.health_monitor import HealthMonitor
        hm = HealthMonitor(max_innovation=10.0, consecutive_reject_limit=3)
        S = np.eye(2) * 100.0
        for _ in range(4):
            hm.check_innovation(np.array([50.0, 0.0]), S)  # norm=50 > 10 -> gated
        diverged, reason = hm.is_diverged()
        assert diverged is True

    def test_covariance_check(self):
        from tercom_nav.core.health_monitor import HealthMonitor
        hm = HealthMonitor(max_position_std=100.0)
        P_small = np.eye(15) * 1.0
        result = hm.check_covariance(P_small)
        assert result['healthy'] is True

        P_large = np.eye(15) * 200.0**2  # std = 200 > 100
        result = hm.check_covariance(P_large)
        assert result['healthy'] is False

    def test_reset_clears_history(self):
        from tercom_nav.core.health_monitor import HealthMonitor
        hm = HealthMonitor(nis_threshold=5.0, nis_window=3, max_innovation=10.0,
                           consecutive_reject_limit=2)
        S = np.eye(2) * 100.0
        for _ in range(3):
            hm.check_innovation(np.array([50.0, 0.0]), S)
        diverged, _ = hm.is_diverged()
        assert diverged is True
        hm.reset()
        diverged, _ = hm.is_diverged()
        assert diverged is False
