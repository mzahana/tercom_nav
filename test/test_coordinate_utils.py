"""Unit tests for coordinate frame conversions."""
import numpy as np
import pytest


class TestLatLonToUTM:
    def test_taif_area(self):
        from tercom_nav.core.coordinate_utils import latlon_to_utm
        e, n, zone, letter = latlon_to_utm(21.265, 40.354)
        assert zone == 37
        assert letter == 'N'
        assert 500000 < e < 800000  # UTM Zone 37N typical range
        assert 2300000 < n < 2400000

    def test_round_trip(self):
        from tercom_nav.core.coordinate_utils import latlon_to_utm, utm_to_latlon
        lat_in, lon_in = 21.265, 40.354
        e, n, zone, letter = latlon_to_utm(lat_in, lon_in)
        lat_out, lon_out = utm_to_latlon(e, n, zone, letter)
        assert abs(lat_out - lat_in) < 1e-6
        assert abs(lon_out - lon_in) < 1e-6

    def test_southern_hemisphere(self):
        from tercom_nav.core.coordinate_utils import latlon_to_utm
        e, n, zone, letter = latlon_to_utm(-33.8688, 151.2093)  # Sydney
        assert letter == 'S'


class TestUTMtoENU:
    def test_origin_maps_to_zero(self):
        from tercom_nav.core.coordinate_utils import utm_to_local_enu
        e0, n0, alt0 = 700000.0, 2350000.0, 1859.7
        result = utm_to_local_enu(e0, n0, alt0, e0, n0, alt0)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-6)

    def test_east_offset(self):
        from tercom_nav.core.coordinate_utils import utm_to_local_enu
        e0, n0, alt0 = 700000.0, 2350000.0, 0.0
        result = utm_to_local_enu(e0 + 100.0, n0, alt0, e0, n0, alt0)
        np.testing.assert_allclose(result, [100.0, 0.0, 0.0], atol=1e-6)

    def test_round_trip(self):
        from tercom_nav.core.coordinate_utils import utm_to_local_enu, local_enu_to_utm
        e0, n0, alt0 = 700000.0, 2350000.0, 1859.7
        enu = utm_to_local_enu(700150.0, 2350300.0, 1900.0, e0, n0, alt0)
        e, n, alt = local_enu_to_utm(enu[0], enu[1], enu[2], e0, n0, alt0)
        assert abs(e - 700150.0) < 1e-4
        assert abs(n - 2350300.0) < 1e-4
        assert abs(alt - 1900.0) < 1e-4


class TestPixelConversions:
    def test_utm_to_pixel_and_back(self):
        from tercom_nav.core.coordinate_utils import utm_to_pixel, pixel_to_utm

        class MockTransform:
            a = 5.93    # pixel width (east)
            e = -5.93   # pixel height (negative = north-up)
            c = 700000.0  # west edge
            f = 2352000.0  # north edge

        t = MockTransform()
        e, n = 700059.3, 2351940.7  # 10 pixels from top-left
        col, row = utm_to_pixel(e, n, t)
        assert abs(col - 10.0) < 0.1
        assert abs(row - 10.0) < 0.1

        e_back, n_back = pixel_to_utm(col, row, t)
        assert abs(e_back - e) < 0.01
        assert abs(n_back - n) < 0.01
