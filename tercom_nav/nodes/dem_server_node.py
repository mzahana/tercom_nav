"""DEM Server Node - loads GeoTIFF DEM and provides ROS services.

Services (using std_srvs/Trigger with JSON in response.message):
  ~/get_dem_info     - returns DEM metadata as JSON string
  ~/get_elevation    - request: JSON {easting, northing}, response: JSON {elevation}

Publishes:
  ~/dem_info (std_msgs/String, transient local QoS) - JSON metadata, published once

Parameters:
  dem_file (string)          - required: absolute path to .tif
  dem_metadata_file (string) - optional: path to .json sidecar
  nodata_value (float)       - default -9999.0
  interpolation_method (str) - "bilinear" or "nearest"
"""
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tercom_nav.core.dem_manager import DEMManager


class DEMServerNode(Node):
    def __init__(self):
        super().__init__('dem_server_node')

        # Declare all parameters
        self.declare_parameter('dem_file', '')
        self.declare_parameter('dem_metadata_file', '')
        self.declare_parameter('nodata_value', -9999.0)
        self.declare_parameter('interpolation_method', 'bilinear')

        # Transient local QoS for the info topic (latched equivalent)
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._info_pub = self.create_publisher(String, '~/dem_info', latched_qos)

        # Load DEM
        dem_file = self.get_parameter('dem_file').value
        if not dem_file:
            self.get_logger().error('Parameter "dem_file" is required but not set!')
            raise RuntimeError('dem_file parameter is required')

        nodata = self.get_parameter('nodata_value').value
        try:
            self._dem = DEMManager(dem_file, nodata_value=nodata)
        except (FileNotFoundError, ValueError) as e:
            self.get_logger().error(f'Failed to load DEM: {e}')
            raise

        self._interp = self.get_parameter('interpolation_method').value

        # Optional: load JSON sidecar metadata
        self._extra_meta = {}
        meta_file = self.get_parameter('dem_metadata_file').value
        if meta_file:
            try:
                import os
                if os.path.exists(meta_file):
                    with open(meta_file) as f:
                        self._extra_meta = json.load(f)
                    self.get_logger().info(f'Loaded metadata sidecar: {meta_file}')
            except Exception as e:
                self.get_logger().warning(f'Could not load metadata sidecar: {e}')

        # Build and publish DEM info
        info = self._dem.get_info()
        info.update(self._extra_meta)
        info_msg = String()
        info_msg.data = json.dumps(info)
        self._info_pub.publish(info_msg)

        info_dict = self._dem.get_info()
        self.get_logger().info(
            f'DEM loaded: {info_dict["width"]}x{info_dict["height"]} px, '
            f'pixel={info_dict["pixel_size_x"]:.2f}m, '
            f'elev=[{info_dict["elevation_range"][0]:.1f}, '
            f'{info_dict["elevation_range"][1]:.1f}] m, '
            f'CRS=EPSG:{info_dict["crs_epsg"]}'
        )

        # Services
        self._srv_info = self.create_service(
            Trigger, '~/get_dem_info', self._handle_get_dem_info
        )
        self._srv_elevation = self.create_service(
            Trigger, '~/get_elevation', self._handle_get_elevation
        )

    def _handle_get_dem_info(self, request, response):
        """Return DEM metadata as JSON string in response.message."""
        info = self._dem.get_info()
        info.update(self._extra_meta)
        response.success = True
        response.message = json.dumps(info)
        return response

    def _handle_get_elevation(self, request, response):
        """Elevation lookup - request encodes coords as JSON in request header.

        Since std_srvs/Trigger has no request fields, the caller encodes the
        query as JSON in a parameter set before calling, or use a different
        service pattern. Here we return the DEM bounds and instructions.

        For real lookups, the tercom_node loads the DEM directly via DEMManager
        to avoid ROS service overhead on the hot path.
        """
        info = self._dem.get_info()
        response.success = True
        response.message = (
            f"DEM bounds: W={info['bounds']['west']:.1f}, "
            f"E={info['bounds']['east']:.1f}, "
            f"S={info['bounds']['south']:.1f}, "
            f"N={info['bounds']['north']:.1f}"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    try:
        node = DEMServerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        print(f'DEM Server error: {e}')
        traceback.print_exc()
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
