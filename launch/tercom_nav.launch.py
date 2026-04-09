"""Full TERCOM navigation system launch file.

Launches all four nodes with parameters and topic remappings.

Arguments:
  dem_file            - Path to .tif DEM file (required)
  dem_metadata_file   - Path to .json sidecar (default: '')
  mavros_ns           - MAVROS namespace prefix (default: 'target/mavros')
  world_origin_lat    - Gazebo world origin latitude (default: 0.0)
  world_origin_lon    - Gazebo world origin longitude (default: 0.0)
  world_origin_alt    - Gazebo world origin altitude MSL (default: 0.0)
  params_file         - Override params YAML (default: package config)
  use_sim_time        - Use simulation time (default: true)

Example:
  ros2 launch tercom_nav tercom_nav.launch.py \\
      dem_file:=/path/to/dem.tif \\
      world_origin_lat:=21.2651 \\
      world_origin_lon:=40.3542 \\
      world_origin_alt:=1859.7
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    dem_file = LaunchConfiguration('dem_file').perform(context)
    dem_meta = LaunchConfiguration('dem_metadata_file').perform(context)
    mavros_ns = LaunchConfiguration('mavros_ns').perform(context)
    lat0 = LaunchConfiguration('world_origin_lat').perform(context)
    lon0 = LaunchConfiguration('world_origin_lon').perform(context)
    alt0 = LaunchConfiguration('world_origin_alt').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    params_file = LaunchConfiguration('params_file').perform(context)

    sim_time_bool = use_sim_time.lower() in ('true', '1', 'yes')

    pkg_share = get_package_share_directory('tercom_nav')
    if not params_file:
        params_file = os.path.join(pkg_share, 'config', 'tercom_params.yaml')

    params_list = [params_file]

    # Build override dicts — only include values that were explicitly provided
    # via CLI args so that YAML file values are not silently clobbered by defaults.
    base_overrides = {'use_sim_time': sim_time_bool}

    dem_overrides = {}
    if dem_file:
        dem_overrides['dem_file'] = dem_file
    if dem_meta:
        dem_overrides['dem_metadata_file'] = dem_meta

    world_overrides = {}
    if lat0 != '__unset__':
        world_overrides['world_origin_lat'] = float(lat0)
    if lon0 != '__unset__':
        world_overrides['world_origin_lon'] = float(lon0)
    if alt0 != '__unset__':
        world_overrides['world_origin_alt'] = float(alt0)

    dem_server = Node(
        package='tercom_nav',
        executable='dem_server',
        name='dem_server_node',
        namespace='tercom',
        output='screen',
        parameters=[
            *params_list,
            {**base_overrides, **dem_overrides},
        ],
    )

    tercom_node = Node(
        package='tercom_nav',
        executable='tercom_node',
        name='tercom_node',
        namespace='tercom',
        output='screen',
        parameters=[
            *params_list,
            {**base_overrides, **dem_overrides, **world_overrides},
        ],
        remappings=[
            ('altitude',         f'/{mavros_ns}/altitude'),
            ('eskf_odom',        '/tercom/eskf_node/odom'),
            ('imu_data',         f'/{mavros_ns}/imu/data'),
            ('distance_sensor',  '/scan'),
            ('eskf_covariance',  '/tercom/eskf_node/pose'),
        ],
    )

    eskf_node = Node(
        package='tercom_nav',
        executable='eskf_node',
        name='eskf_node',
        namespace='tercom',
        output='screen',
        parameters=[
            *params_list,
            {**base_overrides, **world_overrides},
        ],
        remappings=[
            ('imu_data',       f'/{mavros_ns}/imu/data'),
            ('gps_global',     f'/{mavros_ns}/global_position/global'),
            ('altitude',       f'/{mavros_ns}/altitude'),
            ('velocity_local', f'/{mavros_ns}/local_position/velocity_local'),
            ('local_odom',     f'/{mavros_ns}/local_position/odom'),
            ('tercom_fix',     '/tercom/tercom_node/position_fix'),
            ('tercom_quality', '/tercom/tercom_node/match_quality'),
        ],
    )

    diagnostics_node = Node(
        package='tercom_nav',
        executable='diagnostics_node',
        name='diagnostics_node',
        namespace='tercom',
        output='screen',
        parameters=[
            *params_list,
            {**base_overrides, **dem_overrides, **world_overrides},
        ],
        remappings=[
            ('eskf_odom',           '/tercom/eskf_node/odom'),
            ('ground_truth_odom',   f'/{mavros_ns}/local_position/odom'),
            ('ground_truth_global', f'/{mavros_ns}/global_position/global'),
            ('tercom_fix',          '/tercom/tercom_node/position_fix'),
            ('tercom_quality',      '/tercom/tercom_node/match_quality'),
            ('eskf_state',          '/tercom/eskf_node/state'),
            ('eskf_health',         '/tercom/eskf_node/health'),
            ('rejected_fix',        '/tercom/tercom_node/rejected_fix'),
            ('rejection_reason',    '/tercom/tercom_node/rejection_reason'),
        ],
    )

    return [dem_server, tercom_node, eskf_node, diagnostics_node]


def generate_launch_description():
    pkg_share = get_package_share_directory('tercom_nav')
    default_params = os.path.join(pkg_share, 'config', 'tercom_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('dem_file', default_value='',
                              description='Absolute path to GeoTIFF DEM file'),
        DeclareLaunchArgument('dem_metadata_file', default_value='',
                              description='Optional .json sidecar metadata'),
        DeclareLaunchArgument('mavros_ns', default_value='target/mavros',
                              description='MAVROS namespace prefix'),
        DeclareLaunchArgument('world_origin_lat', default_value='__unset__',
                              description='Gazebo world origin latitude (overrides params_file if set)'),
        DeclareLaunchArgument('world_origin_lon', default_value='__unset__',
                              description='Gazebo world origin longitude (overrides params_file if set)'),
        DeclareLaunchArgument('world_origin_alt', default_value='__unset__',
                              description='Gazebo world origin altitude MSL (overrides params_file if set)'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation clock'),
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='Override parameters YAML file'),
        OpaqueFunction(function=launch_setup),
    ])
