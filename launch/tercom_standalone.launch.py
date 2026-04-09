"""Standalone TERCOM launch - only tercom_node + eskf_node.

Use when MAVROS is already running and you don't need diagnostics.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    dem_file = LaunchConfiguration('dem_file').perform(context)
    mavros_ns = LaunchConfiguration('mavros_ns').perform(context)
    lat0 = LaunchConfiguration('world_origin_lat').perform(context)
    lon0 = LaunchConfiguration('world_origin_lon').perform(context)
    alt0 = LaunchConfiguration('world_origin_alt').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)

    sim_time_bool = use_sim_time.lower() in ('true', '1', 'yes')
    pkg_share = get_package_share_directory('tercom_nav')
    params_file = os.path.join(pkg_share, 'config', 'tercom_params.yaml')

    world_params = {
        'world_origin_lat': float(lat0),
        'world_origin_lon': float(lon0),
        'world_origin_alt': float(alt0),
        'use_sim_time': sim_time_bool,
    }

    tercom_node = Node(
        package='tercom_nav', executable='tercom_node',
        name='tercom_node', namespace='tercom', output='screen',
        parameters=[params_file, {'dem_file': dem_file, **world_params}],
        remappings=[
            ('altitude',         f'/{mavros_ns}/altitude'),
            ('eskf_odom',        '/tercom/eskf_node/odom'),
            ('imu_data',         f'/{mavros_ns}/imu/data'),
            ('distance_sensor',  '/scan'),
            ('eskf_covariance',  '/tercom/eskf/pose'),
        ],
    )

    eskf_node = Node(
        package='tercom_nav', executable='eskf_node',
        name='eskf_node', namespace='tercom', output='screen',
        parameters=[params_file, {**world_params}],
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

    return [tercom_node, eskf_node]


def generate_launch_description():
    pkg_share = get_package_share_directory('tercom_nav')
    default_params = os.path.join(pkg_share, 'config', 'tercom_params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('dem_file', default_value=''),
        DeclareLaunchArgument('mavros_ns', default_value='target/mavros'),
        DeclareLaunchArgument('world_origin_lat', default_value='0.0'),
        DeclareLaunchArgument('world_origin_lon', default_value='0.0'),
        DeclareLaunchArgument('world_origin_alt', default_value='0.0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
