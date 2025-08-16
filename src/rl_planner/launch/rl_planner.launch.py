#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld_preload = SetEnvironmentVariable(
        'LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
    )
    
    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='sensor',
        description='Robot base frame ID'
    )

    sensor_range_arg = DeclareLaunchArgument(
        'sensor_range',
        default_value='20.0',
        description='Maximum sensor range in meters'
    )

    map_resolution_arg = DeclareLaunchArgument(
        'map_resolution',
        default_value='0.4',
        description='Map resolution in meters per cell'
    )

    node_resolution_arg = DeclareLaunchArgument(
        'node_resolution',
        default_value='2.0',
        description='Path planning node resolution in meters'
    )

    octomap_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap',
        output='screen',
        remappings=[('cloud_in', 'sensor_scan')],
        parameters=[
            {'frame_id': 'map'},
            {'base_frame_id': LaunchConfiguration('base_frame')},
            {'resolution': LaunchConfiguration('map_resolution')},
            {'occupancy_min_z': 0.0},
            {'occupancy_max_z': 1.2},
            {'sensor_model.max_range': LaunchConfiguration('sensor_range')},
            {'sensor_model.hit': 1.0},
            {'sensor_model.miss': 0.45},
            {'sensor_model.max': 1.0},
            {'sensor_model.min': 0.2},
        ]
    )

    rl_planner_node = Node(
        package='rl_planner',
        executable='rl_planner',
        name='rl_planner',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'publish_graph': True},
            {'node_resolution': LaunchConfiguration('node_resolution')},
            {'sensor_range': LaunchConfiguration('sensor_range')},
            {'utility_range_factor': 0.5},
            {'min_utility': 3},
            {'frontier_downsample_factor': 1},
            {'map_resolution': LaunchConfiguration('map_resolution')},
            {'waypoint_threshold': 2.0},
            {'next_waypoint_threshold': 4.0},
            {'hard_update_threshold': 10.0},
            {'frontier_cluster_range': 10.0},
            {'enable_save_mode': False},
            {'enable_dstarlite': False},
            {'replanning_frequency': 2.5}
        ]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rl_rviz',
        output='log',
        arguments=['-d', PathJoinSubstitution([get_package_share_directory('rl_planner'), 'rviz', 'rviz.rviz']), '--ros-args', '--log-level', 'warn'],
        respawn=True
    )

    return LaunchDescription([
        ld_preload,
        base_frame_arg,
        sensor_range_arg,
        map_resolution_arg,
        node_resolution_arg,
        octomap_node,
        rl_planner_node,
        rviz_node
    ])

