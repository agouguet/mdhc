from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def launch_setup(context, *args, **kwargs):
    num_instances = int(LaunchConfiguration('num_instances').perform(context))
    speed_time = float(LaunchConfiguration('speed_time').perform(context))
    log_level = LaunchConfiguration('log_level')

    nodes = []

    # ROS TCP
    nodes.append(Node(
        package='ros_tcp_endpoint',
        executable='default_server_endpoint',
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    ))

    nodes.append(Node(
        package='mdhc',
        executable='rl_trainer_node.py',
        parameters=[{'num_envs': num_instances, 'speed_time': speed_time}],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'num_instances',
            default_value='1',
            description='Number of agent instances to launch'
        ),
        DeclareLaunchArgument(
            'speed_time',
            default_value='1',
            description='Speed Time'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='error',
            description='Log level: debug, info, warn, error, fatal'
        ),
        OpaqueFunction(function=launch_setup)
    ])
