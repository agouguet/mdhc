from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Paths
    mdhc_dir = get_package_share_directory('mdhc')
    model_file_default = os.path.join(mdhc_dir, 'model', 'test')

    namespace = "env_0"

    return LaunchDescription([
        # Declare model file argument
        DeclareLaunchArgument(
            name='model_file',
            default_value=model_file_default,
            description='Path to trained DRL-VO model'
        ),

        Node(
            package='ros_tcp_endpoint',
            executable='default_server_endpoint',
            output='screen'
        ),

        Node(
            package='pkg-nav',
            executable='velocity_smoother_node.py',
            name=f'velocity_smoother',
            namespace=namespace,
            output='screen',
        ),

        Node(
            package='mdhc',
            executable='rl_inference_node.py',
            name='rl_inference',
            namespace=namespace,
            output='screen',
            parameters=[{
                'model_file': LaunchConfiguration('model_file')
            }]
        ),
    ])
