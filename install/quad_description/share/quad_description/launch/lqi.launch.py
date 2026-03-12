import os
import time
import tempfile
import subprocess

from ament_index_python.packages import get_package_share_directory


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction
from launch_ros.actions import Node
import xacro

def generate_launch_description():

    package_name = "quad_description"

    lqi_node = Node(
        package="quad_description",
        executable="LQI.py",
    )


    visualize = Node(
        package="quad_description",
        executable="trajectory_visualizer.py",
    )

    gazebo_sim = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(get_package_share_directory(package_name), "launch", "sim.launch.py")
                ),
                launch_arguments={"use_sim_time": "true"}.items()
            )
        ]
    )


    trajectory = TimerAction(
        period=0.0,
        actions=[
            Node(
                package="quad_description",
                executable="send_traject.py",
                output="screen",
                parameters=[{'use_sim_time': True}],
            )
        ]
    )

    # Create LaunchDescription
    launch_description = LaunchDescription()
    # Add launch actions
    # launch_description.add_action(rviz)
    launch_description.add_action(lqi_node)
    launch_description.add_action(gazebo_sim)
    launch_description.add_action(visualize)
    launch_description.add_action(trajectory)


    return launch_description