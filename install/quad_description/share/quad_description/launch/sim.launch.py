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

from launch_ros.actions import Node
import xacro

def generate_launch_description():

    package_name = "quad_description"
    rviz_file_name = "sim.rviz"


    spawn_x_val = "0.0"
    spawn_y_val = "0.0"
    spawn_z_val = "5.0" #"0.055" # "10.0" #

    # Paths
    rviz_file_path = os.path.join(get_package_share_directory(package_name), "rviz", rviz_file_name)


    default_world = os.path.join(
        get_package_share_directory(package_name),
        'worlds',
        'empty.sdf'
        )    

    world = LaunchConfiguration('world')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='World to load'
        )


    # Include Robot State Publisher
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package_name), "launch", "rsp.launch.py")
        ),
        launch_arguments={"use_sim_time": "true"}.items()
    )

    # Gazebo simulation launch
    gz_sim = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
                    launch_arguments={'gz_args': ['-r -v4 ', world], 'on_exit_shutdown': 'true'}.items()
             )

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_entity",
        arguments=[
            "-topic", "robot_description",  
            "-name", "quadrotor",
            '-timeout', '120.0',
            "-x", spawn_x_val,
            "-y", spawn_y_val,
            "-z", spawn_z_val,
        ],
        output="screen"
    )


    bridge_params = os.path.join(get_package_share_directory(package_name),'config','gz_bridge.yaml')
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ]
    )

    # Start RViz
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_file_path],
        output="screen"
    )

    # Create LaunchDescription
    launch_description = LaunchDescription()

    # Add launch actions
    # launch_description.add_action(rviz)
    launch_description.add_action(world_arg)
    launch_description.add_action(gz_sim)
    launch_description.add_action(rsp) 
    launch_description.add_action(spawn_entity)
    launch_description.add_action(bridge)


    return launch_description