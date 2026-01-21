from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.substitutions import Command

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    servobot_description_path = get_package_share_path("servobot_description")

    robot_description_content = Command(["cat ", str(servobot_description_path / "urdf" / "servobot.urdf")])

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            {
                "robot_description": ParameterValue(robot_description_content, value_type=str)
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", str(servobot_description_path / "rviz" / "servobot.rviz")],
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        output="screen",
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            rviz_node,
            joint_state_publisher_gui_node,
        ]
    )
