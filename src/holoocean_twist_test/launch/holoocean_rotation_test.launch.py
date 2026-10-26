from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

MAX_TIME = 20.0

def generate_launch_description():
    ns = "holoocean"

    holoocean_node = Node(
        package="holoocean_main",
        executable="holoocean_node",
        name="holoocean_node",
        namespace=ns,
        output="screen",
        emulate_tty=True,
        parameters=[{
            "use_sim_time": True,
            "show_viewport": True,
            "relative_path": True,
            "scenario_path": "config/blueROV.json",
            "publish_commands": True,
        }],
    )

    adapter_node = Node(
        package="holoocean_twist_test",
        # executable="twist_to_agent",
        # name="twist_to_agent",
        executable="pose_bypass_adapter",
        name="pose_bypass_adapter",
        namespace=ns,
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "max_sim_time": MAX_TIME,
            "agent_name": "auv0",
            "control_rate_hz": 50.0,
            "debug_log": True,
        }],
    )

    rotation_node = Node(
        package="holoocean_twist_test",
        executable="rotation_test",
        name="rotation_test",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "agent_name": "auv0",
            "axis": "pitch",           # change per run
            "mode": "regulate",      # 'regulate' or 'velocity'
            "target_omega": -0.5,
            # "target_angle": 0.5,
            "target_angle": 0.3,
            "kp": 1.0,
            "max_sim_time": MAX_TIME,
        }],
    )

    return LaunchDescription([
        holoocean_node,
        adapter_node,
        rotation_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=rotation_node,
                on_exit=[EmitEvent(event=Shutdown(reason="test done"))],
            )
        ),
    ])
