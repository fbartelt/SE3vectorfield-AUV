# ~/holoocean_ws/src/holoocean_twist_test/launch/holoocean_twist_test.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

MAX_TIME = 60.0
CURVE_X0, CURVE_Y0, CURVE_Z0 = -2.0, -2.0, -2.0
CURVE_POINTS = 5000
CURVE_RADIUS = 1.5 * 2
CURVE_N_ROLLS = 0.0

kn1 = 0.6
kn2 = 1.0
kt1 = 0.05
kt2 = 1.0
kt3 = kn2

def generate_launch_description():
    # Reuse the example's config YAML - it already sets 'scenario' and agent names
    params_file = str(
        Path(get_package_share_directory("holoocean_examples"))
        / "config"
        / "command_config.yaml"
    )

    holoocean_ns = "holoocean"

    holoocean_node = Node(
        package="holoocean_main",
        executable="holoocean_node",
        name="holoocean_node",
        namespace=holoocean_ns,
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "use_sim_time": True,
                "show_viewport": True,
                "relative_path": True,
                "scenario_path": "config/blueROV.json",
                "publish_commands": True,
            }
        ],
    )

    # Node(
    #     name='holoocean_node',
    #     package='holoocean_main',
    #     executable='holoocean_node',
    #     namespace=holoocean_ns,
    #     output='screen',
    #     emulate_tty=True,
    #     parameters=[params_file],
    # )

    adapter_node = Node(
        package="holoocean_twist_test",
        executable="twist_to_agent",
        name="twist_to_agent",
        namespace=holoocean_ns,
        output="screen",
        parameters=[
            {
                "use_sim_time": True,  # <-- add this
                "max_sim_time": MAX_TIME,  # match the controller
                "agent_name": "auv0",
                "control_rate_hz": 50.0,
                "debug_log": True,
            }
        ],
    )

    # Node(
    #     name="twist_to_agent_adapter",
    #     package="holoocean_twist_test",
    #     executable="twist_to_agent",
    #     namespace=holoocean_ns,  # so it lands on /holoocean/command/agent
    #     output="screen",
    # )

    twist_node = Node(
        package="holoocean_twist_test",
        executable="path_controller",
        name="path_controller",
        # namespace=holoocean_ns,
        output="screen",
        parameters=[
            {
                "use_sim_time": True,  # <-- add this
                "max_sim_time": MAX_TIME,  # match the controller
                "agent_name": "auv0",
                "rate_hz": 1000.0,
                "radius": CURVE_RADIUS,
                "z0": CURVE_Z0,
                "center_x": CURVE_X0,
                "center_y": CURVE_Y0,
                "n_points": CURVE_POINTS,
                "n_rolls": CURVE_N_ROLLS,
                "kt1": kt1,
                "kt2": kt2,
                "kt3": kt3,
                "kn1": kn1,
                "kn2": kn2,
                "twist_is_body_frame": False,
                "angular_first": False,
            }
        ],
    )

    # --- Path visualizer: draws the reference circle + vehicle trail ---
    path_visualizer_node = Node(
        package="holoocean_twist_test",
        executable="path_visualizer",
        name="path_visualizer",
        namespace=holoocean_ns,  # matches the bridge's debug/points topic
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                # These MUST match the twist_node parameters exactly:
                "agent_name": "auv0",
                "radius": CURVE_RADIUS,
                "z0": CURVE_Z0,
                "center_x": CURVE_X0,
                "center_y": CURVE_Y0,
                "n_points": 200,  # visual only; 300 is plenty for drawing
                "n_rolls": CURVE_N_ROLLS,
                "show_trail": True,
                "trail_max_len": 500,
                "republish_hz": 2.0,
            }
        ],
    )
    return LaunchDescription(
        [
            holoocean_node,
            adapter_node,
            twist_node,
            path_visualizer_node,  # <-- add this line
            # When the controller exits, shut down the whole launch.
            RegisterEventHandler(
                OnProcessExit(
                    target_action=twist_node,
                    on_exit=[EmitEvent(event=Shutdown(reason="controller finished"))],
                )
            ),
        ]
    )
