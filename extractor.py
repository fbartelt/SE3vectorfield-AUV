import holoocean
import numpy as np

scenario = {
    "name": "thruster_probe",
    "package_name": "Ocean",
    "world": "SimpleUnderwater",
    "main_agent": "auv0",
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "control_scheme": 0,          # AUV_THRUSTERS
            "location": [0, 0, -5],
            "rotation": [0, 0, 0],
            "sensors": []                  # <-- required key
        }
    ]
}

with holoocean.make(scenario_cfg=scenario) as env:
    agent = env.agents["auv0"]

    print("=== thruster_p (positions, 8x3) ===")
    print(agent.thruster_p)

    print("\n=== thruster_d (directions, 8x3) ===")
    print(agent.thruster_d)

    print("\n=== Physical properties ===")
    for attr in ("mass", "volume", "I"):
        print(f"{attr}: {getattr(agent, attr, 'N/A')}")
