# Delta JSON application logic for Condition C.
# Applies delta updates to a cognitive map: M_t = Apply(M_{t-1}, delta_t)
# Delta format: {"updates": {"obj_name": {"position": [x,y], "facing": "east"}, ...}}

import copy
from typing import Any, Dict


def validate_delta(delta: Any) -> bool:
    if not isinstance(delta, dict):
        return False
    if "updates" not in delta:
        return False
    updates = delta["updates"]
    if not isinstance(updates, dict):
        return False
    for key, value in updates.items():
        if not isinstance(key, str):
            return False
        if not isinstance(value, dict):
            return False
    return True


_OPPOSITE_FACING = {"north": "south", "south": "north", "east": "west", "west": "east"}


def apply_delta(M_prev: dict, delta: dict) -> dict:
    M_new = copy.deepcopy(M_prev)
    for key, value in delta.get("updates", {}).items():
        if key in M_new:
            M_new[key].update(value)
        else:
            M_new[key] = value

    agent_facing = None
    agent_data = M_new.get("agent", {})
    if isinstance(agent_data, dict) and "facing" in agent_data:
        agent_facing = agent_data["facing"]

    for key, value in M_new.items():
        if key == "agent":
            continue
        if isinstance(value, dict) and "facing" not in value:
            if agent_facing and agent_facing in _OPPOSITE_FACING:
                value["facing"] = _OPPOSITE_FACING[agent_facing]
            else:
                value["facing"] = "north"
    return M_new
