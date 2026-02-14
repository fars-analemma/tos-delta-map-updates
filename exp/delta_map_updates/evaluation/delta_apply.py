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


def apply_delta(M_prev: dict, delta: dict) -> dict:
    M_new = copy.deepcopy(M_prev)
    for key, value in delta.get("updates", {}).items():
        if key in M_new:
            M_new[key].update(value)
        else:
            M_new[key] = value
    return M_new
