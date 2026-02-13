# Condition B: Rule-based full regeneration with preserve/overwrite rules.
# The model receives M_{t-1} as explicit context alongside O_t and must output
# a full updated map M_t following evidence-based update rules:
#   1. Preserve all entries from M_{t-1} unchanged unless current observation contradicts them
#   2. Only update objects visible in O_t or newly observed
#   3. Overwrite entries that contradict current observation

import json
import sys
from pathlib import Path

_TOS_ROOT = str(Path(__file__).resolve().parents[2] / "Theory-of-Space")
if _TOS_ROOT not in sys.path:
    sys.path.insert(0, _TOS_ROOT)

from vagen.env.spatial.Base.tos_base.prompts.cogmap_prompts import (
    BASE_COGMAP_PROMPT,
    COGMAP_INSTRUCTION_GLOBAL_ONLY,
    _cogmap_format_rules,
)

CONDITION_B_UPDATE_RULES = """\
## Cognitive Map Update Rules

You are seeing **only the current step's observation**. Your previous cognitive map below contains all your knowledge from prior observations. Use it as your primary memory.

{agent_state_section}

### Previous Cognitive Map (M_t-1)
```json
{prev_map_json}
```

### Update Rules (MUST follow strictly)
1. **Preserve rule**: Copy ALL entries from the previous map unchanged, UNLESS the current observation provides direct evidence that they should change. Do NOT modify entries for objects not visible in the current observation.
2. **Evidence restriction**: Only add or update entries for objects that are visible in the current observation or are newly observed. If an object is not visible right now, keep its previous entry exactly as-is.
3. **Conflict resolution**: If the current observation contradicts a previous entry for a currently visible object (e.g., different position or facing direction), overwrite that object's entry to match the current observation.
4. **Agent update**: Update the agent's position and facing to reflect the actions taken in this step. Use the actions described in the current observation to compute the new agent state from the previous map's agent entry.

**IMPORTANT**: Do NOT remove or modify entries for objects that are NOT visible in the current observation. Only update what you can currently see. The previous map is your ONLY source of knowledge about objects not in your current view.

Now output the full updated cognitive map incorporating both the preserved entries and any updates from the current observation.
"""


def _get_agent_state_section(prev_map: dict) -> str:
    agent = prev_map.get("agent", {})
    pos = agent.get("position")
    facing = agent.get("facing")
    if pos and facing:
        return f"**Your current state**: You are at position {pos}, facing **{facing}**. Interpret the current observation relative to this position and orientation."
    elif not prev_map:
        return "**Your current state**: This is your first observation. You start at position [0, 0], facing **north**."
    return ""


def get_condition_b_prompt(prev_map: dict, enable_think: bool = True) -> str:
    prev_map_str = json.dumps(prev_map, indent=2) if prev_map else "{}"
    agent_section = _get_agent_state_section(prev_map)
    update_section = CONDITION_B_UPDATE_RULES.format(
        prev_map_json=prev_map_str,
        agent_state_section=agent_section,
    )
    fmt = _cogmap_format_rules(enable_think)
    return (
        f"{BASE_COGMAP_PROMPT}\n\n"
        f"{COGMAP_INSTRUCTION_GLOBAL_ONLY}\n\n"
        f"{update_section}\n\n"
        f"{fmt}"
    )


CONDITION_B_DESCRIPTION = (
    "Condition B (rule-based full regeneration): the model receives M_{t-1} "
    "as explicit context alongside O_t and outputs a full updated map M_t "
    "following preserve/overwrite rules. This tests whether providing the "
    "prior map and enforcing evidence-based update rules reduces belief drift."
)
