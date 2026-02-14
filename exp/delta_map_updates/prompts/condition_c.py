# Condition C: Delta-map update prompt.
# The model is given M_{t-1} and O_t, and must output a compact JSON delta
# describing only the objects whose states should change. The delta is applied
# programmatically to form M_t = Apply(M_{t-1}, delta_t).

import json
import sys
from pathlib import Path

_TOS_ROOT = str(Path(__file__).resolve().parents[2] / "Theory-of-Space")
if _TOS_ROOT not in sys.path:
    sys.path.insert(0, _TOS_ROOT)

from vagen.env.spatial.Base.tos_base.prompts.cogmap_prompts import (
    BASE_COGMAP_PROMPT,
    COGMAP_INSTRUCTION_GLOBAL_ONLY,
)

CONDITION_C_UPDATE_RULES = """\
## Cognitive Map Update Rules (Delta Mode)

You are given your **previous cognitive map** (from the last step) below.
Your task is to produce a **delta update** — a compact JSON describing ONLY the objects whose entries should change. Unchanged entries will be automatically preserved.

{agent_state_section}

### Previous Cognitive Map (M_t-1)
```json
{prev_map_json}
```

### Update Rules (MUST follow strictly)
1. **Preserve rule**: Any object NOT included in your delta output is automatically kept unchanged from the previous map. Do NOT include objects that have not changed.
2. **Evidence restriction**: Only include objects in your delta that are visible in the current observation and whose state has changed, or that are newly observed.
3. **Conflict resolution**: If the current observation contradicts a previous entry for a currently visible object (e.g., different position or facing direction), include that object in the delta with the corrected values.
4. **Agent update**: Always include the "agent" key in your delta with the updated position and facing direction based on the actions taken.

### Facing Direction Rules (CRITICAL)
**EVERY object in your delta MUST have a "facing" key** with a cardinal direction ("north", "south", "east", "west"). This applies to ALL objects without exception — furniture, small items (caps, pans, cups, lamps, shoes, baskets, candles, vases, plants, etc.), and the agent. Objects that appear to lack an obvious front still have a facing direction in this environment.

To determine an object's facing in **global cardinal directions**:
- First determine YOUR current facing (the agent's cardinal direction after all rotations).
- Objects whose front faces toward you: their facing = opposite of your facing direction.
- Objects whose front faces to your left: rotate your facing 90 degrees counterclockwise.
- Objects whose front faces to your right: rotate your facing 90 degrees clockwise.
- Objects whose front faces away from you: their facing = same as your facing direction.
- Remember: when you face north, your right is east. When you face east, your right is south. And so on.

If you are unsure of an object's facing, default to the opposite of your current facing direction (i.e., assume it faces toward you).

### Output Format (CRITICAL)
Output ONLY a delta JSON with the following structure. Do NOT output the full map.
Every entry MUST include both "position" and "facing".

```json
{{"updates": {{"obj_name": {{"position": [x, y], "facing": "east"}}, "obj2": {{"position": [x, y], "facing": "north"}}}}}}
```

- The "updates" object contains ONLY entries that should change or be added.
- Each entry MUST include "position" and "facing" fields.
- Omitted entries are preserved from M_t-1.
- If nothing needs to change, output: {{"updates": {{}}}}
- Do NOT output the full updated map. Output ONLY the delta.
"""

CONDITION_C_FORMAT_RULES = """\
## Response Format

Think step-by-step inside <think>...</think> tags. Then output your delta JSON inside a ```json``` code block.

Example response:
<think>
I moved forward, so the agent position changed. I can see a new chair to my right.
The desk from before is not visible, so I won't include it (it stays unchanged).
</think>

```json
{"updates": {"agent": {"position": [1, 0], "facing": "north"}, "chair_1": {"position": [2, 1], "facing": "west"}}}
```
"""


def _get_agent_state_section(prev_map: dict) -> str:
    agent = prev_map.get("agent", {})
    pos = agent.get("position")
    facing = agent.get("facing")
    if pos and facing:
        return f"**Your current state (from previous map)**: position {pos}, facing **{facing}**."
    elif not prev_map:
        return "**Your current state**: This is your first observation. You start at position [0, 0], facing **north**."
    return ""


def get_condition_c_prompt(prev_map: dict, enable_think: bool = True) -> str:
    prev_map_str = json.dumps(prev_map, indent=2) if prev_map else "{}"
    agent_section = _get_agent_state_section(prev_map)
    update_section = CONDITION_C_UPDATE_RULES.format(
        prev_map_json=prev_map_str,
        agent_state_section=agent_section,
    )
    fmt = CONDITION_C_FORMAT_RULES if enable_think else ""
    return (
        f"{BASE_COGMAP_PROMPT}\n\n"
        f"{COGMAP_INSTRUCTION_GLOBAL_ONLY}\n\n"
        f"{update_section}\n\n"
        f"{fmt}"
    )


CONDITION_C_DESCRIPTION = (
    "Condition C (delta-map updates): the model receives M_{t-1} "
    "as explicit context alongside O_t and outputs only a compact delta JSON "
    "describing entries that should change. The delta is applied programmatically "
    "to form M_t = Apply(M_{t-1}, delta_t). This tests whether reducing output "
    "to only changed entries further reduces transcription errors beyond Condition B."
)
