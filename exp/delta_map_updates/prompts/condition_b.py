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

You are given your **previous cognitive map** (from the last step) below.
Your task is to produce an **updated cognitive map** that reflects ALL of your knowledge so far.

### Previous Cognitive Map (M_t-1)
```json
{prev_map_json}
```

### Update Rules (MUST follow strictly)
1. **Preserve rule**: Copy ALL entries from the previous map unchanged, UNLESS the current observation provides direct evidence that they should change. Do NOT modify entries for objects not visible in the current observation.
2. **Evidence restriction**: Only add or update entries for objects that are visible in the current observation or are newly observed. If an object is not visible right now, keep its previous entry exactly as-is.
3. **Conflict resolution**: If the current observation contradicts a previous entry for a currently visible object (e.g., different position or facing direction), overwrite that object's entry to match the current observation.

**IMPORTANT**: Do NOT remove or modify entries for objects that are NOT visible in the current observation. Only update what you can currently see.

Now output the full updated cognitive map incorporating both the preserved entries and any updates from the current observation.
"""


def get_condition_b_prompt(prev_map: dict, enable_think: bool = True) -> str:
    prev_map_str = json.dumps(prev_map, indent=2) if prev_map else "{}"
    update_section = CONDITION_B_UPDATE_RULES.format(prev_map_json=prev_map_str)
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
