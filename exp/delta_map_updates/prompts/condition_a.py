# Condition A: Scratch regeneration prompt (original ToS cognitive-map probing).
# At each step the model receives full observation history and outputs a complete
# global cognitive map from scratch. This re-exports the standard ToS prompts
# from cogmap_prompts.py so conditions B/C can diverge from this baseline.

import sys
from pathlib import Path

_TOS_ROOT = str(Path(__file__).resolve().parents[2] / "Theory-of-Space")
if _TOS_ROOT not in sys.path:
    sys.path.insert(0, _TOS_ROOT)

from vagen.env.spatial.Base.tos_base.prompts.cogmap_prompts import (
    BASE_COGMAP_PROMPT,
    COGMAP_INSTRUCTION_GLOBAL_ONLY,
    get_cogmap_prompt,
    _cogmap_format_rules,
)


def get_condition_a_prompt(enable_think: bool = True) -> str:
    """Return the Condition A (scratch regeneration) cogmap prompt.

    Identical to the standard ToS global cogmap prompt: schema rules,
    global-map instructions, and output format rules.
    """
    return get_cogmap_prompt("global", enable_think=enable_think)


CONDITION_A_DESCRIPTION = (
    "Condition A (scratch regeneration): the model receives the full observation "
    "history up to step t and generates a complete global cognitive map M_t from "
    "scratch. No prior map is provided. This is the original ToS protocol."
)
