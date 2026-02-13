# Cognitive map probing runner for initial exploration (conditions A, B, C).
# Wraps the ToS spatial_run.py pipeline with helpers for running each condition
# and extracting per-sample results from the output directories.

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_TOS_ROOT = Path(__file__).resolve().parents[2] / "Theory-of-Space"
_SPATIAL_RUN = _TOS_ROOT / "scripts" / "SpatialGym" / "spatial_run.py"


def run_exploration(
    num: int = 25,
    model_name: str = "gemini-3-pro",
    data_dir: str = "room_data/3-room/",
    output_root: str = "results/",
    render_mode: str = "vision",
    exp_type: str = "passive",
    inference_mode: str = "direct",
    extra_args: Optional[List[str]] = None,
) -> int:
    cmd = [
        sys.executable, str(_SPATIAL_RUN),
        "--phase", "explore",
        "--exp-type", exp_type,
        "--render-mode", render_mode,
        "--num", str(num),
        "--model-name", model_name,
        "--data-dir", data_dir,
        "--output-root", output_root,
        "--inference-mode", inference_mode,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd, cwd=str(_TOS_ROOT))


def run_cogmap(
    num: int = 25,
    model_name: str = "gemini-3-pro",
    data_dir: str = "room_data/3-room/",
    output_root: str = "results/",
    render_mode: str = "vision",
    exp_type: str = "passive",
    inference_mode: str = "direct",
    override: bool = False,
    extra_args: Optional[List[str]] = None,
) -> int:
    cmd = [
        sys.executable, str(_SPATIAL_RUN),
        "--phase", "cogmap",
        "--exp-type", exp_type,
        "--render-mode", render_mode,
        "--num", str(num),
        "--model-name", model_name,
        "--data-dir", data_dir,
        "--output-root", output_root,
        "--inference-mode", inference_mode,
    ]
    if override:
        cmd.append("--cogmap-override")
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd, cwd=str(_TOS_ROOT))


def ensure_passive_turn_logs(results_dir: str, model_name: str = "gemini-3-pro") -> int:
    """Create synthetic exploration_turn_logs for passive-mode combos.

    The ToS passive pipeline stores observations in messages.json but does not
    create exploration_turn_logs.json (needed for cogmap evaluation). This
    function creates a single-entry turn log from history_state.json.

    Returns the number of combo dirs fixed.
    """
    model_dir = os.path.join(results_dir, model_name)
    count = 0
    for room_hash in os.listdir(model_dir):
        combo = os.path.join(model_dir, room_hash, "vision", "passive", "think", "scout")
        if not os.path.isdir(combo):
            continue
        state_file = os.path.join(combo, "history_state.json")
        exp_log_file = os.path.join(combo, "exploration_turn_logs.json")
        if not os.path.exists(state_file):
            continue
        with open(state_file) as f:
            state = json.load(f)
        room_dict = state["room_dict"]
        all_objects = room_dict.get("all_objects", room_dict.get("objects", []))
        turn_log = {
            "turn_number": 1,
            "is_exploration_phase": True,
            "room_state": room_dict,
            "agent_state": state["agent_dict"],
            "exploration_log": {
                "observed_items": [obj["name"] for obj in all_objects],
            },
            "user_message": "",
            "assistant_raw_message": "",
            "assistant_think_message": "",
            "assistant_parsed_message": "",
            "message_images": [],
            "info": {},
        }
        with open(exp_log_file, "w") as f:
            json.dump([turn_log], f, ensure_ascii=False, indent=2)
        count += 1
    return count


def iter_combo_dirs(results_dir: str, model_name: str = "gemini-3-pro") -> List[str]:
    """List all passive/scout combo directories under results_dir/model_name."""
    model_dir = os.path.join(results_dir, model_name)
    dirs = []
    for room_hash in sorted(os.listdir(model_dir)):
        combo = os.path.join(model_dir, room_hash, "vision", "passive", "think", "scout")
        if os.path.isdir(combo):
            dirs.append(combo)
    return dirs


def extract_cogmap_metrics(combo_dir: str) -> Optional[Dict]:
    """Extract cogmap metrics from a single combo directory."""
    exp_log_file = os.path.join(combo_dir, "exploration_turn_logs.json")
    if not os.path.exists(exp_log_file):
        return None
    with open(exp_log_file) as f:
        logs = json.load(f)
    if not logs:
        return None
    last_log = logs[-1]
    cm = last_log.get("cogmap_log", {})
    global_data = cm.get("global", {})
    if not global_data.get("extraction_success", False):
        return {"extraction_success": False}
    metrics = global_data.get("metrics", {})
    return {
        "extraction_success": True,
        "overall": metrics.get("overall", 0.0),
        "pos": metrics.get("pos", 0.0),
        "dir": metrics.get("dir", 0.0),
        "facing": metrics.get("facing", 0.0),
    }


def aggregate_metrics(
    results_dir: str, model_name: str = "gemini-3-pro"
) -> Tuple[Dict, List[Dict]]:
    """Aggregate cogmap metrics across all scenes.

    Returns (aggregate_dict, per_sample_list).
    """
    combo_dirs = iter_combo_dirs(results_dir, model_name)
    per_sample = []
    for combo_dir in combo_dirs:
        room_hash = combo_dir.split(os.sep)[-5]
        m = extract_cogmap_metrics(combo_dir)
        if m is not None:
            m["room_hash"] = room_hash
            per_sample.append(m)

    successful = [s for s in per_sample if s.get("extraction_success")]
    n = len(successful)
    agg = {
        "n_total": len(per_sample),
        "n_successful": n,
        "extraction_failure_rate": 1 - n / len(per_sample) if per_sample else 0,
    }
    for key in ["overall", "pos", "dir", "facing"]:
        vals = [s[key] for s in successful]
        mean = float(np.mean(vals)) if vals else 0.0
        se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        agg[f"{key}_mean"] = mean
        agg[f"{key}_se"] = se

    return agg, per_sample
