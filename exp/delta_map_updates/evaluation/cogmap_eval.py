# Cognitive map correctness evaluation for sequential conditions (B, C).
# Evaluates final map correctness and per-turn perception/stability metrics
# using the ToS CognitiveMapManager evaluation pipeline.

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

_EXP_ROOT = Path(__file__).resolve().parents[2]
_TOS_ROOT = _EXP_ROOT / "Theory-of-Space"
if str(_TOS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOS_ROOT))

from vagen.env.spatial.Base.tos_base.managers.cognitive_map_manager import CognitiveMapManager
from vagen.env.spatial.Base.tos_base.core.room import Room
from vagen.env.spatial.Base.tos_base.core.object import Agent


def load_ground_truth(combo_dir: str) -> Tuple[Room, Agent, List[str]]:
    """Load ground truth room, agent and observed items from history_state.json."""
    state_path = os.path.join(combo_dir, "history_state.json")
    with open(state_path) as f:
        state = json.load(f)
    room = Room.from_dict(state["room_dict"])
    agent = Agent.from_dict(state["agent_dict"])
    all_objects = state["room_dict"].get("all_objects", state["room_dict"].get("objects", []))
    observed_items = [obj["name"] for obj in all_objects]
    return room, agent, observed_items


def evaluate_final_map(
    response_text: str,
    combo_dir: str,
) -> Optional[Dict[str, float]]:
    """Evaluate a final map response against ground truth, returning pos/dir/facing/overall."""
    room, agent, observed_items = load_ground_truth(combo_dir)
    cm = CognitiveMapManager(cogmap_type="standard", pos_allow_scale=False, scope="all")
    result = cm.evaluate_cogmap_type(response_text, room, agent, observed_items, "global")
    if result is None or not result.extraction_success:
        return None
    m = result.metrics
    return {
        "extraction_success": True,
        "pos": float(getattr(m, "pos", 0.0)),
        "dir": float(getattr(m, "dir", 0.0)),
        "facing": float(getattr(m, "facing", 0.0)),
        "overall": float(getattr(m, "overall", 0.0)),
    }


def evaluate_condition_b(
    output_base: str,
    results_dir: str = None,
    model_name: str = "gemini-3-pro",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate all Condition B scenes. Returns (aggregate, per_sample)."""
    if results_dir is None:
        results_dir = str(_TOS_ROOT / "results")

    from delta_map_updates.runners.cogmap_runner import iter_combo_dirs
    combo_dirs = iter_combo_dirs(results_dir, model_name)

    scene_dirs = sorted([
        d for d in os.listdir(output_base)
        if os.path.isdir(os.path.join(output_base, d)) and d.startswith("run")
    ])

    per_sample = []
    total_api_calls = 0
    total_retries = 0

    for scene_dir_name in scene_dirs:
        scene_path = os.path.join(output_base, scene_dir_name)
        meta_path = os.path.join(scene_path, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            metadata = json.load(f)

        combo_dir = metadata["combo_dir"]
        room_hash = combo_dir.split(os.sep)[-5]
        n_steps = metadata["n_steps"]
        total_api_calls += metadata.get("total_api_calls", 0)
        total_retries += metadata.get("total_retries", 0)

        final_map_path = os.path.join(scene_path, f"step_{n_steps:02d}_map.json")
        final_response_path = os.path.join(scene_path, f"step_{n_steps:02d}_response.txt")

        if not os.path.exists(final_response_path):
            per_sample.append({
                "room_hash": room_hash,
                "scene_dir": scene_dir_name,
                "extraction_success": False,
                "error": "no_final_response",
            })
            continue

        with open(final_response_path) as f:
            final_response = f.read()

        metrics = evaluate_final_map(final_response, combo_dir)
        if metrics is None:
            per_sample.append({
                "room_hash": room_hash,
                "scene_dir": scene_dir_name,
                "extraction_success": False,
            })
        else:
            metrics["room_hash"] = room_hash
            metrics["scene_dir"] = scene_dir_name
            metrics["n_steps"] = n_steps
            per_sample.append(metrics)

    successful = [s for s in per_sample if s.get("extraction_success")]
    n = len(successful)
    n_total = len(per_sample)

    agg: Dict[str, Any] = {
        "n_total": n_total,
        "n_successful": n,
        "extraction_failure_rate": 1 - n / n_total if n_total > 0 else 0,
        "total_api_calls": total_api_calls,
        "total_retries": total_retries,
        "json_retry_rate": total_retries / total_api_calls if total_api_calls > 0 else 0,
    }
    for key in ["overall", "pos", "dir", "facing"]:
        vals = [s[key] for s in successful if key in s]
        mean = float(np.mean(vals)) if vals else 0.0
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        agg[f"{key}_mean"] = mean
        agg[f"{key}_se"] = se

    return agg, per_sample


def evaluate_per_turn_metrics(
    output_base: str,
    results_dir: str = None,
    model_name: str = "gemini-3-pro",
) -> Dict[str, Any]:
    """Compute per-turn perception and stability metrics across all scenes."""
    if results_dir is None:
        results_dir = str(_TOS_ROOT / "results")

    from delta_map_updates.runners.cogmap_runner import iter_combo_dirs
    combo_dirs = iter_combo_dirs(results_dir, model_name)

    scene_dirs = sorted([
        d for d in os.listdir(output_base)
        if os.path.isdir(os.path.join(output_base, d)) and d.startswith("run")
    ])

    all_perception = []
    all_stability = []

    for scene_dir_name in scene_dirs:
        scene_path = os.path.join(output_base, scene_dir_name)
        meta_path = os.path.join(scene_path, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            metadata = json.load(f)

        combo_dir = metadata["combo_dir"]
        n_steps = metadata["n_steps"]
        room, agent, observed_items = load_ground_truth(combo_dir)

        cm = CognitiveMapManager(cogmap_type="standard", pos_allow_scale=False, scope="all")

        prev_map_keys = set()
        for t in range(1, n_steps + 1):
            response_path = os.path.join(scene_path, f"step_{t:02d}_response.txt")
            if not os.path.exists(response_path):
                continue

            with open(response_path) as f:
                response_text = f.read()

            result = cm.evaluate_cogmap_type(response_text, room, agent, observed_items, "global")
            if result is None or not result.extraction_success:
                continue

            current_map_keys = set()
            if result.pred_json:
                current_map_keys = set(result.pred_json.keys())

            new_keys = current_map_keys - prev_map_keys
            if new_keys:
                all_perception.append(float(getattr(result.metrics, "overall", 0.0)))

            if prev_map_keys:
                all_stability.append(float(getattr(result.metrics, "overall", 0.0)))

            prev_map_keys = current_map_keys

    return {
        "perception_mean": float(np.mean(all_perception)) if all_perception else 0.0,
        "perception_se": float(np.std(all_perception, ddof=1) / np.sqrt(len(all_perception))) if len(all_perception) > 1 else 0.0,
        "perception_n": len(all_perception),
        "stability_mean": float(np.mean(all_stability)) if all_stability else 0.0,
        "stability_se": float(np.std(all_stability, ddof=1) / np.sqrt(len(all_stability))) if len(all_stability) > 1 else 0.0,
        "stability_n": len(all_stability),
    }


def evaluate_condition_c(
    output_base: str,
    results_dir: str = None,
    model_name: str = "gemini-3-pro",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate Condition C scenes using applied_response files (full map after delta apply)."""
    if results_dir is None:
        results_dir = str(_TOS_ROOT / "results")

    scene_dirs = sorted([
        d for d in os.listdir(output_base)
        if os.path.isdir(os.path.join(output_base, d)) and d.startswith("run")
    ])

    per_sample = []
    total_api_calls = 0
    total_retries = 0
    total_response_chars = 0

    for scene_dir_name in scene_dirs:
        scene_path = os.path.join(output_base, scene_dir_name)
        meta_path = os.path.join(scene_path, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            metadata = json.load(f)

        combo_dir = metadata["combo_dir"]
        room_hash = combo_dir.split(os.sep)[-5]
        n_steps = metadata["n_steps"]
        total_api_calls += metadata.get("total_api_calls", 0)
        total_retries += metadata.get("total_retries", 0)
        total_response_chars += metadata.get("total_response_chars", 0)

        final_applied_path = os.path.join(scene_path, f"step_{n_steps:02d}_applied_response.txt")
        if not os.path.exists(final_applied_path):
            per_sample.append({
                "room_hash": room_hash,
                "scene_dir": scene_dir_name,
                "extraction_success": False,
                "error": "no_final_applied_response",
            })
            continue

        with open(final_applied_path) as f:
            final_response = f.read()

        metrics = evaluate_final_map(final_response, combo_dir)
        if metrics is None:
            per_sample.append({
                "room_hash": room_hash,
                "scene_dir": scene_dir_name,
                "extraction_success": False,
            })
        else:
            metrics["room_hash"] = room_hash
            metrics["scene_dir"] = scene_dir_name
            metrics["n_steps"] = n_steps
            per_sample.append(metrics)

    successful = [s for s in per_sample if s.get("extraction_success")]
    n = len(successful)
    n_total = len(per_sample)

    agg: Dict[str, Any] = {
        "n_total": n_total,
        "n_successful": n,
        "extraction_failure_rate": 1 - n / n_total if n_total > 0 else 0,
        "total_api_calls": total_api_calls,
        "total_retries": total_retries,
        "json_retry_rate": total_retries / total_api_calls if total_api_calls > 0 else 0,
        "total_response_chars": total_response_chars,
    }
    for key in ["overall", "pos", "dir", "facing"]:
        vals = [s[key] for s in successful if key in s]
        mean = float(np.mean(vals)) if vals else 0.0
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        agg[f"{key}_mean"] = mean
        agg[f"{key}_se"] = se

    return agg, per_sample


def evaluate_condition_c_per_turn(
    output_base: str,
    results_dir: str = None,
    model_name: str = "gemini-3-pro",
) -> Dict[str, Any]:
    """Per-turn perception/stability for Condition C using applied_response files."""
    if results_dir is None:
        results_dir = str(_TOS_ROOT / "results")

    scene_dirs = sorted([
        d for d in os.listdir(output_base)
        if os.path.isdir(os.path.join(output_base, d)) and d.startswith("run")
    ])

    all_perception = []
    all_stability = []

    for scene_dir_name in scene_dirs:
        scene_path = os.path.join(output_base, scene_dir_name)
        meta_path = os.path.join(scene_path, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            metadata = json.load(f)

        combo_dir = metadata["combo_dir"]
        n_steps = metadata["n_steps"]
        room, agent, observed_items = load_ground_truth(combo_dir)

        cm = CognitiveMapManager(cogmap_type="standard", pos_allow_scale=False, scope="all")

        prev_map_keys = set()
        for t in range(1, n_steps + 1):
            response_path = os.path.join(scene_path, f"step_{t:02d}_applied_response.txt")
            if not os.path.exists(response_path):
                continue

            with open(response_path) as f:
                response_text = f.read()

            result = cm.evaluate_cogmap_type(response_text, room, agent, observed_items, "global")
            if result is None or not result.extraction_success:
                continue

            current_map_keys = set()
            if result.pred_json:
                current_map_keys = set(result.pred_json.keys())

            new_keys = current_map_keys - prev_map_keys
            if new_keys:
                all_perception.append(float(getattr(result.metrics, "overall", 0.0)))

            if prev_map_keys:
                all_stability.append(float(getattr(result.metrics, "overall", 0.0)))

            prev_map_keys = current_map_keys

    return {
        "perception_mean": float(np.mean(all_perception)) if all_perception else 0.0,
        "perception_se": float(np.std(all_perception, ddof=1) / np.sqrt(len(all_perception))) if len(all_perception) > 1 else 0.0,
        "perception_n": len(all_perception),
        "stability_mean": float(np.mean(all_stability)) if all_stability else 0.0,
        "stability_se": float(np.std(all_stability, ddof=1) / np.sqrt(len(all_stability))) if len(all_stability) > 1 else 0.0,
        "stability_n": len(all_stability),
    }
