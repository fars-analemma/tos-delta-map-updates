# Sequential runner for Condition C (delta-map updates).
# Same pipeline as Condition B but the model outputs only a delta JSON.
# The delta is applied programmatically: M_t = Apply(M_{t-1}, delta_t).

import json
import os
import re
import sys
import time
import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from openai import OpenAI

_EXP_ROOT = Path(__file__).resolve().parents[2]
_TOS_ROOT = _EXP_ROOT / "Theory-of-Space"
if str(_TOS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOS_ROOT))

from delta_map_updates.prompts.condition_c import get_condition_c_prompt
from delta_map_updates.evaluation.delta_apply import apply_delta, validate_delta
from delta_map_updates.runners.cogmap_runner import iter_combo_dirs
from delta_map_updates.runners.condition_b_runner import (
    _parse_exploration_steps,
    _split_preamble_and_history,
    _assign_images_to_steps,
    _build_step_message,
    _extract_json_from_text,
    _call_api,
)

logger = logging.getLogger(__name__)


def _extract_delta_from_text(text: str) -> Optional[Dict[str, Any]]:
    parsed = _extract_json_from_text(text)
    if parsed is None:
        return None
    if validate_delta(parsed):
        return parsed
    if validate_delta({"updates": parsed}):
        return {"updates": parsed}
    return None


def run_condition_c_scene(
    combo_dir: str,
    output_dir: str,
    client: OpenAI,
    model_name: str = "gemini-3-pro",
    temperature: float = 1.0,
    max_completion_tokens: int = 32768,
    max_retries: int = 3,
) -> Dict[str, Any]:
    messages_path = os.path.join(combo_dir, "messages.json")
    with open(messages_path) as f:
        raw_messages = json.load(f)

    system_prompt = raw_messages[0]["content"]
    user_msg = raw_messages[1]
    content = user_msg["content"]
    all_images = user_msg.get("images", [])

    image_base_dir = str(_TOS_ROOT)

    steps = _parse_exploration_steps(content)
    if not steps:
        logger.warning(f"No exploration steps found in {combo_dir}")
        return {"error": "no_steps"}

    preamble, _ = _split_preamble_and_history(content)
    preamble_images, step_images = _assign_images_to_steps(content, all_images, steps)

    os.makedirs(output_dir, exist_ok=True)

    prev_map: Dict = {}
    step_results = []
    total_retries = 0
    total_calls = 0
    total_response_chars = 0

    for t, step_text in enumerate(steps, start=1):
        steps_so_far = steps[:t]
        images_so_far = step_images[:t]

        cc_prompt = get_condition_c_prompt(prev_map, enable_think=True)

        api_messages = _build_step_message(
            system_prompt=system_prompt,
            preamble=preamble,
            steps_so_far=steps_so_far,
            preamble_images=preamble_images,
            step_images_list=images_so_far,
            condition_b_prompt=cc_prompt,
            image_base_dir=image_base_dir,
        )

        response_text = None
        parsed_delta = None
        retries_this_step = 0

        for attempt in range(1, max_retries + 1):
            total_calls += 1
            try:
                response_text = _call_api(
                    client, api_messages, model_name,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    image_base_dir=image_base_dir,
                )
                parsed_delta = _extract_delta_from_text(response_text)
                if parsed_delta is not None:
                    break
                else:
                    retries_this_step += 1
                    logger.warning(f"  Step {t} attempt {attempt}: delta parse failed, retrying...")
            except Exception as e:
                retries_this_step += 1
                logger.error(f"  Step {t} attempt {attempt}: API error: {e}")
                time.sleep(2 ** attempt)

        total_retries += retries_this_step

        if parsed_delta is None:
            logger.warning(f"  Step {t}: All retries exhausted, using empty delta (preserve M_{{t-1}})")
            parsed_delta = {"updates": {}}

        applied_map = apply_delta(prev_map, parsed_delta)

        if response_text is not None:
            total_response_chars += len(response_text)

        with open(os.path.join(output_dir, f"step_{t:02d}_delta.json"), "w") as f:
            json.dump(parsed_delta, f, indent=2, ensure_ascii=False)

        with open(os.path.join(output_dir, f"step_{t:02d}_map.json"), "w") as f:
            json.dump(applied_map, f, indent=2, ensure_ascii=False)

        if response_text is not None:
            with open(os.path.join(output_dir, f"step_{t:02d}_response.txt"), "w") as f:
                f.write(response_text)

        applied_map_json = json.dumps(applied_map, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, f"step_{t:02d}_applied_response.txt"), "w") as f:
            f.write(f"```json\n{applied_map_json}\n```")

        step_results.append({
            "step": t,
            "retries": retries_this_step,
            "fallback_used": len(parsed_delta.get("updates", {})) == 0 and retries_this_step > 0,
            "n_updates": len(parsed_delta.get("updates", {})),
            "n_objects_in_map": len(applied_map),
            "response_chars": len(response_text) if response_text else 0,
        })

        prev_map = applied_map
        logger.info(
            f"  Step {t}/{len(steps)}: delta has {len(parsed_delta.get('updates', {}))} updates, "
            f"map has {len(applied_map)} entries, retries={retries_this_step}"
        )

    metadata = {
        "combo_dir": combo_dir,
        "n_steps": len(steps),
        "total_api_calls": total_calls,
        "total_retries": total_retries,
        "total_response_chars": total_response_chars,
        "step_results": step_results,
        "final_map_n_objects": len(prev_map),
        "model_name": model_name,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def run_condition_c_all(
    results_dir: str = None,
    output_base: str = None,
    model_name: str = "gemini-3-pro",
    temperature: float = 1.0,
    max_completion_tokens: int = 32768,
    max_retries: int = 3,
    num_scenes: int = 25,
    start_scene: int = 0,
) -> List[Dict]:
    if results_dir is None:
        results_dir = str(_TOS_ROOT / "results")
    if output_base is None:
        output_base = str(_EXP_ROOT / "delta_map_updates" / "results" / "cogmap" / "condition_c")

    combo_dirs = iter_combo_dirs(results_dir, model_name)
    combo_dirs = combo_dirs[start_scene:start_scene + num_scenes]

    api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("LEMMA_MAAS_API_KEY", ""))
    base_url = os.environ.get("LEMMA_MAAS_BASE_URL", "139.224.231.89:8765")
    if not base_url.startswith("http"):
        base_url = f"http://{base_url}/v1"
    elif not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=500, max_retries=5)

    all_results = []
    for i, combo_dir in enumerate(combo_dirs):
        room_hash = combo_dir.split(os.sep)[-5]
        scene_output = os.path.join(output_base, f"run{i:02d}_{room_hash}")
        logger.info(f"Scene {i}/{len(combo_dirs)}: {room_hash}")

        if os.path.exists(os.path.join(scene_output, "metadata.json")):
            logger.info(f"  Skipping (already completed)")
            with open(os.path.join(scene_output, "metadata.json")) as f:
                all_results.append(json.load(f))
            continue

        result = run_condition_c_scene(
            combo_dir=combo_dir,
            output_dir=scene_output,
            client=client,
            model_name=model_name,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_retries=max_retries,
        )
        all_results.append(result)

    summary_path = os.path.join(output_base, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results
