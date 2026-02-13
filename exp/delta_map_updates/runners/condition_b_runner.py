# Sequential runner for Condition B (rule-based full regeneration).
# Decomposes passive SCOUT trajectories into per-step observations, calls the
# API at each step with M_{t-1} as context, and chains M_t forward. Uses the
# OpenAI-compatible MaaS proxy directly via the openai SDK.

import json
import os
import re
import sys
import time
import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from openai import OpenAI

_EXP_ROOT = Path(__file__).resolve().parents[2]
_TOS_ROOT = _EXP_ROOT / "Theory-of-Space"
if str(_TOS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOS_ROOT))

from delta_map_updates.prompts.condition_b import get_condition_b_prompt
from delta_map_updates.runners.cogmap_runner import iter_combo_dirs

logger = logging.getLogger(__name__)


def _parse_exploration_steps(content: str) -> List[str]:
    """Split the Exploration History section into individual step strings."""
    match = re.search(r"## Exploration History\n", content)
    if not match:
        return []
    history_text = content[match.end():]
    steps = re.split(r"\n(?=\d+\.\s)", history_text.strip())
    steps = [s.strip() for s in steps if s.strip()]
    result = []
    for s in steps:
        if re.match(r"\d+\.\s+Actions:\s*\[Term\(\)\]", s):
            break
        result.append(s)
    return result


def _split_preamble_and_history(content: str) -> Tuple[str, str]:
    """Split message content into preamble (before Exploration History) and history section."""
    match = re.search(r"## Exploration History\n", content)
    if not match:
        return content, ""
    return content[:match.start()], content[match.start():]


def _count_image_placeholders(text: str) -> int:
    return text.count("<image>")


def _assign_images_to_steps(
    content: str, all_images: List[str], steps: List[str]
) -> Tuple[List[str], List[List[str]]]:
    """Assign images to preamble and each step based on <image> placeholder counts."""
    preamble, _ = _split_preamble_and_history(content)
    preamble_img_count = _count_image_placeholders(preamble)
    preamble_images = all_images[:preamble_img_count]
    remaining = all_images[preamble_img_count:]

    step_images = []
    idx = 0
    for step_text in steps:
        n = _count_image_placeholders(step_text)
        step_images.append(remaining[idx:idx + n])
        idx += n

    return preamble_images, step_images


def _build_step_message(
    system_prompt: str,
    preamble: str,
    steps_so_far: List[str],
    preamble_images: List[str],
    step_images_list: List[List[str]],
    condition_b_prompt: str,
    image_base_dir: str,
) -> List[Dict[str, Any]]:
    """Build the OpenAI-format messages for a single step."""
    history_section = "## Exploration History\n" + "\n".join(steps_so_far) + "\n"
    user_content_text = preamble + history_section + "\n" + condition_b_prompt

    all_step_images = list(preamble_images)
    for si in step_images_list:
        all_step_images.extend(si)

    resolved_images = []
    for img_path in all_step_images:
        if not os.path.isabs(img_path):
            full_path = os.path.join(image_base_dir, img_path)
        else:
            full_path = img_path
        resolved_images.append(full_path)

    messages = [{"role": "system", "content": system_prompt}]
    user_msg = {"role": "user", "content": user_content_text, "images": resolved_images}
    messages.append(user_msg)
    return messages


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from model response (replicates CognitiveMapManager logic)."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = list(fenced) if fenced else []
    stack, start = [], None
    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start = i
            stack.append(ch)
        elif ch == '}' and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _call_api(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 1.0,
    max_completion_tokens: int = 32768,
    image_base_dir: str = "",
) -> str:
    """Call the OpenAI-compatible API with Qwen-format messages (image paths in 'images' key)."""
    from vagen.inference.model_interface.openai.model import OpenAIModelInterface

    openai_messages = OpenAIModelInterface._convert_qwen_to_openai_format(messages)

    response = client.chat.completions.create(
        model=model_name,
        messages=openai_messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    return response.choices[0].message.content


def run_condition_b_scene(
    combo_dir: str,
    output_dir: str,
    client: OpenAI,
    model_name: str = "gemini-3-pro",
    temperature: float = 1.0,
    max_completion_tokens: int = 32768,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Run Condition B for a single scene, iterating through exploration steps."""
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

    for t, step_text in enumerate(steps, start=1):
        steps_so_far = steps[:t]
        images_so_far = step_images[:t]

        cb_prompt = get_condition_b_prompt(prev_map, enable_think=True)

        api_messages = _build_step_message(
            system_prompt=system_prompt,
            preamble=preamble,
            steps_so_far=steps_so_far,
            preamble_images=preamble_images,
            step_images_list=images_so_far,
            condition_b_prompt=cb_prompt,
            image_base_dir=image_base_dir,
        )

        response_text = None
        parsed_map = None
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
                parsed_map = _extract_json_from_text(response_text)
                if parsed_map is not None:
                    break
                else:
                    retries_this_step += 1
                    logger.warning(f"  Step {t} attempt {attempt}: JSON parse failed, retrying...")
            except Exception as e:
                retries_this_step += 1
                logger.error(f"  Step {t} attempt {attempt}: API error: {e}")
                time.sleep(2 ** attempt)

        total_retries += retries_this_step

        if parsed_map is None:
            logger.warning(f"  Step {t}: All retries exhausted, falling back to M_{{t-1}}")
            parsed_map = copy.deepcopy(prev_map)

        with open(os.path.join(output_dir, f"step_{t:02d}_map.json"), "w") as f:
            json.dump(parsed_map, f, indent=2, ensure_ascii=False)
        if response_text is not None:
            with open(os.path.join(output_dir, f"step_{t:02d}_response.txt"), "w") as f:
                f.write(response_text)

        step_results.append({
            "step": t,
            "retries": retries_this_step,
            "fallback_used": parsed_map is prev_map or (retries_this_step == max_retries and parsed_map == prev_map),
            "n_objects_in_map": len(parsed_map),
        })

        prev_map = parsed_map
        logger.info(f"  Step {t}/{len(steps)}: map has {len(parsed_map)} entries, retries={retries_this_step}")

    metadata = {
        "combo_dir": combo_dir,
        "n_steps": len(steps),
        "total_api_calls": total_calls,
        "total_retries": total_retries,
        "step_results": step_results,
        "final_map_n_objects": len(prev_map),
        "model_name": model_name,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def run_condition_b_all(
    results_dir: str = None,
    output_base: str = None,
    model_name: str = "gemini-3-pro",
    temperature: float = 1.0,
    max_completion_tokens: int = 32768,
    max_retries: int = 3,
    num_scenes: int = 25,
    start_scene: int = 0,
) -> List[Dict]:
    """Run Condition B across all scenes."""
    if results_dir is None:
        results_dir = str(_TOS_ROOT / "results")
    if output_base is None:
        output_base = str(_EXP_ROOT / "delta_map_updates" / "results" / "cogmap" / "condition_b")

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

        result = run_condition_b_scene(
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
