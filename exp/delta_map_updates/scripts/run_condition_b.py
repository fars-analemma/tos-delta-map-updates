#!/usr/bin/env python3
# Entry point for running Condition B (rule-based full regeneration) experiment.
# Usage: python -m delta_map_updates.scripts.run_condition_b [--num N] [--start S]

import argparse
import logging
import os
import sys
from pathlib import Path

_EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_EXP_ROOT))

import dotenv
dotenv.load_dotenv(str(_EXP_ROOT / ".env"))
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("LEMMA_MAAS_API_KEY", ""))

from delta_map_updates.runners.condition_b_runner import run_condition_b_all


def main():
    parser = argparse.ArgumentParser(description="Run Condition B cognitive map probing")
    parser.add_argument("--num", type=int, default=25, help="Number of scenes")
    parser.add_argument("--start", type=int, default=0, help="Starting scene index")
    parser.add_argument("--model", type=str, default="gemini-3-pro")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    results = run_condition_b_all(
        model_name=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.max_tokens,
        max_retries=args.max_retries,
        num_scenes=args.num,
        start_scene=args.start,
    )

    total_calls = sum(r.get("total_api_calls", 0) for r in results if isinstance(r, dict))
    total_retries = sum(r.get("total_retries", 0) for r in results if isinstance(r, dict))
    print(f"\nDone. {len(results)} scenes, {total_calls} API calls, {total_retries} retries.")


if __name__ == "__main__":
    main()
