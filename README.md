# Delta-Map Belief Updates for Stable Spatial Revision in Theory of Space

## Overview

This project experiments with belief-update interfaces for Theory-of-Space (ToS) cognitive maps.
Three conditions are compared: (A) scratch regeneration, (B) rule-based full regeneration,
(C) delta-map updates. All experiments use Gemini-3 Pro via API (0 GPU-hours).

## Quick Start

```bash
# Activate the environment (sets up venv + API keys)
source setup_env.sh

# Run ToS experiments (example)
cd Theory-of-Space
python scripts/SpatialGym/spatial_run.py \
    --phase all \
    --model-name gemini-3-pro \
    --num 25 \
    --data-dir room_data/3-room/ \
    --output-root results/ \
    --render-mode vision \
    --exp-type passive \
    --inference-mode batch
```

## Environment

- **Python**: 3.12.12 (venv at `.venv/`)
- **Activation**: `source setup_env.sh` (activates venv, loads .env, sets OPENAI_API_KEY for MaaS proxy)
- **Model**: `gemini-3-pro` via LEMMA MaaS proxy (OpenAI-compatible at `http://139.224.231.89:8765/v1`)

## Project Structure

```
exp/
├── .venv/                    # Python virtual environment
├── .env                      # API keys (LEMMA_MAAS_API_KEY, HF_TOKEN, etc.)
├── setup_env.sh              # Environment setup script (source this)
├── task_plan.json            # Experiment task plan with status tracking
├── Theory-of-Space/          # Official ToS repository (release branch)
│   ├── scripts/SpatialGym/   # Main experiment scripts
│   │   ├── spatial_run.py    # Main runner
│   │   └── base_model_config.yaml  # Model configs (gemini-3-pro added)
│   ├── vagen/                # Core library (model interfaces, env, evaluation)
│   ├── analysis/             # ToS analysis tools
│   └── room_data/3-room/     # Offline dataset (100 scenes: run00-run99)
├── delta_map_updates/        # Our experiment extensions
│   ├── prompts/              # Condition A/B/C cognitive-map probing prompts
│   ├── runners/              # Experiment runners (cogmap, false-belief)
│   ├── evaluation/           # Evaluation scripts (correctness, inertia, delta)
│   ├── analysis/             # Analysis (edit magnitude, failure stratification)
│   ├── scripts/              # Shell scripts for running experiments
│   ├── results/              # Raw outputs (cogmap/, falsebelief/)
│   └── configs/              # Experiment configuration files
├── EXPERIMENT_RESULTS/       # Formal experiment reports
└── idea/                     # Research proposal and references
```

## Passive Mode Workaround

The ToS passive pipeline stores all SCOUT observations in a single messages.json but does
not create `exploration_turn_logs.json` (needed for cogmap evaluation). Before running the
cogmap phase, use `delta_map_updates.runners.cogmap_runner.ensure_passive_turn_logs()` to
create synthetic turn log entries from `history_state.json`.

## Condition B Sequential Runner

Condition B uses a custom sequential runner (`delta_map_updates/runners/condition_b_runner.py`)
that decomposes passive SCOUT trajectories into per-step observations, calling the API at each
step with M_{t-1} as context. Key files:

- `delta_map_updates/prompts/condition_b.py`: Prompt with preserve/overwrite rules
- `delta_map_updates/runners/condition_b_runner.py`: Sequential runner (OpenAI client)
- `delta_map_updates/scripts/run_condition_b.py`: Entry point
- `delta_map_updates/evaluation/cogmap_eval.py`: Evaluation for sequential conditions

Run: `python -m delta_map_updates.scripts.run_condition_b --num 25`

### Optimization History (Condition B)

**Iteration 0** (FAILED): Switched to current-step-only observations + spatial anchoring + temp=0.3.
Overall dropped from 0.2230 to 0.2065 (-7.4%). Cumulative history was proven beneficial.

**Iteration 1** (current): Reverted to cumulative history. Added explicit facing direction guidance
with egocentric-to-cardinal conversion rules, agent state anchoring from M_{t-1}, temp=0.5.
Overall improved from 0.2230 to 0.2246 (+0.7%), facing from 0.2020 to 0.2067 (+2.3%),
perception from 0.1523 to 0.2035 (+33.6%).

- Code changes: `condition_b.py` (facing direction rules), `condition_b_runner.py` (reverted to cumulative),
  `run_condition_b.py` (temperature 0.5)
- Results: `EXPERIMENT_RESULTS/optimize_trace/iteration_1/`, `EXPERIMENT_RESULTS/condition_b_cogmap/`

## Condition C Sequential Runner (Delta-Map Updates)

Condition C uses a custom sequential runner (`delta_map_updates/runners/condition_c_runner.py`)
that is forked from Condition B. Instead of outputting a full map, the model outputs only a
compact delta JSON. The delta is applied programmatically via `apply_delta()`. Key files:

- `delta_map_updates/prompts/condition_c.py`: Delta-format prompt with same update rules as B
- `delta_map_updates/runners/condition_c_runner.py`: Sequential runner with delta parsing + application
- `delta_map_updates/evaluation/delta_apply.py`: `apply_delta(M_prev, delta)` and `validate_delta()`
- `delta_map_updates/scripts/run_condition_c.py`: Entry point
- `delta_map_updates/evaluation/cogmap_eval.py`: `evaluate_condition_c()` and `evaluate_condition_c_per_turn()`

Run: `python -m delta_map_updates.scripts.run_condition_c --num 25 --temperature 0.5`

### Optimization History (Condition C)

**Original**: temp=1.0, no facing-fill. Overall=0.2278, facing=0.2231.

**Iteration 0** (current, SUCCESS): Three fixes applied together:
1. Facing-fill post-processing in `apply_delta()`: fills missing facing with opposite of agent's facing direction
2. Stronger facing requirement in prompt: mandatory for ALL delta entries, explicit small-object list
3. Temperature 1.0 -> 0.5

Results: Overall improved from 0.2278 to 0.2356 (+3.4%), facing from 0.2231 to 0.2528 (+13.3%).
Optimized C is now the best condition overall (A=0.2187, B=0.2246, C=0.2356).

- Code changes: `delta_apply.py` (facing fill), `condition_c.py` (stronger prompt), `run_condition_c.py` (temp=0.5)
- Results: `EXPERIMENT_RESULTS/optimize_trace/iteration_0/`, `EXPERIMENT_RESULTS/condition_c_cogmap/`

## Key Files Modified in ToS

- `Theory-of-Space/scripts/SpatialGym/base_model_config.yaml`: Added `gemini-3-pro` entry using MaaS proxy base_url.

## API Configuration

The `gemini-3-pro` model is configured via the LEMMA MaaS proxy (OpenAI-compatible).
When `source setup_env.sh` is run, it sets `OPENAI_API_KEY` to the LEMMA MaaS key.
The ToS codebase's OpenAI model interface uses this key with the base_url from the config.

## Dataset

- Source: `MLL-Lab/tos-data` on Hugging Face
- Location: `Theory-of-Space/room_data/3-room/` (100 scenes: run00-run99)
- Each scene contains: `meta_data.json`, `falsebelief_exp.json`, agent/object images (*.png), false-belief images (*_fbexp.png)
- Experiments use scenes run00-run24 (N=25)

## Dependencies

All dependencies from the ToS `requirements.txt` are installed in `.venv/`.
Key packages: `openai`, `google-genai`, `numpy`, `scipy`, `pandas`, `matplotlib`, `Pillow`, `tqdm`, `PyYAML`, `hydra-core`, `omegaconf`.
The `tos` package is installed in editable mode from `Theory-of-Space/`.
