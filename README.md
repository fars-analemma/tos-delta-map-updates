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
