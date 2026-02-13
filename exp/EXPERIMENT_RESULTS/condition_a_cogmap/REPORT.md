# Condition A: Scratch Regeneration Cognitive Map Probing

## Experiment Overview

Condition A replicates the original Theory-of-Space (ToS) cognitive-map probing protocol.
At the end of exploration, the model receives the full observation history and generates
a complete global cognitive map from scratch. This is the control condition against which
Conditions B (rule-based full regeneration) and C (delta-map updates) are compared.

## Setup

- **Model**: Gemini-3 Pro (via LEMMA MaaS proxy, OpenAI-compatible API)
- **Scenes**: N=25 (run00-run24) from the ToS 3-room offline dataset
- **Exploration**: Passive SCOUT proxy trajectories (~9-12 steps per scene)
  - 360-degree sweep at each location + fixed room-visitation order
- **Render mode**: Vision (3D-rendered first-person images)
- **Cogmap prompt**: Standard ToS global cogmap prompt (BASE_COGMAP_PROMPT + COGMAP_INSTRUCTION_GLOBAL_ONLY)
- **API parameters**: temperature=1.0, max_completion_tokens=32768, enable_think=True
- **Evaluation**: ToS cogmap metrics (pos/dir/facing/overall composite, each 1/3 weight)
  - pos_allow_scale=False (no scale correction)

## Key Results

| Metric | Mean | SE |
|---|---:|---:|
| **Overall (composite)** | **0.2187** | 0.0143 |
| Positional accuracy | 0.3053 | 0.0166 |
| Direction accuracy | 0.1120 | 0.0095 |
| Facing accuracy | 0.2388 | 0.0265 |

- **JSON parse failure rate**: 0.00% (0/25 scenes required retries)
- **Extraction success**: 25/25 (100%)

## Key Observations

1. **Overall score (21.9%)** is substantially below the ToS paper's reported 52.1% for
   Gemini-3 Pro on active exploration. This is expected because:
   - Passive mode presents all observations at once in a single prompt (no interactive exploration)
   - The model must infer absolute coordinates from sequential relative observations
   - No scale correction is applied (pos_allow_scale=False)

2. **Positional accuracy (30.5%)** is the strongest component, indicating the model
   captures rough spatial relationships. Direction accuracy (11.2%) is near chance for
   8-bin cardinal directions (~12.5%), suggesting pairwise directional relationships
   are poorly captured.

3. **Facing accuracy (23.9%)** is moderate but highly variable across scenes (SE=0.027).
   Some scenes achieve 55.6% facing accuracy while others score 0%.

4. **Zero JSON failures** indicate the model consistently produces valid JSON maps with
   the standard ToS prompt format when thinking mode is enabled.

5. These baseline numbers establish the control condition. Conditions B and C should
   ideally improve on these metrics by providing the prior map state as context.

## Reproduction

```bash
source setup_env.sh
cd Theory-of-Space

# Step 1: Generate SCOUT trajectories
python scripts/SpatialGym/spatial_run.py \
  --phase explore --exp-type passive --render-mode vision \
  --num 25 --model-name gemini-3-pro \
  --data-dir room_data/3-room/ --output-root results/ --inference-mode direct

# Step 2: Create synthetic turn logs for passive mode evaluation
# (handled by delta_map_updates/runners/cogmap_runner.py::ensure_passive_turn_logs)

# Step 3: Run cogmap probing
python scripts/SpatialGym/spatial_run.py \
  --phase cogmap --exp-type passive --render-mode vision \
  --num 25 --model-name gemini-3-pro \
  --data-dir room_data/3-room/ --output-root results/ --inference-mode direct
```

## Data Location

- Raw outputs: `Theory-of-Space/results/gemini-3-pro/<room_hash>/vision/passive/think/scout/`
- Per-sample metrics: `delta_map_updates/results/cogmap/condition_a/results.json`
- Aggregated env_data: `Theory-of-Space/results/gemini-3-pro/env_data.json`
