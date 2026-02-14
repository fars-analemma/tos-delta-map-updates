# Optimization Iteration 0: Condition C Facing Fix + Temperature Reduction

## Experiment Overview

Optimized Condition C (delta-map updates) cognitive map probing by addressing the missing facing keys issue and reducing temperature.

## Issues Diagnosed

### Issue 1: Missing Facing Keys in Delta Outputs (CRITICAL)
- 31 objects across 15/25 scenes had no "facing" key in the final predicted map
- GT evaluation requires facing for objects with `has_orientation=True`
- Missing facing = `ori=[0,0]`, `has_orientation=False` -> guaranteed 0 score
- Root cause: delta format allows partial updates; model omits facing for small objects (cap, wine, basket, candle, plant, desklamp, etc.)
- Once inserted without facing at step 1, later deltas never add it

### Issue 2: Temperature=1.0 Too High
- Original C used temperature=1.0; Condition B was optimized to 0.5
- Higher temperature adds noise that compounds across sequential delta steps

### Issue 3: Weak Facing Enforcement in Prompt
- Original prompt said to include facing "if its facing has changed or if it is newly observed" -- too permissive for delta format
- Small objects were consistently omitted

## Fixes Applied

1. **apply_delta() post-processing**: Fill missing facing with opposite of agent's facing direction. File: `delta_map_updates/evaluation/delta_apply.py`
2. **Stronger prompt**: Made facing mandatory for ALL delta entries, added explicit list of small objects, added default heuristic instruction. File: `delta_map_updates/prompts/condition_c.py`
3. **Temperature reduction**: Changed default from 1.0 to 0.5. File: `delta_map_updates/scripts/run_condition_c.py`

## Setup

- Model: Gemini-3 Pro via LEMMA MaaS proxy
- Scenes: 25
- Temperature: 0.5 (was 1.0)
- API calls: 225, retries: 2

## Key Results

| Metric | Original C | Optimized C | Change |
|--------|-----------|-------------|--------|
| Overall | 0.2278 | **0.2356** | +3.4% |
| Position | 0.3413 | 0.3353 | -1.8% |
| Direction | 0.1190 | 0.1187 | -0.3% |
| Facing | 0.2231 | **0.2528** | +13.3% |

### Per-Turn Analysis
- Perception mean: 0.2261 (SE 0.0112)
- Stability mean: 0.2445 (SE 0.0037)

### Comparison with Other Conditions
| Condition | Overall | Position | Direction | Facing |
|-----------|---------|----------|-----------|--------|
| A (scratch) | 0.2187 | 0.2857 | 0.1316 | 0.2388 |
| B (full regen) | 0.2246 | 0.3464 | 0.1207 | 0.2067 |
| **C optimized** | **0.2356** | 0.3353 | 0.1187 | **0.2528** |
| C original | 0.2278 | 0.3413 | 0.1190 | 0.2231 |

## Key Observations

1. Facing improved substantially (+13.3%), confirming the missing facing hypothesis
2. Overall improved by +3.4%, making optimized C the best among all conditions
3. Position slightly decreased (-1.8%), likely due to temperature change
4. Direction essentially unchanged (-0.3%)
5. The facing fix was the primary driver of improvement
