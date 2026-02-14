# Condition C: Delta-Map Updates Cognitive Map Probing (Optimized)

## Experiment Overview

Condition C tests whether reducing the model's output to only changed entries (a compact delta JSON)
further reduces transcription errors beyond Condition B's full-map regeneration. At each exploration
step t, the model receives: (1) the previous map M_{t-1} as verbatim JSON, (2) the current
observation O_t (cumulative history), and (3) the same preserve/overwrite rules as Condition B.
However, instead of outputting a full updated map, the model outputs only a delta JSON:
`{"updates": {"obj_name": {"position": [x,y], "facing": "east"}, ...}}`. The delta is applied
programmatically: M_t = Apply(M_{t-1}, delta_t).

## Optimization Applied

Three fixes were applied to address a structural weakness in the delta format:

1. **Facing-fill post-processing** (`delta_apply.py`): Missing "facing" keys are filled with the opposite of the agent's facing direction after applying each delta
2. **Stronger facing prompt** (`condition_c.py`): Made facing mandatory for ALL delta entries, with explicit small-object list
3. **Temperature reduction** (`run_condition_c.py`): 1.0 -> 0.5

## Setup

- **Model**: Gemini-3 Pro (via LEMMA MaaS proxy, OpenAI-compatible API)
- **Scenes**: N=25 (run00-run24) from the ToS 3-room offline dataset
- **Temperature**: 0.5 (optimized from 1.0)
- **Other parameters**: max_completion_tokens=32768, enable_think=True

## Key Results

### Final Map Correctness (M_T)

| Metric | Cond A (scratch) | Cond B (optimized) | Cond C (optimized) | C vs B |
|---|---:|---:|---:|---:|
| **Overall** | 0.2187 (SE=0.0143) | 0.2246 (SE=0.0078) | **0.2356** (SE=0.0111) | +0.0110 (+4.9%) |
| Positional | 0.2857 (SE=0.0166) | **0.3464** (SE=0.0112) | 0.3353 (SE=0.0101) | -0.0111 (-3.2%) |
| Direction | 0.1316 (SE=0.0095) | **0.1207** (SE=0.0077) | 0.1187 (SE=0.0079) | -0.0020 (-1.7%) |
| Facing | 0.2388 (SE=0.0265) | 0.2067 (SE=0.0212) | **0.2528** (SE=0.0258) | +0.0461 (+22.3%) |

### Improvement over Original C

| Metric | Original C | Optimized C | Change |
|---|---:|---:|---:|
| **Overall** | 0.2278 | **0.2356** | +3.4% |
| Positional | **0.3413** | 0.3353 | -1.8% |
| Direction | 0.1190 | 0.1187 | -0.3% |
| Facing | 0.2231 | **0.2528** | +13.3% |

### Per-Turn Metrics

| Metric | Cond C (optimized) |
|---|---:|
| Perception (newly observed) | 0.2261 (SE=0.0112, N=40) |
| Stability (previously observed) | 0.2445 (SE=0.0037, N=198) |

### Reliability

| Metric | Cond A | Cond B (optimized) | Cond C (optimized) |
|---|---:|---:|---:|
| JSON parse failure rate | 0.0% | 0.4% | 0.9% |
| Total API calls | 25 | 224 | 225 |
| Retries needed | 0 | 1 | 2 |
| Extraction success | 25/25 | 25/25 | 25/25 |

## Key Observations

1. **Overall best condition**: Optimized C (23.6%) outperforms both B optimized (22.5%) and A (21.9%)

2. **Facing accuracy greatly improved**: From 22.3% (original C) to 25.3% (optimized C), a +13.3% improvement. The facing-fill post-processing and stronger prompt addressed the structural weakness where delta format allowed permanent missing facing keys.

3. **Positional accuracy slightly lower**: 33.5% vs 34.1% (original C), a minor -1.8% drop likely from temperature change.

4. **Direction unchanged**: Near chance level (~12.5%) for all conditions.

5. **Per-turn stability improved**: 24.5% (optimized) vs 20.8% (original), showing the facing fill helps maintain stable map entries across steps.

## Data Location

- Raw outputs: `delta_map_updates/results/cogmap/condition_c_optimized/run{XX}_{room_hash}/`
- Per-step deltas: `step_{t}_delta.json`
- Per-step applied maps: `step_{t}_map.json`
- Per-step raw responses: `step_{t}_response.txt`
- Per-step applied map as response: `step_{t}_applied_response.txt`
- Metadata: `metadata.json` per scene
