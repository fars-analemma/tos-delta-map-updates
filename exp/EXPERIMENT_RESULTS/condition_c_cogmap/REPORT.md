# Condition C: Delta-Map Updates Cognitive Map Probing

## Experiment Overview

Condition C tests whether reducing the model's output to only changed entries (a compact delta JSON)
further reduces transcription errors beyond Condition B's full-map regeneration. At each exploration
step t, the model receives: (1) the previous map M_{t-1} as verbatim JSON, (2) the current
observation O_t (cumulative history), and (3) the same preserve/overwrite rules as Condition B.
However, instead of outputting a full updated map, the model outputs only a delta JSON:
`{"updates": {"obj_name": {"position": [x,y], "facing": "east"}, ...}}`. The delta is applied
programmatically: M_t = Apply(M_{t-1}, delta_t).

## Setup

- **Model**: Gemini-3 Pro (via LEMMA MaaS proxy, OpenAI-compatible API)
- **Scenes**: N=25 (run00-run24) from the ToS 3-room offline dataset
- **Exploration**: Passive SCOUT proxy trajectories (~8-10 steps per scene)
- **Render mode**: Vision (3D-rendered first-person images)
- **Condition C protocol**: Sequential per-step API calls with M_{t-1} context
  - Step 1: empty map {} + O_1 -> delta
  - Steps 2..T: M_{t-1} + O_t -> delta, then M_t = Apply(M_{t-1}, delta)
- **Delta output format**: `{"updates": {"obj": {"field": value, ...}, ...}}`
  - Only changed/new entries included; omitted entries preserved from M_{t-1}
  - Empty delta `{"updates": {}}` if nothing changes
- **Update rules in prompt**: Same as Condition B (preserve, evidence restriction, conflict resolution)
  - Explicit facing direction rules with egocentric-to-cardinal conversion guidance
  - Agent state anchoring from M_{t-1}
- **API parameters**: temperature=1.0, max_completion_tokens=32768, enable_think=True
- **Retry logic**: Up to 3 retries for invalid delta JSON; fallback to empty delta (preserve M_{t-1})
- **Evaluation**: ToS cogmap metrics (pos/dir/facing/overall composite, each 1/3 weight)

## Key Results

### Final Map Correctness (M_T)

| Metric | Cond A (scratch) | Cond B (optimized) | Cond C (delta) | C vs B |
|---|---:|---:|---:|---:|
| **Overall** | 0.2187 (SE=0.0143) | 0.2246 (SE=0.0078) | **0.2278** (SE=0.0100) | +0.0032 (+1.4%) |
| Positional | 0.3053 (SE=0.0166) | **0.3414** (SE=0.0112) | 0.3413 (SE=0.0097) | -0.0001 (~0%) |
| Direction | 0.1120 (SE=0.0095) | **0.1257** (SE=0.0077) | 0.1190 (SE=0.0082) | -0.0067 (-5.3%) |
| Facing | 0.2388 (SE=0.0265) | 0.2067 (SE=0.0212) | **0.2231** (SE=0.0227) | +0.0164 (+7.9%) |

### Per-Turn Metrics

| Metric | Cond B (optimized) | Cond C (delta) | Delta |
|---|---:|---:|---:|
| Perception (newly observed) | **0.2035** (SE=0.0089, N=49) | 0.1687 (SE=0.0088, N=80) | -0.0348 (-17.1%) |
| Stability (previously observed) | **0.2207** (SE=0.0040, N=198) | 0.2077 (SE=0.0049, N=198) | -0.0130 (-5.9%) |

### Reliability

| Metric | Cond A | Cond B (optimized) | Cond C (delta) |
|---|---:|---:|---:|
| JSON parse failure rate | 0.0% | 0.4% | 0.4% |
| Total API calls | 25 | 224 | 224 |
| Retries needed | 0 | 1 | 1 |
| Extraction success | 25/25 | 25/25 | 25/25 |

### Output Length Comparison

| Metric | Cond B | Cond C | Reduction |
|---|---:|---:|---:|
| Total response chars | 571,905 | 516,775 | 9.6% |
| Avg chars per step | 2,565 | 2,317 | 9.7% |

### Delta Sparsity Statistics

| Metric | Value |
|---|---:|
| Mean updates per step | 3.90 |
| Median updates per step | 3.0 |
| Std | 3.09 |
| Min / Max | 1 / 15 |
| Steps with 1-2 updates | 43.5% |
| Steps with 3-5 updates | 33.6% |
| Steps with 6+ updates | 22.9% |

## Key Observations

1. **Overall performance slightly improved**: Condition C (22.8%) marginally outperforms both
   Condition B optimized (22.5%) and Condition A (21.9%). The improvement over B is small
   (+1.4%) and within standard error overlap.

2. **Facing accuracy improved over B**: From 20.7% (B optimized) to 22.3% (C), a +7.9%
   improvement. The delta format appears to reduce facing-related transcription errors by
   avoiding re-stating unchanged facing values. Still below Condition A (23.9%).

3. **Positional accuracy unchanged**: Nearly identical between B and C (34.1% both), suggesting
   position output quality is not affected by the delta format.

4. **Direction slightly degraded**: From 12.6% (B) to 11.9% (C), a -5.3% drop. Near chance
   level for 8-bin cardinal directions (~12.5%), so this is within noise.

5. **Per-turn perception lower**: C (16.9%) vs B (20.4%). However, N=80 perception events
   for C vs N=49 for B (different counting due to delta-based key tracking), making direct
   comparison of these per-turn metrics less meaningful between conditions.

6. **Output length reduced by ~10%**: Average response length drops from 2565 to 2317 chars
   per step (9.7% reduction). The delta format does produce shorter outputs, though the
   reduction is modest because the model's think block still accounts for most tokens.

7. **Sparse updates confirmed**: Median 3 updates per step, with 43.5% of steps updating
   only 1-2 objects. This validates the "sparse evidence" premise: most steps provide
   information about only a small fraction of the map.

8. **Equally reliable**: Same JSON parse failure rate (0.4%) and retry count (1) as
   Condition B, indicating the delta format does not introduce additional parsing fragility.

## Data Location

- Raw outputs: `delta_map_updates/results/cogmap/condition_c/run{XX}_{room_hash}/`
- Per-step deltas: `step_{t}_delta.json`
- Per-step applied maps: `step_{t}_map.json`
- Per-step raw responses: `step_{t}_response.txt`
- Per-step applied map as response: `step_{t}_applied_response.txt`
- Metadata: `metadata.json` per scene
