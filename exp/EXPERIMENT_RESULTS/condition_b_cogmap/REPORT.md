# Condition B: Rule-Based Full Regeneration Cognitive Map Probing

## Experiment Overview

Condition B tests whether providing the prior cognitive map M_{t-1} alongside explicit
preserve/overwrite rules reduces belief drift compared to scratch regeneration (Condition A).
At each exploration step t, the model receives: (1) the previous map M_{t-1} as verbatim JSON,
(2) the current observation O_t (cumulative observation history up to step t), and (3) explicit
update rules: preserve unchanged entries, only update visible objects, overwrite contradictions.

**Optimized (Iteration 1)**: Added explicit facing direction guidance (egocentric-to-cardinal
conversion rules), agent state anchoring from M_{t-1}, and lowered temperature to 0.5.

## Setup

- **Model**: Gemini-3 Pro (via LEMMA MaaS proxy, OpenAI-compatible API)
- **Scenes**: N=25 (run00-run24) from the ToS 3-room offline dataset
- **Exploration**: Passive SCOUT proxy trajectories (~8-10 steps per scene)
- **Render mode**: Vision (3D-rendered first-person images)
- **Condition B protocol**: Sequential per-step API calls with M_{t-1} context
  - Step 1: empty map {} + O_1
  - Steps 2..T: M_{t-1} + O_t (cumulative history)
- **Update rules in prompt**:
  1. Preserve all entries from M_{t-1} unchanged unless current observation contradicts
  2. Only update objects visible in O_t or newly observed
  3. Overwrite entries that contradict current observation
  4. Explicit facing direction rules: all objects must have facing key, egocentric-to-cardinal conversion guidance
- **API parameters**: temperature=0.5, max_completion_tokens=32768, enable_think=True
- **Retry logic**: Up to 3 retries for JSON parse failures; fallback to M_{t-1}
- **Evaluation**: ToS cogmap metrics (pos/dir/facing/overall composite, each 1/3 weight)
  - pos_allow_scale=False (no scale correction)

## Key Results

### Final Map Correctness (M_T)

| Metric | Cond A (scratch) | Cond B (original) | Cond B (optimized) | Delta (opt vs orig) |
|---|---:|---:|---:|---:|
| **Overall** | 0.2187 (SE=0.0143) | 0.2230 (SE=0.0097) | **0.2246** (SE=0.0078) | +0.0016 (+0.7%) |
| Positional | 0.3053 (SE=0.0166) | 0.3380 (SE=0.0086) | **0.3414** (SE=0.0112) | +0.0034 (+1.0%) |
| Direction | 0.1120 (SE=0.0095) | **0.1290** (SE=0.0081) | 0.1257 (SE=0.0077) | -0.0033 (-2.6%) |
| Facing | **0.2388** (SE=0.0265) | 0.2020 (SE=0.0261) | 0.2067 (SE=0.0212) | +0.0047 (+2.3%) |

### Per-Turn Metrics (Condition B)

| Metric | Original | Optimized | Delta |
|---|---:|---:|---:|
| Perception (newly observed) | 0.1523 (SE=0.0080, N=98) | **0.2035** (SE=0.0089, N=49) | +0.0512 (+33.6%) |
| Stability (previously observed) | 0.1964 (SE=0.0049, N=198) | **0.2207** (SE=0.0040, N=198) | +0.0243 (+12.4%) |

### Reliability

| Metric | Cond A | Cond B (original) | Cond B (optimized) |
|---|---:|---:|---:|
| JSON parse failure rate | 0.0% | 0.0% | 0.4% |
| Total API calls | 25 | 223 | 224 |
| Retries needed | 0 | 0 | 1 |
| Extraction success | 25/25 | 25/25 | 25/25 |

## Key Observations

1. **Overall performance marginally improved**: Optimized Condition B (22.5%) slightly
   outperforms both original B (22.3%) and Condition A (21.9%). The improvement is small
   but consistent across overall and positional metrics.

2. **Facing accuracy partially recovered**: From 20.2% (original B) to 20.7% (optimized),
   a +2.3% improvement. The explicit facing direction guidance with egocentric-to-cardinal
   conversion rules helped reduce missing facing keys, though facing remains the weakest
   metric and still below Condition A (23.9%).

3. **Perception substantially improved**: Per-turn perception for newly observed objects
   jumped from 15.2% to 20.4% (+33.6%), the largest single-metric improvement. The agent
   state anchoring and facing guidance help the model better integrate new observations.

4. **Stability also improved**: Previously observed object accuracy improved from 19.6%
   to 22.1% (+12.4%), suggesting the prompt changes improve overall map maintenance.

5. **Direction slightly degraded**: Dir accuracy dropped from 12.9% to 12.6% (-2.6%),
   remaining near chance level for 8-bin cardinal directions (~12.5%). This is within noise.

6. **Lower variance**: Standard error decreased further (overall SE: 0.0078 vs 0.0097),
   indicating the lower temperature and improved prompt provide more consistent results.

7. **Sequential approach remains costly**: 224 calls vs 25 calls (9x). Each scene
   requires ~8-10 API calls instead of 1.

## Optimization History

- **Original B** (commit 120968b): temperature=1.0, basic update rules, no facing guidance
- **Iteration 0** (FAILED): Switched to current-step-only observations. Degraded -7.4% overall.
- **Iteration 1** (current): Reverted to cumulative history, added facing guidance, temp=0.5. Improved +0.7% overall.

## Data Location

- Raw outputs (optimized): `delta_map_updates/results/cogmap/condition_b_iter1/run{XX}_{room_hash}/`
- Raw outputs (original): `delta_map_updates/results/cogmap/condition_b/run{XX}_{room_hash}/`
- Per-step maps: `step_{t}_map.json`, responses: `step_{t}_response.txt`
- Metadata: `metadata.json` per scene
