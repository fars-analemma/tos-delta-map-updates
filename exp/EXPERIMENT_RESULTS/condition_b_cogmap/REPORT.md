# Condition B: Rule-Based Full Regeneration Cognitive Map Probing

## Experiment Overview

Condition B tests whether providing the prior cognitive map M_{t-1} alongside explicit
preserve/overwrite rules reduces belief drift compared to scratch regeneration (Condition A).
At each exploration step t, the model receives: (1) the previous map M_{t-1} as verbatim JSON,
(2) the current observation O_t (cumulative observation history up to step t), and (3) explicit
update rules: preserve unchanged entries, only update visible objects, overwrite contradictions.

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
- **API parameters**: temperature=1.0, max_completion_tokens=32768, enable_think=True
- **Retry logic**: Up to 3 retries for JSON parse failures; fallback to M_{t-1}
- **Evaluation**: ToS cogmap metrics (pos/dir/facing/overall composite, each 1/3 weight)
  - pos_allow_scale=False (no scale correction)

## Key Results

### Final Map Correctness (M_T)

| Metric | Cond A (scratch) | Cond B (rule-based) | Delta |
|---|---:|---:|---:|
| **Overall** | 0.2187 (SE=0.0143) | **0.2230** (SE=0.0097) | +0.0043 |
| Positional | 0.3053 (SE=0.0166) | **0.3380** (SE=0.0086) | +0.0327 |
| Direction | 0.1120 (SE=0.0095) | **0.1290** (SE=0.0081) | +0.0170 |
| Facing | **0.2388** (SE=0.0265) | 0.2020 (SE=0.0261) | -0.0368 |

### Per-Turn Metrics (Condition B only)

| Metric | Mean | SE | N |
|---|---:|---:|---:|
| Perception (newly observed) | 0.1523 | 0.0080 | 98 |
| Stability (previously observed) | 0.1964 | 0.0049 | 198 |

### Reliability

| Metric | Cond A | Cond B |
|---|---:|---:|
| JSON parse failure rate | 0.0% | 0.0% |
| Total API calls | 25 | 223 |
| Retries needed | 0 | 0 |
| Extraction success | 25/25 | 25/25 |

## Key Observations

1. **Overall performance is comparable**: Condition B (22.3%) is marginally better than
   Condition A (21.9%), but the difference is not statistically significant at this sample size.
   Providing M_{t-1} with explicit rules does not clearly improve overall map quality.

2. **Positional accuracy improved**: Condition B shows a meaningful gain in positional
   accuracy (33.8% vs 30.5%), suggesting that carrying forward prior position estimates
   helps anchor spatial reasoning.

3. **Direction accuracy improved**: A moderate gain from 11.2% to 12.9%, still near
   chance level for 8-bin cardinal directions (~12.5%).

4. **Facing accuracy slightly degraded**: Condition B (20.2%) is lower than Condition A
   (23.9%). This may indicate that the preserve rules introduce some rigidity that
   prevents correcting earlier facing errors, or that the sequential update approach
   carries forward facing mistakes.

5. **Lower variance**: Condition B has notably lower standard errors (overall SE: 0.0097
   vs 0.0143), indicating more consistent performance across scenes. The prior map
   provides a stabilizing anchor.

6. **Perfect JSON reliability**: Zero JSON parse failures across 223 API calls,
   identical to Condition A's 25 calls. The Condition B prompt format is well-handled.

7. **Perception vs Stability**: Per-turn perception (15.2%) for newly observed objects
   is lower than stability (19.6%) for previously observed objects. This suggests the
   model is slightly better at preserving prior entries than accurately placing new ones.

8. **Sequential approach increases API cost**: 223 calls vs 25 calls (8.9x). Each scene
   requires ~8-10 API calls instead of 1.

## Data Location

- Raw outputs: `delta_map_updates/results/cogmap/condition_b/run{XX}_{room_hash}/`
- Per-step maps: `step_{t}_map.json`, responses: `step_{t}_response.txt`
- Metadata: `metadata.json` per scene
- Run summary: `delta_map_updates/results/cogmap/condition_b/run_summary.json`
