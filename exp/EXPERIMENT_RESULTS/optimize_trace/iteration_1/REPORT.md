# Condition B Optimization Iteration 1: Cumulative History + Facing Guidance

## Experiment Overview

This iteration reverts the failed iteration 0 changes (current-step-only observation)
back to cumulative history, while adding explicit facing direction guidance to the prompt
and lowering temperature to 0.5.

## Changes from Original Condition B

1. **Reverted to cumulative history**: Sends all steps 1..t at step t (matching original),
   instead of current-step-only from iteration 0.
2. **Added facing direction guidance**: New "Facing Direction Rules (CRITICAL)" section
   explaining how to convert egocentric observations to cardinal directions, and requiring
   ALL objects to have a "facing" key.
3. **Added agent state section**: Previous map's agent position/facing is highlighted
   at the top of the update rules for spatial context.
4. **Lowered temperature**: 1.0 -> 0.5 for reduced noise across sequential steps.

## Key Results

| Metric | Cond A | Orig B | Iter 1 B | Iter1 vs Orig |
|---|---:|---:|---:|---:|
| **Overall** | 0.2187 | 0.2230 | **0.2246** | +0.7% |
| Positional | 0.3053 | 0.3380 | **0.3414** | +1.0% |
| Direction | 0.1120 | 0.1290 | 0.1257 | -2.6% |
| Facing | 0.2388 | 0.2020 | **0.2067** | +2.3% |

### Per-Turn Metrics

| Metric | Orig B | Iter 1 B |
|---|---:|---:|
| Perception (new objects) | 0.1523 | **0.2035** |
| Stability (prior objects) | 0.1964 | **0.2207** |

### Per-Scene Comparison (vs Original B)

- Scenes improved: 13/25
- Scenes degraded: 12/25
- Mean improvement when improved: +0.053
- Mean degradation when degraded: -0.049

## Key Observations

1. **Marginal overall improvement**: +0.7% overall, driven by small gains in pos and facing.
2. **Facing improved but still weak**: 0.2020 -> 0.2067 (+2.3%). The explicit facing guidance
   helped slightly but the fundamental difficulty of egocentric-to-allocentric conversion remains.
3. **Perception substantially improved**: 0.1523 -> 0.2035 (+33.6%). The model is much better
   at placing newly observed objects, likely due to the agent state anchoring.
4. **Stability improved**: 0.1964 -> 0.2207 (+12.4%). Prior entries are better preserved.
5. **Direction slightly degraded**: 0.1290 -> 0.1257 (-2.6%). Near chance level for 8-bin directions.
6. **Temperature 0.5 reduces variance**: SE decreased from 0.0097 to 0.0078 on overall.
7. **Mixed per-scene results**: Almost even split (13 vs 12), suggesting the changes help in
   some configurations but not others.
