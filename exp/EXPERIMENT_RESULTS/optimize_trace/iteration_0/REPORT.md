# Optimization Iteration 0: Current-Step-Only Observation for Condition B

## Experiment Overview
Attempted to fix what appeared to be a bug in Condition B (rule-based full regeneration): the runner was sending cumulative observation history (steps 1..t) at each step t, rather than just the current observation O_t as specified in the research plan. The hypothesis was that this made M_{t-1} redundant (since the model already saw all observations) and prevented the update rules from working properly.

## Changes Made
1. **Runner fix** (`condition_b_runner.py`): Changed `_build_step_message` to accept only current step text/images instead of cumulative history
2. **Prompt enhancement** (`condition_b.py`): Added spatial anchoring section extracting agent position/facing from M_{t-1}, clarified that observation is current-step-only
3. **Temperature reduction** (`run_condition_b.py`): Lowered default from 1.0 to 0.3

## Key Results

| Metric | Condition A | Original B | Optimized B | Opt vs Orig |
|--------|-------------|------------|-------------|-------------|
| Overall | 0.2187 | 0.2230 | 0.2065 | -7.4% |
| Position | 0.3053 | 0.3380 | 0.3224 | -4.6% |
| Direction | 0.1120 | 0.1290 | 0.1253 | -2.8% |
| Facing | 0.2388 | 0.2020 | 0.1718 | -14.9% |

Per-scene wins: Opt B beats Orig B in 11/25 scenes, loses in 14/25.

## Key Observations
1. **Optimization failed**: All metrics degraded compared to both Original B and Condition A
2. **Facing accuracy most affected**: Dropped 14.9% vs Original B and 28.1% vs Condition A
3. **The cumulative history was actually helpful**: Even though it deviates from the theoretical "M_{t-1} + O_t" design, providing the full observation history gives the model more context to build accurate maps
4. **M_{t-1} alone is insufficient memory**: The structured JSON map format loses information (spatial nuance, visual details) that raw observation text preserves
5. **Temperature change alone unlikely the cause**: The main driver of degradation is the observation scope change, not temperature

## Conclusion
The "bug" (cumulative history) was actually a feature. The model benefits from seeing all observations because:
- M_{t-1} in JSON format is a lossy compression of spatial knowledge
- The model cannot perfectly interpret spatial relationships from JSON alone
- Raw text observations contain richer spatial cues that help map generation
