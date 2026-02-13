## Proposed Approach

### Overview

We propose a **belief-update interface** for Theory-of-Space cognitive maps that treats the global map as an external state `M_{t}` and updates it at each step using the latest observation `O_t`.

We test three conditions that differ only in how the model is prompted to update the map:

- **A (Scratch regeneration baseline)**: original ToS probing style—at each step, the model outputs a full global map from the interaction history.
- **B (Rule-based full regeneration)**: the model is given the previous map `M_{t-1}` and the new observation `O_t`, and must output a full updated map `M_t`, with explicit rules to preserve unchanged entries and overwrite contradicted entries.
- **C (Delta-map updates)**: the model is given `M_{t-1}` and `O_t`, and must output a compact JSON “delta” describing only the objects whose states should change; the evaluator applies the delta to form `M_t`.

### Method Details

**Belief state (global map) schema.** We use the ToS global cognitive map format:

```json
{
  "agent": {"position": [x,y], "facing": "north|south|east|west"},
  "obj_name": {"position": [x,y], "facing": "north|south|east|west"},
  "gate_name": {"position": [x,y], "facing": "north|south|east|west"}
}
```

(We follow ToS’s rule: include only observed objects; facing is required only for entities with a facing direction.)

**Condition A (scratch regeneration).** For each timestep `t`, provide the ToS observation history up to `t` (the same history ToS uses for probing) and ask the model to output the full global map.

**Condition B (full regeneration with explicit update rules).** For each timestep `t`, provide:
1) previous map `M_{t-1}` (verbatim JSON),
2) current observation `O_t` (image + text observation in ToS format),
3) update rules:
   - Preserve: copy all entries from `M_{t-1}` unchanged unless the current observation provides evidence they should change.
   - Evidence restriction: only update objects that are visible in `O_t` (or newly observed).
   - Conflict resolution: if `O_t` contradicts an entry for a visible object, overwrite that object’s state to be consistent with `O_t`.

Then ask for the full updated `M_t`.

**Condition C (delta-map updates).** Same inputs and rules as B, but the model outputs a compact delta JSON:

```json
{
  "updates": {
    "obj_name": {"position": [x,y], "facing": "east"},
    "obj2": {"position": [x,y]}
  }
}
```

The evaluator computes `M_t = Apply(M_{t-1}, delta)` by replacing only the keys in `updates`.

**Why delta might help beyond B.** Even if B has the right rule, generating a large JSON repeatedly can still introduce transcription errors (missing keys, accidental edits). Delta outputs reduce output length and isolate the model’s generation to the minimal set of updates.

**False-belief revision (dynamic update).** To test belief inertia, we use ToS’s released `falsebelief_exp.json` per scene (k=4 changed objects) and the corresponding post-change images (`*_fbexp.png`). We run the same three conditions during re-exploration (using a fixed proxy trajectory; see Experiments) and measure:
- identification F1 of changed objects,
- inertia metrics as defined in ToS (positional alignment + orientation inertia).

### Key Innovations

- A **verification-first causal test** for whether ToS belief drift/inertia is partly an artifact of full-map regeneration from long context, by comparing scratch regeneration vs explicit-map update.
- A simple, implementation-friendly **delta-map update interface** for cognitive-map probing that is model-agnostic (prompt-only; no fine-tuning).
- A focused evaluation on **dynamic belief revision** (false belief) where improvements cannot be explained by “never changing old entries,” because the task requires overwriting obsolete beliefs.

---