## Success Criteria

**Criterion 1: Belief revision improves under false belief**
- Hypothesis: Providing explicit prior map and conflict-based overwriting (B) reduces ToS belief inertia and improves identification of changed objects vs scratch regeneration (A).
- Validation: Directional improvement on inertia and identification F1 across scenes, with paired bootstrap confidence interval excluding 0 for the mean difference.

**Criterion 2: Delta outputs reduce structured-state transcription errors**
- Hypothesis: Delta updates (C) match or improve over B while producing fewer invalid JSON outputs / fewer retries.
- Validation: C ≥ B on correctness/inertia, and retry rate decreases.

---