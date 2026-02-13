## Experiments

### Experimental Setup

**Base Models:**

| Model | Size | Download Link | Notes |
|---|---:|---|---|
| Gemini-3 Pro | - | https://ai.google.dev/ | Used in the ToS paper; strong VLM baseline |

(Verification may optionally also run GPT-5.2 as a secondary check, but the decisive experiment uses a single model.)

**Training Data (if applicable):**

No training data needed — **inference only**.

**Other Resources (if applicable):**

- ToS code (release branch) and evaluation scripts: **[GitHub - mll-lab-nu/Theory-of-Space](./references/GitHub%20-%20mll-lab-nu%20Theory-of-Space/meta/meta_info.txt)**
- ToS offline dataset (100 runs, includes false-belief images): **[MLL-Lab/tos-data](./references/MLL-Lab%20tos-data%20%C2%B7%20Datasets%20at%20Hugging%20Face/meta/meta_info.txt)**

**Resource Estimate**:

- **GPU budget**: 0 GPU-hours
- **API calls (rough)**:
  - Choose `N=25` scenes (run00–run24) for verification.
  - Use passive **SCOUT** proxy trajectories (ToS’s scripted baseline: 360° sweep + room-visitation) with length ≈ 9–12 steps per scene.
  - Cognitive-map update calls: ~`N * steps * 3 conditions` ≈ 25 * 12 * 3 = 900 calls.
  - False-belief revision calls: similar order (another ~900).
  - Total: ~1.8k multimodal calls.

### Benchmarks and Metrics

| Benchmark | Description | Metrics | Split | Download Link | Evaluation Script |
|---|---|---|---|---|---|
| Theory-of-Space (vision world) cognitive map probing | Build a global map from sequential partial views | Final map correctness (pos/dir/facing composite), perception/local↔global (optional) | run00–run24 | [tos-data](./references/MLL-Lab%20tos-data%20%C2%B7%20Datasets%20at%20Hugging%20Face/meta/meta_info.txt) | ToS repo (`scripts/SpatialGym/`) with prompt modifications |
| Theory-of-Space false-belief revision (vision world) | After k=4 object changes, revise belief | **Identification F1** (which objects changed), **belief inertia** (pos/ori; ↓ better) per ToS | run00–run24 | same | same |

**Evaluation Scripts:**
- Use ToS’s released pipeline (`scripts/SpatialGym/spatial_run.py`) and modify only the cognitive-map probing prompts and the state-passing logic for conditions B/C.

### Main Results

**Decision rule (verification):**

- Primary: On the false-belief revision task, **B** reduces belief inertia and improves identification F1 vs **A**.
- Secondary (format effect): **C** improves over **B** on the same metrics (or matches B with fewer invalid outputs / fewer retries).
- Refute: If **B** does not improve final map correctness and inertia vs **A**.

#### Results Table

| Method | Base Model | Benchmark | Final map correctness (↑) | False-belief F1 (↑) | Pos inertia (↓) | Ori inertia (↓) | Source | Notes |
|---|---|---|---:|---:|---:|---:|---|---|
| Scratch regeneration (A) | Gemini-3 Pro | ToS vision (N=25) | **TBD** | **TBD** | **TBD** | **TBD** | - | Needs re-run (subset + prompt differs from paper reporting) |
| Full regen + preserve/overwrite rules (B) | Gemini-3 Pro | ToS vision (N=25) | **TBD** | **TBD** | **TBD** | **TBD** | - | Proposed |
| **Delta updates (C)** | Gemini-3 Pro | ToS vision (N=25) | **TBD** | **TBD** | **TBD** | **TBD** | - | Proposed |

Reference numbers from the ToS paper (full setting; not directly comparable to N=25 subset): Gemini-3 Pro vision-world final correctness 52.1% and false-belief positional inertia 51.1% / orientation inertia 14.4% (Tables 5 and 7 in the ToS paper).

### Ablation Studies

| Variant | What’s changed | Expected finding |
|---|---|---|
| B vs A | Adds explicit state + preserve/overwrite update rule | Improves correctness and reduces inertia if regeneration noise / inconsistent conflict handling matters |
| C vs B | Changes output format to delta | If transcription errors matter, C ≥ B on correctness/inertia and/or requires fewer retries |

### Analysis (Optional)

- **Edit magnitude**: average number of objects changed per step under B/C; if it is small, it supports the “sparse evidence” premise.
- **Failure stratification**: split by objects with facing direction vs not; hypothesis: gains are larger for orientation (where drift/inertia are worst).

---