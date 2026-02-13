# Delta-Map Belief Updates for Stable Spatial Revision in Theory of Space

## Scope and Constraints

- **Paper Type**: Short paper
- **Target Venues**: NeurIPS, ICML, ICLR, ACL, EMNLP (or similar top AI conferences)
- **Core constraint**: Fully automated evaluation (no human-in-the-loop)
- **Infrastructure constraint**: No interactive simulator required; use the released **offline** Theory-of-Space dataset (pre-rendered images + metadata)
- **Compute constraint**: ≤768 A100 GPU-hours (expected **0 GPU-hours**; API-only)

## Introduction

### Context and Motivation

Embodied agents (robots, navigation assistants, and computer-use systems) must act under partial observability: the agent cannot see the whole world state at once, so it must build and maintain an internal **spatial belief** (a map-like hypothesis about object locations and orientations) from a sequence of observations. Recent multimodal foundation models (vision-language models, VLMs) can answer many spatial questions when given the relevant views, but it is less clear whether they can **construct, maintain, and revise** a coherent spatial belief over time.

**Theory of Space (ToS)** is a recent benchmark that isolates this challenge by asking models to explore a multi-room environment, then evaluating (i) how well their induced belief supports downstream spatial queries and (ii) how accurate their **explicitly probed cognitive map** is at each step. ToS further tests belief revision with a **false-belief paradigm**: after exploration, the environment changes (k=4 objects moved or rotated), and the agent must re-explore and update its belief. Empirically, ToS finds severe failures in vision-based settings: even strong models show low final map correctness and large belief inertia under changes (e.g., in vision-world false-belief, GPT-5.2 has positional inertia 68.9% and orientation inertia 34.7%—**lower is better**; Table 7 in the ToS paper).

### The Problem

ToS attributes these failures to perception bottlenecks (especially orientation) and to **belief instability**, where previously correct information is overwritten over time. However, ToS’s cognitive-map probing also creates a practical systems question: the benchmark repeatedly asks an LLM/VLM to output a *complete* global JSON cognitive map from a growing interaction history.

In many LLM systems, repeatedly regenerating a large structured state is error-prone: the model may inadvertently mutate fields unrelated to the current observation (“copy noise”), which can appear as belief drift. In addition, when observations conflict with a prior belief (false belief), the model may not reliably apply a consistent conflict resolution rule, leading to **belief inertia**.

If a large portion of ToS “belief drift/inertia” comes from the *interface* by which the belief is externalized (full regeneration from long context), then a simple state-management intervention—treating the map as an external state and updating it incrementally—could substantially improve ToS map correctness and false-belief revision without changing the underlying model.

### Key Insight and Hypothesis

**Key insight**: For structured beliefs like ToS’s cognitive map, the model should not need to rewrite the entire belief every step. Most steps only provide evidence about a small subset of objects (those currently visible). If we provide the previous belief explicitly and require the model to update only the parts supported by current evidence, we may reduce accidental overwrites and make belief revision under conflicts more reliable.

**Hypothesis**: On Theory-of-Space (vision world), providing the previous cognitive map as explicit context and enforcing a “preserve unless evidenced / overwrite on contradiction” update rule will (i) increase final cognitive-map correctness and (ii) reduce false-belief inertia. A further hypothesis is that asking for **delta updates** (a small JSON object describing only changed entries) reduces transcription errors beyond the same rule with full-map regeneration.

Why this could be wrong:
- Drift/inertia may be dominated by perception errors (wrong object identity/orientation) rather than map-regeneration errors.
- A “preserve unless evidenced” rule could increase apparent stability while hurting correction of earlier mistakes, potentially worsening final correctness.

---