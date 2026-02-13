## Related Work

### Field Overview

**Spatial reasoning benchmarks for VLMs and embodied agents.** Many benchmarks test spatial reasoning from static images or short clips, but fewer isolate long-horizon belief construction under partial observability. ToS contributes by probing explicit cognitive maps and measuring belief revision under environment shifts.

**Explicit spatial representations (cognitive maps, scene graphs).** A common strategy to improve spatial reasoning is to convert perception into structured intermediate representations—scene graphs, maps, or coordinate lists—then condition a VLM/LLM on this representation. This can improve spatial queries but introduces new failure modes in representation generation, maintenance, and revision.

**Incremental structured-state updates.** Outside spatial reasoning, recent work shows that LLMs often struggle to repeatedly regenerate large structured objects, motivating incremental update formats (diffs, patches, constrained edits). This proposal applies that insight to spatial belief probing.

### Related Papers

- **[Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?](./references/THEORY%20OF%20SPACE%20CAN%20FOUNDATION%20MODELS%20CON-STRUCT%20SPATIAL%20BELIEFS%20THROUGH%20ACTIVE%20EXPLORATION/meta/meta_info.txt)**: Introduces ToS, cognitive-map probing, and the belief inertia metric we target.
- **[Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](https://arxiv.org/abs/2412.14171)**: Studies spatial memory and recall in MLLMs; motivates explicit belief representations.
- **[SpatialVLM](https://arxiv.org/abs/2401.12168)**: Enhances VLMs with spatial reasoning capabilities; representative of spatially focused VLM adaptation.
- **[SpatialRGPT](https://arxiv.org/abs/2406.01584)**: Grounded spatial reasoning with region-centric prompting; highlights grounding limitations.
- **[EmbodiedBench](https://arxiv.org/abs/2502.09560)**: Benchmarks embodied decision making for multimodal LLM agents.
- **[Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making](https://arxiv.org/abs/2410.07166)**: Evaluates LLMs in embodied settings; complements ToS’s belief-centric focus.
- **[Embodied Question Answering](https://arxiv.org/abs/1711.11543)**: Classic task-driven exploration benchmark; contrasts with ToS’s task-agnostic exploration.
- **[IQA: Visual Question Answering in Interactive Environments](https://arxiv.org/abs/1712.03316)**: Early interactive VQA benchmark; related environment-query framing.
- **[ALFRED](https://arxiv.org/abs/1912.01734)**: Instruction following in interactive environments; emphasizes long-horizon interaction.
- **[TEACh](https://arxiv.org/abs/2110.00534)**: Task-driven embodied agents with dialogue; highlights language grounding needs.
- **[EXCALIBUR](https://arxiv.org/abs/2303.07342)**: Encourages embodied exploration; relates to ToS’s exploration efficiency axis.
- **[Reverie](https://arxiv.org/abs/1904.10151)**: Remote embodied referring expression; spatial grounding under partial observability.
- **[Seeing from Another Perspective: Evaluating Multi-view Understanding in MLLMs](https://arxiv.org/abs/2504.15280)**: Multi-view spatial evaluation; complements ToS with passive multi-view settings.
- **[MMSI-Bench](https://arxiv.org/abs/2505.23764)**: Multi-image spatial intelligence benchmark; broader coverage of spatial skills.
- **[3DSRBench](https://arxiv.org/abs/2412.07825)**: 3D spatial reasoning benchmark; focuses on 3D geometry reasoning.
- **[InternSpatial](https://arxiv.org/abs/2506.18385)**: Large dataset for spatial reasoning in VLMs; data-centric angle.
- **[What’s “Up” with Vision-Language Models?](https://arxiv.org/abs/2310.19785)**: Diagnoses spatial failure modes of VLMs; motivates targeted interventions.
- **[SpatialTree](https://arxiv.org/abs/2512.20617)**: Capability-centric taxonomy of spatial abilities; situates ToS at agentic competence.
- **[Ego3D-VLM / Ego3D-Bench](https://arxiv.org/abs/2509.06266)**: Uses textual/JSON cognitive maps to improve ego-centric multi-view spatial reasoning.
- **[3DThinker](https://arxiv.org/abs/2510.18632)**: Distills 3D latent representations to improve spatial reasoning; a model-internal alternative to explicit maps.
- **[JSON Whisperer: Efficient JSON Editing with LLMs](https://arxiv.org/abs/2510.04717)**: Uses diff/patch-style edits for structured outputs; provides tools and motivation for incremental updates.
- **[GraphPad: Inference-Time 3D Scene Graph Updates for Embodied Question Answering](https://arxiv.org/abs/2506.01174)**: Updates structured scene graphs online; conceptually similar but different benchmark and modality.
- **[CoSPlan: Corrective Sequential Planning via Scene Graph Incremental Updates](https://arxiv.org/abs/2512.10342)**: Uses incremental scene-graph updates for sequential planning; supports the incremental-update framing.

### Taxonomy

| Family / cluster | Core idea | Representative papers | Benchmarks / evaluation | Known limitations |
|---|---|---|---|---|
| Belief-centric embodied evaluation | Evaluate active exploration + belief construction + revision | ToS, EXCALIBUR, EmbodiedBench | ToS, embodied QA tasks | Conflates exploration vs belief maintenance; belief externalization may be lossy |
| Explicit spatial representations | Convert perception to structured maps/graphs | Ego3D-VLM, GraphPad | Ego3D-Bench, OpenEQA | Errors in representation can dominate; maintaining consistency over time is hard |
| Spatial VLM adaptation | Train/augment VLMs for spatial reasoning | SpatialVLM, SpatialRGPT, 3DThinker | Multiple spatial benchmarks | Often targets static questions, not belief revision |
| Incremental structured editing | Avoid full regeneration; output diffs/patches | JSON Whisperer | JSON editing tasks | Output validity and conflict handling remain challenges |

### Closest Prior Work

1) **Theory of Space (ToS)**: Defines cognitive-map probing and belief inertia metrics. It reports belief drift and inertia but does not propose interventions beyond analysis.

2) **Ego3D-VLM**: Builds explicit cognitive maps to help spatial reasoning, but focuses on constructing maps from multi-view perception modules rather than maintaining/revising a belief under changes.

3) **GraphPad / CoSPlan**: Maintain/update structured scene representations over time, but they are evaluated on embodied QA or planning tasks rather than ToS’s belief probing and false-belief revision.

4) **JSON Whisperer**: Studies incremental structured edits for JSON documents, but not in embodied spatial belief settings.

### Comparison Table

| Related work | What it does | Key limitation | What we change | Why ours should win |
|---|---|---|---|---|
| ToS | Benchmarks belief construction + revision; probes full-map JSON | No intervention; probing requires full regeneration | Add explicit belief-update protocol (external state) | Removes regeneration noise; enforces conflict revision |
| Ego3D-VLM | Uses cognitive maps to improve spatial QA | Not about belief revision; maps built per query | Apply incremental belief updates across steps | Targets drift/inertia over time |
| GraphPad | Updates 3D scene graph via APIs | Different task; heavier perception stack | Apply update interface to ToS map probing | ToS offers direct stability/inertia metrics |
| JSON Whisperer | JSON diff/patch editing with LLMs | Not spatial; no inertia metric | Use delta updates for cognitive maps | Smaller outputs reduce transcription errors |

---