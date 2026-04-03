# Blog Article Design: LLM Simulator Comparison

**Date:** 2026-04-03
**Type:** Magazine feature / Technical blog post
**Target Length:** 1800-2000 words (~7-minute read)
**Audience:** Hybrid (executives + data scientists/practitioners)

---

## Goals

1. **Educational** - Help readers understand what factors matter when evaluating LLM inference simulators
2. **Actionable** - Provide clear, data-driven recommendations for different use cases
3. **Unbiased** - Present all simulators (including BLIS) as peers, letting data speak for itself
4. **Engaging** - Accessible for executives while providing depth for practitioners

---

## Key Principles

- **Accuracy-first focus** - Lead with prediction accuracy (MAPE), but also cover speed, coverage, and versatility
- **Scenario-driven** - Use concrete examples throughout to reduce cognitive load and increase relatability
- **Progressive disclosure** - Executives can skim headers and visuals; practitioners can read deeply
- **Conversational tone** - Active voice, punchy transitions, personality without sacrificing credibility
- **Visual storytelling** - Each major section anchored by a figure from results_iter26/

---

## Simulators Covered

All comparisons include only the following BLIS variants:
- **BLIS-Roofline** - Pure analytical baseline
- **BLIS-Evolved (iter26)** - Learned corrections with activated TP All-Reduce modeling

**Other simulators:**
- Vidur (trace-driven, discrete-event simulation)
- LLM-Optimizer (analytical roofline estimator)
- AIConfigurator (analytical estimator, H100-focused)
- LLMServingSim (trace-driven, hours-per-run)

---

## Article Structure

### Section 1: Hook & Problem Setup (200-250 words)

**Purpose:** Grab both audiences immediately with value proposition and relatable scenario

**Content:**
- Opening line: Direct value prop about saving time/money with simulators
- Embedded scenario: Data scientist deploying Mixtral-8x7B, needs to choose GPU count and batch size
- Stakes: Five simulators with wildly different approaches, no obvious winner
- Tease: We tested them head-to-head across 38 real experiments

**Tone:** Conversational, sets up problem clearly

**Key elements:**
- Money/time savings (executive hook)
- Specific technical scenario (practitioner hook)
- Promise of rigorous evaluation ahead

---

### Section 2: Meet the Contenders (150-200 words)

**Purpose:** Introduce all five simulators objectively, provide mental model for categorization

**Content:**
- Group by approach type:
  - **Trace-driven:** Vidur, LLMServingSim (replay actual request patterns)
  - **Analytical:** LLM-Optimizer, AIConfigurator (roofline models)
  - **Hybrid:** BLIS-Roofline, BLIS-Evolved (analytical + learned corrections)
- 1-2 sentences per simulator with one concrete differentiator
- Foreshadow trade-offs: trace-driven = high fidelity but slow; analytical = fast but less accurate; hybrid = middle ground

**Tone:** Educational, neutral presentation

**Key elements:**
- No bias in presentation order or description
- Help readers build mental model (categories)
- Set up for findings that follow

---

### Section 3: How We Tested (150-200 words)

**Purpose:** Build credibility, introduce MAPE simply

**Content:**
- Test scope:
  - 38 real-world experiments on production hardware using vLLM v0.15.1
  - 7 models: Llama-3.1-8B, Qwen3-14B, CodeLlama-34B, Llama-2-70B (dense); Mixtral-8x7B, Mixtral-8x22B, Llama-4-Scout (MoE)
  - 4 workload types: general-purpose, code generation, roleplay, reasoning
  - 3 GPU types: H100, A100-80GB, L40S
  - Config sweeps: tensor parallelism, batch sizes, memory settings, CPU offload
- **MAPE Callout Box:**
  - Title: "How We Measure Accuracy"
  - Definition: "MAPE (Mean Absolute Percentage Error) tells us how far predictions are from reality."
  - Example: "If predicted = 1000ms, actual = 1500ms → 33% MAPE"
  - Benchmark: "Under 20% is excellent. Over 50% means you're better off guessing."

**Tone:** Credible but accessible

**Key elements:**
- Emphasize "real-world" and "production hardware" (not toy benchmarks)
- MAPE definition with concrete example
- Set expectations for good/bad accuracy

---

### Section 4: Accuracy First—Who Gets It Right? (350-400 words)

**Purpose:** Lead with the most important comparison dimension (accuracy)

**Opening beat:** "The accuracy spread was shocking: from 15% error to 91% error on the same workload."

**Visual anchor:** Figure 1 (fig1_model_sensitivity.png) + sim comparison figures

**Content structure:**

1. **The Winners (BLIS-Evolved)**
   - Overall: 37% MAPE (TTFT: 24%, E2E: 13%)
   - Complex MoE + high parallelism: 15-16% error
   - Example: Mixtral-8x7B with TP=4 achieved 15.84% E2E MAPE

2. **The Middle Ground**
   - LLM-Optimizer, AIConfigurator: 40-60% MAPE depending on workload
   - Good enough for rough estimates, not production capacity planning

3. **The Struggles**
   - BLIS-Roofline: 60%+ MAPE on complex workloads (shows value of learned corrections)
   - LLMServingSim: 91% error on shared MoE experiment (severe overestimation despite being trace-driven)

4. **Where Accuracy Matters Most**
   - Model architecture: MoE harder to predict (expert routing, uneven compute)
   - Hardware config: Multi-GPU (TP≥4) introduces communication overhead many miss
   - Workload type: Code generation and reasoning (variable lengths) vs. general-purpose (predictable)

**Embedded scenario:**
"Imagine capacity planning for peak load. A 15% error means ordering 115 GPUs instead of 100—safe buffer. A 90% error means ordering 190 GPUs or catastrophic under-provisioning."

**Tone:** Direct, uses specific numbers, builds surprise

**Key elements:**
- Lead with dramatic finding (15% to 91% spread)
- Name specific MAPE numbers for each simulator tier
- Use figures from results_iter26/figures/sim_comparisons/
- Concrete consequence of error (GPU ordering)
- No bias: present BLIS-Evolved as winner but explain why (data-driven)

---

### Section 5: Speed vs. Accuracy—The Pareto Frontier (250-300 words)

**Purpose:** Show the accuracy/speed trade-off, help readers understand their position on the curve

**Opening beat:** "But accuracy isn't everything. If a simulator takes hours to run, you might as well run the real experiment."

**Visual anchor:** Figure 5 (fig5_pareto.png)

**Content structure:**

1. **The Speed Spectrum**
   - LLM-Optimizer: 0.1 seconds (23,000× faster than real experiments)
   - BLIS-Roofline: 1.6 seconds (770× speedup)
   - BLIS-Evolved: 1.8 seconds (667× speedup)
   - Vidur: 9.9 seconds (121× speedup)
   - LLMServingSim: Hours per experiment (700× slower than BLIS)

2. **The Trade-off**
   - Ultra-fast analytical → sacrifice accuracy (40-60% MAPE)
   - BLIS-Evolved → sweet spot: 667× speedup + 37% MAPE
   - Trace-driven → slower but NOT always more accurate (LLMServingSim's 91% error proves fidelity ≠ accuracy)

3. **The Pareto Insight**
   - Explain Pareto frontier concept simply: "dominated quadrant" = worse on BOTH dimensions
   - Most simulators escape domination (good at speed OR accuracy)
   - BLIS-Evolved escapes by being strong on both

**Embedded scenario:**
"For rapid config search (trying 100+ configurations), LLM-Optimizer's 0.1-second runtime is game-changing. For production capacity planning where a 15% error costs millions, spending 2 seconds per simulation is worth it."

**Tone:** Analytical but conversational, shows trade-offs clearly

**Key elements:**
- Dramatic speedup numbers (23,000×!)
- Visual concept: Pareto frontier, dominated quadrant
- Two contrasting scenarios showing when speed vs. accuracy matters
- Acknowledge LLM-Optimizer's strength (speed) while noting accuracy cost

---

### Section 6: Coverage—Can It Even Run Your Workload? (250-300 words)

**Purpose:** Show that accuracy/speed don't matter if simulator can't run your experiment

**Opening beat:** "Here's the catch: accuracy and speed don't matter if a simulator can't run your experiment at all."

**Content structure:**

1. **Coverage by the Numbers**
   - Vidur: 4 of 38 experiments (11%)
     - Requires pre-profiled models, no MoE support, no L40S
   - AIConfigurator: ~19 of 38 experiments (50%)
     - H100 only, no MoE
   - LLM-Optimizer: ~38 of 38 experiments (100%)
     - Treats MoE as dense
   - LLMServingSim: 1 experiment (3%)
     - Prohibitive runtime limits practical coverage
   - BLIS variants: 38 experiments (100%)
     - All models, all hardware, all configs

2. **What Gets Excluded**
   - MoE models: Vidur and AIConfigurator can't run them at all
   - Multi-GPU communication: Only some simulators model TP All-Reduce overhead
   - Hardware diversity: Vidur and AIConfigurator limited to H100/A100
   - Advanced vLLM features: CPU offload, GPU memory limits mostly ignored

3. **Why Coverage Matters**
   - Research exploration requires broad support
   - Multi-cloud deployments need hardware flexibility
   - Cost optimization often compares dense vs. MoE

**Embedded scenario:**
"You're comparing Mixtral-8x7B (MoE) vs. Llama-2-70B (dense) for cost-efficiency. Two simulators can't run MoE at all. A third treats it as dense (wrong architecture). Only two can actually compare apples-to-apples."

**Tone:** Pragmatic, shows coverage as a gate

**Key elements:**
- Concrete numbers: X of 50 experiments
- Specific exclusions (MoE, hardware, features)
- Scenario showing why coverage blocks decision-making
- Reference docs/simulator-limitations.md methodology

---

### Section 7: Which Simulator For Your Use Case (400-450 words)

**Purpose:** Deliver actionable recommendations organized by use case

**Opening beat:** "So which simulator should you choose? It depends on what you're trying to do."

**Content structure:** Three use cases with tiered recommendations

---

#### Use Case 1: Production Capacity Planning

**What you need:** High accuracy for infrastructure sizing, contract negotiation, peak load planning. Errors cost real money.

**Tier 1 Recommendation: BLIS-Evolved**
- 37% overall MAPE, 13-16% on E2E latency
- Handles all model types (dense, MoE), all hardware (H100, A100, L40S)
- 667× speedup = answers in ~2 seconds
- Models vLLM serving parameters (batch size, memory, offload)
- Activated TP All-Reduce physics (critical for multi-GPU accuracy)

**Tier 2 Alternative: Vidur**
- IF your model is pre-profiled AND you're on H100/A100
- High-fidelity vLLM scheduler emulation
- Slower (121× speedup) but accurate for supported models
- Coverage gap is the blocker (11% of experiments)

**Avoid:**
- LLMServingSim (hours per run)
- Pure analytical estimators (40-60% error too high for capacity planning)

---

#### Use Case 2: Config Search & Optimization

**What you need:** Speed + reasonable accuracy for sweeping 100+ configurations (batch size, TP, memory allocation). Need fast iteration.

**Tier 1 Recommendation: LLM-Optimizer**
- 0.1-second runtime = thousands of configs per hour
- 40-50% MAPE isn't perfect but identifies trends
- Broad model support (queries HuggingFace Hub automatically)
- Ideal for exploration phase, not final sizing

**Tier 2 Alternative: BLIS-Roofline**
- 1.6-second runtime (still fast for sweeps)
- ~60% MAPE but better coverage than LLM-Optimizer
- Handles L40S and MoE architectural nuances

**Avoid:**
- Trace-driven simulators (too slow for large sweeps)
- LLMServingSim (prohibitive runtime for config search)

---

#### Use Case 3: Research & AI-Driven Algorithm Discovery

**What you need:** Coverage + flexibility for exploring novel architectures (MoE variants, custom attention), multi-GPU communication patterns, hardware-algorithm co-design. Simulator can't break on edge cases.

**Tier 1 Recommendation: BLIS-Evolved**
- 100% coverage of test set (all architectures, hardware, configs)
- Extensible: roofline + learned corrections (can retrain on new data)
- Physics-based terms (TP All-Reduce) make behavior interpretable
- Fast enough for RL-driven tuning loops (667× speedup)

**Tier 2 Alternative: LLM-Optimizer**
- For pure dense models only
- Fast enough for optimization loops
- Limited to analytical roofline (no queueing dynamics)

**Avoid:**
- Simulators with hard coverage limits (Vidur: 11%, AIConfigurator: 50%)
- LLMServingSim (runtime prohibits exploration)

---

**Tone:** Prescriptive but acknowledges trade-offs

**Key elements:**
- Clear use case headers
- "What you need" frames the decision criteria
- Tiered structure (best, alternative, avoid)
- Specific scenarios embedded in each use case
- Concrete numbers in recommendations (MAPE, speedup, coverage %)

---

### Section 8: The Bottom Line (150-200 words)

**Purpose:** Provide quick reference and final guidance

**Content:**

**Decision Matrix:**
| Your Priority | Recommended Simulator | Why |
|---------------|----------------------|-----|
| Best Accuracy | BLIS-Evolved | 37% MAPE, 15-16% on MoE+TP |
| Fastest | LLM-Optimizer | 0.1s, 23,000× speedup |
| Best Balance | BLIS-Evolved | 667× speedup + 37% MAPE + full coverage |
| Research/Edge Cases | BLIS-Evolved | Handles MoE, multi-GPU, all hardware |
| Config Exploration | LLM-Optimizer | Speed optimized for large sweeps |

**Final Thought:**
"The simulator landscape isn't one-size-fits-all. LLM-Optimizer excels at rapid config exploration. Vidur offers high-fidelity scheduling for its supported models. But for most production use cases—capacity planning, cost optimization, multi-model comparisons—BLIS-Evolved delivers the accuracy, speed, and coverage to make decisions you can trust."

**Tone:** Balanced, authoritative, helpful

**Key elements:**
- Scannable decision matrix
- Acknowledge multiple valid choices
- Clear guidance for "most users" without dismissing alternatives
- End on credibility note (trust decisions)

---

## Visual Assets

All figures sourced from `results_iter26/figures/`:

1. **Section 4 (Accuracy):**
   - `fig1_model_sensitivity.png` - BLIS-Roofline vs BLIS-Evolved across 7 models
   - `sim_comparisons/blis_vs_vidur.png`
   - `sim_comparisons/blis_vs_llm_optimizer.png`
   - `sim_comparisons/blis_vs_aiconfigurator.png`
   - `sim_comparisons/blis_vs_llmservingsim.png`

2. **Section 5 (Speed vs Accuracy):**
   - `fig5_pareto.png` - Accuracy vs. Speed Pareto frontier

3. **Section 6 (Coverage):**
   - Optional: Create simple coverage bar chart or use text-based comparison

4. **Section 7 (Use Cases):**
   - Optional: Use case icons or decision tree diagram

5. **Section 8 (Bottom Line):**
   - Decision matrix table (text-based, not figure)

---

## Data Sources

- **Accuracy metrics:** `results_iter26/error_records.csv`, `results_iter26/sim_comparison_captions.md`
- **Speed metrics:** `results_iter26/runtime.csv`, `results_iter26/figure_captions.md` (Table 1)
- **Coverage data:** `docs/simulator-limitations.md`
- **Methodology:** `results_iter26/figure_captions.md`, `results_iter26/sim_comparison_captions.md`

---

## MAPE Numbers Reference

From `results_iter26/sim_comparison_captions.md`:

- **BLIS-Evolved (iter26):** 37.42% overall (TTFT: 24.34%, E2E: 13.09%)
  - Mixtral-8x7B TP4: 15.84% E2E Mean MAPE
- **BLIS-Roofline:** 60.19% overall (baseline)
  - Mixtral-8x7B TP4: 97.57% E2E Mean MAPE
- **LLMServingSim:**
  - Mixtral-8x7B TP4: 91.42% E2E Mean MAPE
- **LLM-Optimizer:** 40-60% MAPE range (38 experiments)
- **AIConfigurator:** 40-60% MAPE range (19 experiments)
- **Vidur:** Limited data (4 experiments on general-lite only)

---

## Runtime Numbers Reference

From `results_iter26/figure_captions.md` Table 1:

| Simulator | Median Runtime (s) | Speedup vs. Real |
|---|---|---|
| BLIS-Roofline | 1.6 | 770× |
| BLIS-Evolved (iter26) | 1.8 | 667× |
| Vidur | 9.9 | 121× |
| LLM-Optimizer | 0.1 | 23,094× |
| AIConfigurator | 3.3 | 363× |
| LLMServingSim | ~hours | 0.0014× (700× slower than BLIS) |

---

## Coverage Numbers Reference

From `results_iter26/sim_comparison_captions.md`:

- **Vidur:** 4 experiments (11%) - general-lite only, 2 models
- **AIConfigurator:** 19 experiments (50%) - H100, dense models only
- **LLM-Optimizer:** 38 experiments (100%) - broadest coverage, treats MoE as dense
- **LLMServingSim:** 1 experiment (3%) - single cluster dataset experiment
- **BLIS variants:** 38 experiments (100%)

---

## Writing Guidelines

1. **Conversational but credible**
   - Use active voice: "We tested five simulators" not "Five simulators were tested"
   - Direct address when appropriate: "You're deploying Mixtral..." not "One might deploy..."
   - Contractions OK: "can't", "isn't", "doesn't"

2. **Concrete over abstract**
   - Specific numbers: "15% error" not "low error"
   - Real scenarios: "You're sizing infrastructure for Mixtral-8x7B" not "Users need accurate predictions"
   - Named examples: "Mixtral-8x7B with TP=4" not "complex MoE workloads"

3. **Progressive disclosure**
   - Lead sentences can stand alone (executives skim)
   - Supporting details follow (practitioners read)
   - Example: "BLIS-Evolved achieved 37% MAPE. [new sentence] On complex MoE workloads with TP=4, it hit 15-16% error—6× more accurate than the baseline roofline."

4. **Punchy transitions**
   - "But accuracy isn't everything..."
   - "Here's the catch..."
   - "So which simulator should you choose?"
   - Avoid academic: "Moreover", "Furthermore", "In addition"

5. **Avoid jargon escalation**
   - Define MAPE once, use freely after
   - Spell out acronyms first use: "Tensor Parallelism (TP)"
   - Keep technical depth in supporting sentences, not leads

6. **Show, don't just tell**
   - Use figures to carry story
   - Reference specific data points from captions
   - Example: "LLMServingSim overestimated by 91% on the Mixtral experiment—nearly double the actual latency."

---

## Success Criteria

The article succeeds if:

1. **Executive can skim** headers, callout box, decision matrix and walk away with:
   - Clear understanding that simulator choice matters
   - One or two memorable data points (15% vs. 91% error, 23,000× speedup)
   - A decision heuristic (use X for capacity planning, Y for config search)

2. **Practitioner can deep-read** and walk away with:
   - Understanding of evaluation methodology (MAPE, test scope)
   - Specific MAPE/speed/coverage numbers for each simulator
   - Confidence to choose the right tool for their use case
   - Appreciation for trade-offs (speed vs. accuracy, coverage vs. runtime)

3. **Both audiences trust the evaluation:**
   - Unbiased presentation (all simulators introduced equally)
   - Data-driven (specific numbers, not claims)
   - Credible methodology (49 real experiments on vLLM v0.15.1, production hardware)

4. **Article is engaging:**
   - Not boring (punchy writing, surprising findings, concrete scenarios)
   - Not overwhelming (progressive disclosure, clear structure)
   - Flows naturally (magazine feature style, not academic paper)

---

## Implementation Notes

- Target format: Markdown for easy conversion to blog platform
- Include alt text for all figures (accessibility)
- Embedded figures as images with captions
- Decision matrix as markdown table (renders everywhere)
- MAPE callout box as blockquote or highlighted section
- Total reading time estimate: 7 minutes (include at top)

---

## Out of Scope

- Deep technical dive into BLIS architecture (keep it 1-2 sentences)
- Detailed methodology of other simulators (reference their papers if needed)
- Raw data tables (use figures instead)
- Comprehensive benchmark suite discussion (focus on key findings)
- Future work or research directions (stay practical)
- Performance tuning tips for simulators (focus on selection, not optimization)
