# The LLM Simulator Showdown: Which Tool Actually Delivers?

**Reading time:** 7 minutes
**Published:** April 2026
**Topics:** LLM inference, simulator evaluation, capacity planning

---

## Table of Contents
1. [Why Simulator Choice Matters](#why-simulator-choice-matters)
2. [Meet the Contenders](#meet-the-contenders)
3. [How We Tested](#how-we-tested)
4. [Accuracy First: Who Gets It Right?](#accuracy-first-who-gets-it-right)
5. [Speed vs. Accuracy: The Pareto Frontier](#speed-vs-accuracy-the-pareto-frontier)
6. [Coverage: Can It Even Run Your Workload?](#coverage-can-it-even-run-your-workload)
7. [Which Simulator For Your Use Case](#which-simulator-for-your-use-case)
8. [The Bottom Line](#the-bottom-line)

---

<a name="why-simulator-choice-matters"></a>
## Why Simulator Choice Matters

Choosing the right LLM inference simulator can save weeks of experimentation and thousands in compute costs. But which one actually works?

Imagine you're deploying Mixtral-8x7B for your AI-powered coding assistant. Four GPUs or eight? Batch size 2048 or 8192? Running real experiments could take days and cost thousands. A simulator promises answers in minutes—if you trust its predictions.

We tested five popular simulators head-to-head across 38 real-world experiments on production hardware. The results? Wildly different approaches, dramatically different accuracy, and no obvious winner—until you look at the data.

Here's what we found: accuracy ranged from 15% error to 91% error on the same workload. Speed varied from milliseconds to hours per simulation. And coverage? Some tools couldn't run two-thirds of our experiments at all.

This guide breaks down which simulator to use for capacity planning, config search, and research—backed by hard data from 38 experiments across seven models, four workload types, and three GPU types.

<a name="meet-the-contenders"></a>
## Meet the Contenders

Five simulators, three distinct approaches:

**Trace-driven simulators** replay actual request patterns through discrete-event simulation:
- **Vidur** emulates vLLM's scheduler directly, requires pre-profiled models for each architecture
- **LLMServingSim** uses fine-grained hardware simulation but takes hours per experiment

**Analytical estimators** use roofline models and hardware specs for fast predictions:
- **LLM-Optimizer** queries HuggingFace Hub on the fly, supports broad model coverage
- **AIConfigurator** focuses on H100 hardware with dense model optimizations

**Hybrid approaches** combine analytical baselines with learned corrections:
- **BLIS-Roofline** provides a pure analytical baseline
- **BLIS-Evolved** adds physics-based TP All-Reduce modeling and learned correction terms on top of roofline

The theory: trace-driven means high fidelity but slow execution. Analytical means lightning-fast but potentially less accurate. Hybrid tries to split the difference. But what does the data actually show?
<!-- Section 3: How We Tested -->
<!-- Section 4: Accuracy First: Who Gets It Right? -->
<!-- Section 5: Speed vs. Accuracy: The Pareto Frontier -->
<!-- Section 6: Coverage: Can It Even Run Your Workload? -->
<!-- Section 7: Which Simulator For Your Use Case -->
<!-- Section 8: The Bottom Line -->
