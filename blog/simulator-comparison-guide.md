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

Five tools. Five completely different bets on how to predict LLM inference performance.

**[Vidur](https://github.com/microsoft/vidur)** replays your exact workload through a discrete-event simulator, using profiled GPU kernel times to model every scheduling decision. High fidelity, but you need to profile each model first.

**[LLMServingSim](https://github.com/casys-kaist/LLMServingSim)** goes even deeper—simulating hardware down to the cycle level. The catch? Hours per experiment. Accuracy through exhaustive detail.

**[LLM-Optimizer](https://github.com/bentoml/llm-optimizer)** takes the opposite approach: pure math. Query HuggingFace for your model config, run the roofline calculation, get your answer in 0.1 seconds. No scheduling simulation, just architectural fundamentals.

**[AIConfigurator](https://github.com/ai-dynamo/aiconfigurator)** is NVIDIA's analytical optimizer, tuned specifically for H100 hardware. Fast estimates for configuration search—if you're on their supported hardware.

**[BLIS](https://github.com/inference-sim/inference-sim)** combines analytical roofline with discrete-event scheduling. It has multiple latency modes—for this article, we're focusing on two: Roofline (pure analytical baseline) and Evolved (adds physics-based modeling and learned corrections optimized on real data).

Five different philosophies. One question: which approach actually delivers?

<!-- Section 3: How We Tested -->
<!-- Section 4: Accuracy First: Who Gets It Right? -->
<!-- Section 5: Speed vs. Accuracy: The Pareto Frontier -->
<!-- Section 6: Coverage: Can It Even Run Your Workload? -->
<!-- Section 7: Which Simulator For Your Use Case -->
<!-- Section 8: The Bottom Line -->
