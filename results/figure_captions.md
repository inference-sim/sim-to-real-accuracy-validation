# Publication Figures

---

### Figure 1: Prediction Error Across Model Architectures

![Figure 1](figures/fig1_model_sensitivity.png)

MAPE of each simulator across 7 model architectures (7B–8x22B parameters, dense and MoE) on H100 with default serving configuration and general-purpose workload. Simulators absent from a model lack support for that architecture (e.g., Vidur does not model all architectures in our evaluation set).

---

### Figure 2: Prediction Error Across GPU Types

![Figure 2](figures/fig2_hardware_portability.png)

Median MAPE across all models for three GPU types (H100, A100, L40S) with default configuration. Aggregation uses the median to reduce sensitivity to outlier models. Tests whether simulators calibrated on one GPU family transfer to others with different memory bandwidth and compute characteristics.

---

### Figure 3: Prediction Error Across Workload Types

![Figure 3](figures/fig3_workload_sensitivity.png)

Median MAPE for four workload types (general-purpose, code generation, roleplay, reasoning) on H100 with default configuration. Results are aggregated over four representative models chosen to span the architecture space: a small dense model (Llama-3.1-8B), a medium dense model (Qwen3-14B), a quantized MoE (Llama-4-Scout-17B-16E, FP8), and a large MoE (Mixtral-8x22B).

---

### Figure 4a: Config Sensitivity — Dense Model

![Figure 4a](figures/fig4a_config_dense.png)

MAPE under controlled single-parameter sweeps of serving configuration (TP, chunk size, GPU memory utilization, KV cache offloading) on a dense model, H100, general-purpose workload. Each group varies one parameter while holding all others at their baseline value. Missing bars indicate the simulator does not expose that configuration knob.

---

### Figure 4b: Config Sensitivity — MoE Model

![Figure 4b](figures/fig4b_config_moe.png)

Same controlled single-parameter sweeps as Figure 4a, applied to a MoE model. Adds expert parallelism (DP) as an additional swept dimension. Missing bars indicate lack of simulator support for that configuration.

---

### Figure 5: Accuracy vs. Speed Pareto Frontier

![Figure 5](figures/fig5_pareto.png)

Median MAPE vs. median wall-clock runtime per simulator, aggregated across all experiments. Error bars show interquartile range. The shaded region marks the Pareto-dominated quadrant — simulators falling there are strictly worse on both accuracy and speed.

---

### Table 1: Simulator Runtime Summary

| Simulator | Median Runtime (s) | Speedup vs. Real |
|---|---|---|
| Vidur | 127.1 | 9x |
| LLM-Optimizer | 0.1 | 23,754x |
| AIConfigurator | 3.3 | 360x |

Median runtime per simulator and speedup over real experiment execution (~1200s per experiment).
