# Publication Figures

---

### Figure 1: Prediction Error Across Model Architectures

![Figure 1](figures/fig1_model_sensitivity.png)

MAPE of each simulator across 7 models (7B–8x22B, dense and MoE) on H100 with default serving config and general-purpose workload. Tests whether accuracy generalizes across architectures.

---

### Figure 2: Prediction Error Across GPU Types

![Figure 2](figures/fig2_hardware_portability.png)

Median MAPE across models for three GPU types (H100, A100, L40S) with default config. Tests hardware portability of simulator predictions.

---

### Figure 3: Prediction Error Across Workload Types

![Figure 3](figures/fig3_workload_sensitivity.png)

Median MAPE across 4 representative models for four workload types (general, code generation, roleplay, reasoning) on H100 with default config. Tests robustness to diverse traffic patterns.

---

### Figure 4a: Config Sensitivity — Dense Model

![Figure 4a](figures/fig4a_config_dense.png)

MAPE as individual serving parameters (TP, chunk size, GPU memory utilization, KV cache offloading) are swept one at a time on a dense model (H100, general-purpose), with all other parameters held at baseline. Tests accuracy under production tuning.

---

### Figure 4b: Config Sensitivity — MoE Model

![Figure 4b](figures/fig4b_config_moe.png)

Same controlled single-parameter sweeps as Figure 4a, applied to a MoE model. Adds data/expert parallelism as an additional swept parameter. Tests whether simulators capture MoE-specific parallelism interactions.

---

### Figure 5: Accuracy vs. Speed Pareto Frontier

![Figure 5](figures/fig5_pareto.png)

Median MAPE vs. median wall-clock runtime across all experiments and simulators. Error bars show IQR. The shaded region marks the Pareto-dominated quadrant. Reveals the fundamental accuracy-speed tradeoff.

---

### Table 1: Simulator Runtime Summary

| Simulator | Median Runtime (s) | Speedup vs. Real |
|---|---|---|
| Vidur | 127.1 | 9x |
| LLM-Optimizer | 0.1 | 23,754x |
| AIConfigurator | 3.3 | 360x |

Median runtime per simulator and speedup factor vs. real experiment execution (~1200s).
