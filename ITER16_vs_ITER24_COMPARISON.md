# BLIS-Evolved: Iter16 vs Iter24 Coefficient Comparison

This document compares the prediction accuracy of BLIS-Evolved using iter16 coefficients (7 betas, 60.19% MAPE) versus iter24 coefficients (10 betas, 39.18% MAPE).

**Key Differences:**
- **Iter16**: 7-beta architecture, no prefill/decode split, no MoE term (60.19% MAPE)
- **Iter24**: 10-beta architecture with prefill/decode compute-memory split + MoE term (39.18% MAPE)

---

## Figure 1: Model Sensitivity

Prediction error across 7 model architectures on H100 with default configuration.

<table>
<tr>
<th width="50%">Iter16 (7 betas)</th>
<th width="50%">Iter24 (10 betas)</th>
</tr>
<tr>
<td><img src="results_iter16/figures/fig1_model_sensitivity.png" alt="Iter16 Model Sensitivity"></td>
<td><img src="results_iter24/figures/fig1_model_sensitivity.png" alt="Iter24 Model Sensitivity"></td>
</tr>
</table>

---

## Figure 2: Hardware Portability

Prediction error across 3 GPU types (H100, A100-80GB, L40S), aggregated over models.

<table>
<tr>
<th width="50%">Iter16 (7 betas)</th>
<th width="50%">Iter24 (10 betas)</th>
</tr>
<tr>
<td><img src="results_iter16/figures/fig2_hardware_portability.png" alt="Iter16 Hardware Portability"></td>
<td><img src="results_iter24/figures/fig2_hardware_portability.png" alt="Iter24 Hardware Portability"></td>
</tr>
</table>

---

## Figure 3: Workload Sensitivity

Prediction error across 4 workload types, aggregated over models.

<table>
<tr>
<th width="50%">Iter16 (7 betas)</th>
<th width="50%">Iter24 (10 betas)</th>
</tr>
<tr>
<td><img src="results_iter16/figures/fig3_workload_sensitivity.png" alt="Iter16 Workload Sensitivity"></td>
<td><img src="results_iter24/figures/fig3_workload_sensitivity.png" alt="Iter24 Workload Sensitivity"></td>
</tr>
</table>

---

## Figure 4a: Config Sensitivity (Dense Model)

Configuration parameter sweep for dense models (Llama-3.1-8B).

<table>
<tr>
<th width="50%">Iter16 (7 betas)</th>
<th width="50%">Iter24 (10 betas)</th>
</tr>
<tr>
<td><img src="results_iter16/figures/fig4a_config_dense.png" alt="Iter16 Config Dense"></td>
<td><img src="results_iter24/figures/fig4a_config_dense.png" alt="Iter24 Config Dense"></td>
</tr>
</table>

---

## Figure 4b: Config Sensitivity (MoE Model)

Configuration parameter sweep for MoE models (Mixtral-8x7B).

<table>
<tr>
<th width="50%">Iter16 (7 betas)</th>
<th width="50%">Iter24 (10 betas)</th>
</tr>
<tr>
<td><img src="results_iter16/figures/fig4b_config_moe.png" alt="Iter16 Config MoE"></td>
<td><img src="results_iter24/figures/fig4b_config_moe.png" alt="Iter24 Config MoE"></td>
</tr>
</table>

---

## Figure 5: Accuracy-Speed Pareto

Scatter plot showing accuracy vs execution speed trade-offs.

<table>
<tr>
<th width="50%">Iter16 (7 betas)</th>
<th width="50%">Iter24 (10 betas)</th>
</tr>
<tr>
<td><img src="results_iter16/figures/fig5_pareto.png" alt="Iter16 Pareto"></td>
<td><img src="results_iter24/figures/fig5_pareto.png" alt="Iter24 Pareto"></td>
</tr>
</table>

---

## Simulator Comparisons

### BLIS vs Vidur

<table>
<tr>
<th width="50%">Iter16 (7 betas)</th>
<th width="50%">Iter24 (10 betas)</th>
</tr>
<tr>
<td><img src="results_iter16/figures/sim_comparisons/blis_vs_vidur.png" alt="Iter16 BLIS vs Vidur"></td>
<td><img src="results_iter24/figures/sim_comparisons/blis_vs_vidur.png" alt="Iter24 BLIS vs Vidur"></td>
</tr>
</table>

### BLIS vs LLM Optimizer

<table>
<tr>
<th width="50%">Iter16 (7 betas)</th>
<th width="50%">Iter24 (10 betas)</th>
</tr>
<tr>
<td><img src="results_iter16/figures/sim_comparisons/blis_vs_llm_optimizer.png" alt="Iter16 BLIS vs LLM Optimizer"></td>
<td><img src="results_iter24/figures/sim_comparisons/blis_vs_llm_optimizer.png" alt="Iter24 BLIS vs LLM Optimizer"></td>
</tr>
</table>

### BLIS vs AIConfigurator

<table>
<tr>
<th width="50%">Iter16 (7 betas)</th>
<th width="50%">Iter24 (10 betas)</th>
</tr>
<tr>
<td><img src="results_iter16/figures/sim_comparisons/blis_vs_aiconfigurator.png" alt="Iter16 BLIS vs AIConfigurator"></td>
<td><img src="results_iter24/figures/sim_comparisons/blis_vs_aiconfigurator.png" alt="Iter24 BLIS vs AIConfigurator"></td>
</tr>
</table>

### BLIS vs LLMServingSim

<table>
<tr>
<th width="50%">Iter16 (7 betas)</th>
<th width="50%">Iter24 (10 betas)</th>
</tr>
<tr>
<td><img src="results_iter16/figures/sim_comparisons/blis_vs_llmservingsim.png" alt="Iter16 BLIS vs LLMServingSim"></td>
<td><img src="results_iter24/figures/sim_comparisons/blis_vs_llmservingsim.png" alt="Iter24 BLIS vs LLMServingSim"></td>
</tr>
</table>

---

## Summary

**Iter16 Characteristics:**
- 7 beta coefficients (no MoE term, no prefill/decode split)
- Training MAPE: 60.19%
- Architecture: Single roofline correction per phase

**Iter24 Characteristics:**
- 10 beta coefficients (includes β₈ MoE term + prefill/decode compute-memory split)
- Training MAPE: 39.18%
- Architecture: Separate compute/memory corrections for prefill and decode
- Key insight: Prefill is compute-only (β₁ᵦ=0), Decode is memory-only (β₂ₐ=0)

**Expected Improvement:**
- 34.9% relative MAPE reduction (60.19% → 39.18%)
- Better handling of MoE models (β₈·nMoELayers term)
- More accurate prefill/decode predictions through compute-memory split
