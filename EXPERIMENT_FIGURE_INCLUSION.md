# Experiment Inclusion by Figure

## Summary Table

| Figure | Description | Count | Experiment IDs |
|--------|-------------|-------|----------------|
| **Fig 0a** | Aggregate Analytical (BLIS, LLM-Opt, AIConf) | 5 | 13, 16, 57, 58, 59 |
| **Fig 0b** | Aggregate Trace (BLIS, Vidur) | 4 | 40, 41, 57, 58 |
| **Fig 0c** | Aggregate LLMServingSim | 0 | *No data* |
| **Fig 1** | Model Sensitivity (7 models, H100) | 7 | 13, 16, 49, 56, 57, 58, 59 |
| **Fig 2** | Hardware Portability (H100/A100/L40S) | 14 | 13, 16, 36, 38, 40, 41, 42, 49, 54, 55, 56, 57, 58, 59 |
| **Fig 3** | Workload Sensitivity (4 workloads) | 12 | 13, 14, 15, 16, 18, 19, 46, 49, 50, 51, 53, 59 |
| **Fig 4a** | Config Sensitivity Dense (Llama-3.1-8B) | 6 | 16, 22, 23, 24, 25, 26 |
| **Fig 4b** | Config Sensitivity MoE (Mixtral-8x7B) | 7 | 9, 27, 28, 29, 30, 31, 56 |
| **Fig 5** | Pareto (Accuracy vs Runtime) | 40 | 2, 3, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36, 38, 40, 41, 42, 46, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59 |

## Key Findings

### Coverage
- **Total experiments in results**: 40
- **Experiments appearing in at least one figure**: 40 (100%)
- **SAFE experiments (DP=1 or nil) excluded from all figures**: 0

### Figure-Specific Notes

**Fig 0c (LLMServingSim)**: No experiments meet requirements
- Only experiment 31 has LLMServingSim data
- "Common set" filter requires all 3 simulators (BLIS, LLM-Optimizer, LLMServingSim)
- Single experiment insufficient for comparison figure

**Fig 2 (Hardware Portability)**: Includes L40S experiments
- Experiments 54, 55 (L40S hardware) **DO appear** with BLIS-Roofline bars
- L40S shows only 1 simulator vs 2-3 for other hardware (visual asymmetry)

**Fig 5 (Pareto)**: Most comprehensive
- Only figure including all 40 experiments
- Shows all simulators (no exclusions)
- Aggregates across all workloads/configs

### L40S Experiments (54, 55)
Appear in exactly 2 figures:
- **Fig 2**: Hardware Portability (with blis-roofline only)
- **Fig 5**: Pareto

Excluded from Figs 0a, 0b, 1, 3, 4a, 4b due to:
- H100-only filters (Figs 1, 3, 4a, 4b)
- "Common set" requirements (Figs 0a, 0b need multiple simulators)

## Conclusion

Every SAFE experiment with DP=1 or nil that successfully ran appears in at least one figure. No experiments are orphaned from the visualization pipeline.

## Missing FP8 Experiments (Llama-4-Scout)

**7 Scout experiments exist on disk but have ZERO simulator results:**

| ID | Hardware | DP | Workload | Why Missing |
|---|---|---|---|---|
| 17 | H100 | 1 | general | FP8 - no simulator support |
| 20 | H100 | 1 | codegen | FP8 - no simulator support |
| 21 | H100 | 1 | roleplay | FP8 - no simulator support |
| 33 | H100 | 2 | general | FP8 + DP > 1 |
| 34 | H100 | 4 | general | FP8 + DP > 1 |
| 44 | A100-80GB | 1 | general | FP8 on A100 (no native FP8 ops) |
| 48 | H100 | 1 | reasoning | FP8 - no simulator support |

**Why NO simulator ran these:**
- **llm-optimizer**: Config parse error - missing `hidden_size` key in model's config.json
- **aiconfigurator**: Scout in MoE exclusion list
- **vidur**: No FP8 device profiles
- **blis-roofline**: Not attempted (reason unknown - no FP8 filter in code)

**Impact if BLIS ran them:** Minimal - would be aggregated into Fig 2 bars (invisible) and appear in Fig 5 only.

**Validation study scope:** Effectively FP16 models only. FP8 not supported.

**See FP8_EXPERIMENTS_MISSING.md for details.**
