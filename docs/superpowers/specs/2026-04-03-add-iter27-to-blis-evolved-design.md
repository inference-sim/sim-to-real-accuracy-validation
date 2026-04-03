# Design: Add Iter27 Coefficients to BLIS Evolved Adapter

**Date**: 2026-04-03
**Status**: Approved

## Summary

Add iter27 coefficients to the BLIS evolved adapter while maintaining iter26 as the default for backward compatibility. Iter27 achieves 34.61% loss (vs 37.42% for iter26) through CMA-ES joint optimization of 6 parameters.

## Background

The BLIS evolved adapter currently supports three coefficient sets:
- **Iter16** (7 betas): 60.19% loss - baseline trained-roofline
- **Iter24** (10 betas): 39.18% loss - added prefill/decode split + MoE term
- **Iter26** (10 betas): 37.42% loss - activated TP All-Reduce term (β₄=0.410)

Iter27 represents the latest training iteration with improved accuracy through joint optimization.

## Iter27 Training Summary

- **Dataset**: 15 experiments (H100 / FP16)
- **Best Loss**: 34.61% overall (TTFT=22.81%, E2E=11.79%)
- **Method**: CMA-ES joint optimization of 6 parameters (β₁ₐ, β₄, β₅, β₇, β₈, β₂ᵦ)
- **Trials**: 141 total, best at trial 62
- **Key Finding**: Strong coefficient interactions discovered - β₄ increased 83% to 0.752, allowing β₅ and β₇ to decrease
- **Improvement**: 37.42% → 34.61% (-2.81 points, -7.5% relative)

## Key Coefficient Changes vs Iter26

| Coefficient | Iter26 | Iter27 | Change | Interpretation |
|-------------|--------|--------|--------|----------------|
| β₁ₐ | 0.139 | 0.152 | +9% | Prefill compute - slight increase |
| β₄ | 0.410 | 0.752 | +83% | TP All-Reduce - captures comm overhead better |
| β₅ | 49.6 | 32.4 | -35% | Per-layer - β₄ absorbed overhead |
| β₇ | 169.4 | 126.0 | -26% | Per-step constant - reduced |
| β₈ | 427.3 | 505.5 | +18% | MoE overhead - increased |
| β₂ᵦ | 1.263 | 1.922 | +52% | Decode memory - stronger correction |

## Design

### 1. Add Coefficient Constants

Add `ITER27_ALPHA` and `ITER27_BETA` class constants to `BLISEvolvedAdapter`:

**Alpha (3 values)**:
- α₀: 15563.199579 - QueueingTime (~15.6ms fixed API overhead)
- α₁: 777.3455 - PostDecodeFixedOverhead (~0.8ms per-request)
- α₂: 45.907545 - OutputTokenProcessingTime (µs/token streaming)

**Beta (10 values)**:
- β₁ₐ: 0.152128 - Prefill compute correction
- β₂ₐ: 0.000721 - Decode compute correction (near-zero, memory-bound)
- β₃: 1.363621 - Weight loading correction (36% overhead)
- β₄: 0.752037 - TP All-Reduce correction
- β₅: 32.394131 - Per-layer overhead (µs/layer)
- β₆: 2.805128 - Per-request scheduling (µs/req)
- β₇: 126.024825 - Per-step constant (µs/step)
- β₈: 505.508488 - Per-MoE-layer overhead (µs/MoE-layer)
- β₁ᵦ: 0.000746 - Prefill memory correction (near-zero, compute-bound)
- β₂ᵦ: 1.922366 - Decode memory correction

### 2. Update Constructor

- Accept `iteration` parameter with values `(16, 24, 26, 27)`
- Keep `iteration=26` as default for backward compatibility
- Update validation to raise `ValueError` if not in valid set

### 3. Update `run` Method

Add handling for `iteration == 27` in the coefficient selection logic.

### 4. Update Docstring

Add iter27 training summary section documenting:
- Loss improvement (37.42% → 34.61%)
- CMA-ES joint optimization method
- Key coefficient changes and physical interpretations
- Notable per-experiment improvements (e.g., Mistral TP=2: 27.4% → 20.0% TTFT)

## Implementation Notes

1. **Coefficient Precision**: Use exact values from `inference-sim/training/iterations/iter27/inner_loop_results.json` with full precision
2. **Backward Compatibility**: Default remains `iteration=26` so existing code continues to work
3. **Testing**: Update adapter tests to verify iter27 can be instantiated and runs successfully
4. **Documentation**: Update `experiment/adapters/README.md` to mention iter27 availability

## Success Criteria

- [ ] ITER27_ALPHA and ITER27_BETA constants added with exact coefficients
- [ ] Constructor accepts `iteration=27` parameter
- [ ] `run()` method correctly selects iter27 coefficients
- [ ] Docstring documents iter27 training summary
- [ ] Tests pass for iter27 instantiation
- [ ] Default remains `iteration=26` for backward compatibility
