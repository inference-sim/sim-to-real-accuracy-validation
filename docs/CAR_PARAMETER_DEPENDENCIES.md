# CAR Parameter Validation Hypotheses

**Purpose**: Validate that each of the 4 CAR parameters captures a real, independent error source in pure roofline predictions.

**Related**: `CONTEXT_AWARE_ROOFLINE.md`

**Key constraints**:
- All parameters are GLOBAL (single value fitted across all experiments)
- Available data: **Request-level traces only** (e2e, TTFT, ITL, prompt_tokens, completion_tokens)
- **NO step-level data** available (no batch composition, no per-step mixing factors)

---

## Data Availability

From `vllm_data/other_gt/*/traces.json`:
- **Request-level OpenTelemetry spans** (78+ requests per experiment, 11 experiments = ~858 requests total)
- Available metrics per request:
  - `gen_ai.latency.e2e`: End-to-end latency
  - `gen_ai.latency.time_to_first_token`: TTFT
  - `gen_ai.latency.time_in_model_prefill`: Time spent in prefill
  - `gen_ai.latency.time_in_model_decode`: Time spent in decode
  - `gen_ai.usage.prompt_tokens`: Input tokens
  - `gen_ai.usage.completion_tokens`: Output tokens

**NOT available**:
- Step-level batch composition
- Per-step mixing factors
- Actual KV cache utilization per step

**Implication**: Validation must work at request-level aggregates, not step-level predictions.

---

## The 4 Parameters

CAR extends pure roofline with 4 trainable parameters:

1. **mfu_prefill** ∈ [0.3, 0.8]: Scales prefill compute time by `1/mfu_prefill`
2. **mfu_decode** ∈ [0.1, 0.5]: Scales decode compute time by `1/mfu_decode`
3. **ψ** ∈ [0, 100] µs: Mixed batch overhead per step, scales with `mixing_factor`
4. **λ** ∈ [10, 200] µs: Constant per-step system overhead

**Challenge**: Parameters ψ and λ are per-step, but we only have per-request data. We must:
1. Simulate each request using CAR
2. Aggregate simulated step-level corrections to request-level predictions
3. Compare predicted vs actual request metrics (e2e, TTFT, ITL)

---

## H18: MFU Prefill Correction at Request Level

**Hypothesis**: Scaling prefill compute time by `1/mfu_prefill` reduces error in TTFT predictions uniformly across all requests.

**Claim**:
- Theoretical MFU (0.45) doesn't match actual prefill efficiency
- Fitting mfu_prefill reduces TTFT prediction error
- TTFT error = actual_ttft - (prefill_roofline_ideal / mfu_prefill + λ)

**Refuted if**:
- Fitted mfu_prefill is outside [0.3, 0.8], OR
- TTFT error variance does NOT decrease by ≥30% after correction, OR
- Residual TTFT error correlates with prompt_tokens (systematic bias remains)

### Experiment Design

**Method**:
1. For each request in traces:
   - Extract: actual_ttft, prompt_tokens
   - Compute roofline_ideal_prefill (using prompt_tokens, model config)
2. Fit mfu_prefill to minimize TTFT MAPE:
   - predicted_ttft = roofline_ideal_prefill / mfu_prefill + λ_guess
   - Where λ_guess ≈ 80 µs (temporary constant for fitting)
3. Analyze:
   - Is 0.3 < mfu_prefill < 0.8?
   - Does Std(TTFT_error_after) < 0.7 × Std(TTFT_error_before)?
   - Is residual error uncorrelated with prompt_tokens?

**Success criteria**:
- mfu_prefill ∈ [0.3, 0.8]
- TTFT error variance reduces by ≥30%
- Residual R² with prompt_tokens < 0.1

### Expected Outcome

If confirmed:
- mfu_prefill ≈ 0.35-0.55
- TTFT predictions improve uniformly
- Justifies trainable prefill MFU

---

## H19: MFU Decode Correction at Request Level

**Hypothesis**: Scaling decode compute time by `1/mfu_decode` reduces error in ITL (inter-token latency) predictions.

**Claim**:
- Decode operations achieve lower MFU than prefill
- ITL = time_in_decode / completion_tokens ≈ (roofline_ideal_decode / mfu_decode + λ) / completion_tokens
- mfu_decode < mfu_prefill (asymmetric efficiency)

**Refuted if**:
- Fitted mfu_decode outside [0.1, 0.5], OR
- mfu_decode ≥ mfu_prefill, OR
- ITL error variance does NOT decrease by ≥30%

### Experiment Design

**Method**:
1. For each request:
   - Compute ITL = time_in_model_decode / completion_tokens
   - Compute roofline_ideal_decode_per_token
2. Fit mfu_decode to minimize ITL MAPE:
   - predicted_itl = (roofline_ideal_per_token / mfu_decode) + (λ_guess / completion_tokens)
3. Validate:
   - mfu_decode < mfu_prefill
   - ITL error variance reduction ≥30%

**Success criteria**:
- 0.1 < mfu_decode < 0.5
- mfu_decode < mfu_prefill
- ITL variance reduces by ≥30%

### Expected Outcome

If confirmed:
- mfu_decode ≈ 0.15-0.35
- Validates asymmetric MFU
- Justifies separate decode parameter

---

## H20: System Overhead Constant (λ)

**Hypothesis**: After MFU corrections, there's a constant per-step overhead λ that shows up in both TTFT and ITL predictions.

**Claim**:
- λ contributes once to TTFT: `TTFT = prefill/mfu_prefill + λ`
- λ contributes `λ/N` to ITL: `ITL = (decode_per_token/mfu_decode + λ/N)` where N = completion_tokens
- Fitting λ simultaneously on TTFT and ITL residuals yields consistent value

**Refuted if**:
- Fitted λ outside [10, 200] µs, OR
- λ from TTFT differs from λ from ITL by >50%, OR
- High variance (CV > 0.5)

### Experiment Design

**Method**:
1. Fix mfu_prefill, mfu_decode from H18-H19
2. Fit λ jointly on TTFT and ITL:
   - TTFT_predicted = prefill_corrected + λ
   - ITL_predicted = decode_corrected_per_token + λ/completion_tokens
3. Minimize combined MAPE
4. Check consistency: Is λ constant across TTFT and ITL predictions?

**Success criteria**:
- 10 < λ < 200 µs
- λ_ttft ≈ λ_itl (within 50%)
- CV < 0.5

### Expected Outcome

If confirmed:
- λ ≈ 50-100 µs
- Consistent across TTFT and ITL
- Validates constant system overhead

---

## H21: Mixed Batch Overhead (ψ) - Indirect Test

**Hypothesis**: Mixed batch overhead ψ is observable through simulation variance when requests overlap temporally.

**Claim**:
- Pure roofline + MFU + λ still has residual error
- Error increases when requests overlap (batching effects)
- Fitting ψ via full simulation (with batch mixing) reduces error

**Refuted if**:
- Adding ψ to CAR reduces MAPE by <3%, OR
- Fitted ψ outside [0, 100] µs, OR
- No correlation between overlapping requests and error

### Experiment Design

**Method** (requires simulation):
1. Run full DES simulation with pure roofline + MFU + λ (no ψ)
2. Run full DES simulation with CAR (including ψ)
3. Compare predicted vs actual:
   - E2E latency distribution
   - TTFT distribution
   - ITL distribution
4. Fit ψ to minimize aggregate MAPE across all three metrics

**Challenge**: Requires implementing full CAR in simulator since we can't directly measure mixing_factor from traces.

**Success criteria**:
- Adding ψ reduces MAPE by ≥3%
- 0 < ψ < 100 µs
- Error reduction is higher for experiments with more request overlap

### Expected Outcome

If confirmed:
- ψ ≈ 30-70 µs
- Validates mixed batch overhead model

**If data insufficient**: Mark as "Cannot validate directly - requires simulation" and drop ψ from CAR (reduce to 3 parameters: mfu_prefill, mfu_decode, λ)

---

## H22: Parameter Independence at Request Level

**Hypothesis**: MFU parameters and λ capture independent effects visible in request-level metrics.

**Claim**:
- mfu_prefill primarily affects TTFT
- mfu_decode primarily affects ITL
- λ affects both proportionally
- No strong correlation between corrections

**Refuted if**:
- High correlation |r| > 0.5 between parameter effects, OR
- Removing one parameter causes >50% change in others

### Experiment Design

**Method**:
1. Ablation study on request predictions:
   - Baseline: Pure roofline (theoretical MFU)
   - + mfu_prefill only
   - + mfu_decode only
   - + λ only
   - + All together
2. Measure contribution:
   - TTFT MAPE reduction per parameter
   - ITL MAPE reduction per parameter
   - E2E MAPE reduction per parameter
3. Check independence: Do parameters change significantly in joint vs separate fits?

**Success criteria**:
- mfu_prefill contributes >70% to TTFT improvement
- mfu_decode contributes >70% to ITL improvement
- λ contributes uniformly to both
- Joint fit parameters ≈ separate fit parameters (within 30%)

### Expected Outcome

If confirmed:
- All parameters contribute independently
- Validates 4-parameter design (or 3 if ψ dropped)

---

## H23: CAR End-to-End Validation

**Hypothesis**: CAR achieves 35-45% MAPE reduction over pure roofline on held-out validation set.

**Claim**:
- Pure roofline: 20-25% MAPE on E2E, TTFT, ITL
- CAR: 11-15% MAPE on E2E, TTFT, ITL
- Improvement is statistically significant (p < 0.01)

**Refuted if**:
- CAR MAPE > 15%, OR
- Improvement < 30%, OR
- Not significant (p > 0.01), OR
- Overfitting (validation MAPE > training MAPE + 5%)

### Experiment Design

**Method**:
1. Split experiments: 8 training, 3 validation
2. Fit CAR parameters on training set
3. Run DES with pure roofline on validation set
4. Run DES with CAR on validation set
5. Compare:
   - E2E MAPE
   - TTFT MAPE
   - ITL MAPE
   - Paired t-test

**Success criteria**:
- CAR E2E MAPE ≤ 15%
- MAPE reduction ≥ 35%
- p < 0.01
- No severe overfitting

### Expected Outcome

If confirmed:
- CAR validated end-to-end
- Ready for publication

---

## Revised Testing Order

**Phase 1**: Direct parameter validation (H18, H19, H20)
- Test MFU parameters and λ using request-level data
- Can extract directly from traces without simulation
- Determines mfu_prefill, mfu_decode, λ values

**Phase 2**: Simulation-based validation (H21, H22, H23)
- Requires implementing CAR in simulator
- Tests ψ parameter (mixed batch overhead)
- End-to-end validation

**Decision point**: If ψ cannot be validated (H21), drop it and proceed with 3-parameter CAR.

---

## Expected Results Summary

| Hypothesis | Parameter(s) | Data Needed | Expected Outcome |
|------------|--------------|-------------|------------------|
| H18 | mfu_prefill | Request traces (TTFT) | ✅ mfu ≈ 0.35-0.55 |
| H19 | mfu_decode | Request traces (ITL) | ✅ mfu ≈ 0.15-0.35 |
| H20 | λ | Request traces (TTFT+ITL) | ✅ λ ≈ 50-100 µs |
| H21 | ψ | Full simulation | ⚠️ May be unvalidatable |
| H22 | Independence | Request traces | ✅ Orthogonal |
| H23 | Overall | Full simulation | ✅ 35-45% improvement |

**Most likely final design**:
- **3 parameters** (mfu_prefill, mfu_decode, λ) if ψ cannot be validated
- **4 parameters** (mfu_prefill, mfu_decode, ψ, λ) if simulation shows ψ improves accuracy
