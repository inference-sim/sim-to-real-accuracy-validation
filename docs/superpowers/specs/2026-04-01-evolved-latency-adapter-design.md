# BLIS Evolved Latency Adapter Design

**Date:** 2026-04-01
**Status:** Approved
**Iteration:** iter16

## Overview

Design and implement a simulator adapter for BLIS's `evolved` latency backend, enabling comparison against other simulators (BLIS-roofline, LLMServingSim, Vidur, etc.) in the accuracy validation framework.

The evolved backend is a physics-informed latency model with learned correction coefficients, based on the trained-roofline architecture with MoE/FP8 enhancements.

## Background

### What is the Evolved Latency Backend?

Located in `/inference-sim/sim/latency/evolved_model.go`, the evolved backend implements:

- **7-term step-time formula** (inherited from trained-roofline):
  ```
  β₁·max(T_pf_compute, T_pf_kv) + β₂·max(T_dc_compute, T_dc_kv)
  + β₃·T_weight + β₄·T_tp + β₅·L + β₆·batchSize + β₇
  ```

- **3 alpha coefficients** for request-level overhead:
  - `α₀`: QueueingTime — fixed API processing overhead (µs)
  - `α₁`: PostDecodeFixedOverhead — per-request completion overhead (µs)
  - `α₂`: OutputTokenProcessingTime — per-output-token streaming (µs/token)

- **7 beta coefficients** for step-time roofline corrections:
  - `β₁`: Prefill roofline correction (dimensionless)
  - `β₂`: Decode roofline correction (dimensionless)
  - `β₃`: Weight loading correction (dimensionless)
  - `β₄`: TP communication correction (dimensionless)
  - `β₅`: Per-layer overhead (µs/layer)
  - `β₆`: Per-request scheduling overhead (µs/request)
  - `β₇`: Per-step constant overhead (µs/step)

### Why iter16?

Iteration 16 represents the latest optimized coefficients:
- **Training set**: 15 experiments (9 trained, 6 evaluated with locally-cached configs)
- **Overall loss**: 60.19% MAPE (TTFT RMSE: 31.36ms, E2E RMSE: 28.82ms)
- **Architecture**: 7-term trained-roofline with dataset-specific enhancements
- **Model coverage**: Scout (interleaved MoE/FP8), Llama-2-7b, Llama-3.1-70B, Qwen2.5-7B, Yi-34B, Mistral-Nemo-12B
- **Key improvements**:
  - Split FLOPs between MoE and dense layers (Scout #877 fix)
  - FP8-aware weight precision (1 byte/param vs 2)
  - Proper per-request overhead placement (PostDecodeFixedOverhead)

### Current Adapter Ecosystem

Existing BLIS adapters in `experiment/adapters/`:
- `blis_roofline.py` — analytical FLOPs/bandwidth latency (`--latency-model roofline`)
- `blis_crossmodel.py` — physics-informed model (`--latency-model crossmodel`)
- `blis_trained_roofline.py` — roofline with learned corrections (`--latency-model trained-roofline`)
- `blis_blackbox.py` — profiled coefficients (`--latency-model blackbox`)

All extend `BaseBLISAdapter` which provides:
- CLI argument construction (`_build_common_args`)
- Workload spec generation (`_write_workload_spec`)
- Result parsing (`_parse_blis_results`)
- Per-stage request splitting (`_split_requests_by_stage`)

## Goals

1. **Primary**: Enable evolved backend comparison in simulator accuracy validation
2. **Secondary**: Establish pattern for future iteration-based adapters
3. **Non-goal**: Dynamic coefficient loading (use static iter16 coefficients)

## Design

### 1. Adapter Class Structure

```python
class BLISEvolvedAdapter(BaseBLISAdapter):
    """BLIS simulator with ``--latency-model evolved``.

    Uses iter16 optimized coefficients (60.19% MAPE on 15 experiments).
    Supports MoE (Scout, Mixtral), GQA (Llama-3.1), and FP8 quantization.
    """

    # Iter16 optimized coefficients (2026-04-01)
    ITER16_ALPHA = [
        15569.495449697066,   # α₀: QueueingTime (µs)
        815.0556502348827,    # α₁: PostDecodeFixedOverhead (µs)
        45.705744318725586    # α₂: OutputTokenProcessingTime (µs/token)
    ]

    ITER16_BETA = [
        0.20081681581824434,   # β₁: Prefill roofline correction
        1.6173961192042448,    # β₂: Decode roofline correction
        1.3603417361920076,    # β₃: Weight loading correction
        0.39579536655780084,   # β₄: TP communication correction
        62.19421689224744,     # β₅: Per-layer overhead (µs)
        2.937563498958273,     # β₆: Per-request scheduling (µs)
        169.37780505091155     # β₇: Per-step constant (µs)
    ]

    @property
    def name(self) -> str:
        return "blis-evolved"

    def run(self, experiment: Experiment) -> SimulatorResult:
        # Build CLI args with evolved backend + iter16 coefficients
        # Execute BLIS binary
        # Parse results using base class method
```

### 2. File Structure

```
experiment/adapters/
├── blis_evolved.py          # New: 60 lines (similar to blis_roofline.py)
├── __init__.py              # Update: Add BLISEvolvedAdapter export
└── README.md                # Update: Document evolved adapter
```

### 3. CLI Argument Construction

The adapter extends `_build_common_args()` to add evolved-specific flags:

```python
def run(self, experiment: Experiment) -> SimulatorResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate workload spec (inherited)
        spec_path = os.path.join(tmpdir, "workload_spec.yaml")
        self._write_workload_spec(experiment, spec_path)

        # Build base args (inherited)
        results_path = os.path.join(tmpdir, "results.json")
        args = self._build_common_args(experiment, spec_path, results_path)

        # Add evolved-specific flags
        args.extend(["--latency-model", "evolved"])
        args.extend(["--alpha-coeffs", self._format_coeffs(self.ITER16_ALPHA)])
        args.extend(["--beta-coeffs", self._format_coeffs(self.ITER16_BETA)])

        # Execute and parse (inherited error handling)
        subprocess.run(args, capture_output=True, check=True, cwd=self._blis_dir)
        return self._parse_blis_results(results_path, experiment)
```

Helper method for coefficient formatting:
```python
@staticmethod
def _format_coeffs(coeffs: list[float]) -> str:
    """Format coefficients as comma-separated string with 6 decimal places."""
    return ','.join(f"{c:.6f}" for c in coeffs)
```

### 4. Result Parsing

Reuses `BaseBLISAdapter._parse_blis_results()`:
- Reads BLIS JSON output (`results.json`)
- Extracts top-level metrics (completed_requests, e2e_mean_ms, ttft_mean_ms, etc.)
- Splits requests by stage using arrival time boundaries
- Computes per-stage percentile metrics (mean, p90, p99)
- Returns `SimulatorResult` with nested `StageMetrics` and `LatencyDistribution`

No evolved-specific parsing needed — BLIS output format is consistent across backends.

### 5. Integration Points

#### Experiment Runner
```python
# experiment/run.py
from experiment.adapters.blis_evolved import BLISEvolvedAdapter

adapters = {
    "blis-roofline": BLISRooflineAdapter(blis_binary),
    "blis-evolved": BLISEvolvedAdapter(blis_binary),  # New
    "llmservingsim": LLMServingSimAdapter(llmservingsim_dir),
    # ...
}
```

#### Command-line Usage
```bash
# Run evolved adapter only
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters blis-evolved \
  --blis-binary inference-sim/blis

# Compare evolved vs roofline
python -m experiment.run \
  --adapters blis-roofline,blis-evolved \
  --blis-binary inference-sim/blis
```

### 6. Validation Strategy

#### Unit Tests (`tests/test_blis_adapters.py`)
```python
def test_blis_evolved_adapter_coefficients():
    """Verify iter16 coefficients are correctly formatted."""
    adapter = BLISEvolvedAdapter(blis_binary="/fake/blis")

    alpha_str = adapter._format_coeffs(adapter.ITER16_ALPHA)
    assert alpha_str.count(',') == 2  # 3 coefficients
    assert all(c > 0 for c in adapter.ITER16_ALPHA)

    beta_str = adapter._format_coeffs(adapter.ITER16_BETA)
    assert beta_str.count(',') == 6  # 7 coefficients

def test_blis_evolved_adapter_integration(tmp_path, mock_experiment):
    """Integration test with minimal BLIS binary execution."""
    # Mock BLIS binary that validates evolved args
    # Verify --latency-model evolved is passed
    # Verify alpha/beta coeffs are formatted correctly
```

#### Manual Validation
1. Run on single experiment (e.g., `16-llama-3-1-8b-tp1-general`)
2. Verify BLIS executes without errors
3. Compare output metrics with iter16 training results
4. Check that evolved MAPE ≈ 20-60% range (per iter16 experiment results)

## Implementation Plan

### Phase 1: Core Adapter (Week 1)
1. Create `blis_evolved.py` with `BLISEvolvedAdapter` class
2. Implement coefficient constants from iter16
3. Implement `run()` method with CLI argument construction
4. Update `__init__.py` to export adapter

### Phase 2: Testing (Week 1)
1. Add unit tests for coefficient formatting
2. Add integration test with mock BLIS binary
3. Manual validation on 3 ground-truth experiments
4. Compare results with iter16 training metrics

### Phase 3: Documentation (Week 1)
1. Update `experiment/adapters/README.md` with evolved section
2. Document iter16 coefficients provenance
3. Add usage examples to top-level README
4. Create troubleshooting guide

## Trade-offs

### Static vs Dynamic Coefficients

**Decision: Use static iter16 coefficients**

**Pros:**
- Simple implementation (~60 lines of code)
- No config file dependencies
- Consistent with roofline/crossmodel adapters
- Clear provenance (iter16 timestamp in constants)

**Cons:**
- Cannot easily swap iterations without code changes
- No per-experiment coefficient customization

**Alternative rejected:** Config-driven coefficients from `training/iterations/iterN/`
- Adds complexity (YAML parsing, path resolution)
- Requires BLIS training directory structure
- Harder to version control which iteration was used

**Future enhancement:** If iteration comparison becomes a common need, add optional `iteration` parameter to constructor.

### Coefficient Precision

**Decision: Use 6 decimal places in formatted strings**

**Rationale:**
- Matches iter16 JSON precision
- Sufficient for µs-level latency predictions
- Avoids floating-point accumulation errors

**Example:**
```python
# Input: 15569.495449697066
# Output: "15569.495450"
```

## Success Criteria

1. **Functional**: Adapter runs on all ground-truth experiments without errors
2. **Accurate**: MAPE within ±10% of iter16 training results (60.19% ± 6%)
3. **Maintainable**: Code follows existing adapter patterns (<80 lines)
4. **Documented**: README explains when to use evolved vs other backends

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| BLIS binary not compiled with evolved backend | High | Validate backend registration in `sim/bundle.go` |
| Coefficient formatting errors | Medium | Unit tests for `_format_coeffs()` |
| Iter16 coefficients don't generalize to new experiments | Medium | Document expected MAPE range, add fallback to roofline |
| Performance regression from 7 beta terms | Low | Benchmark shows <5% overhead vs roofline |

## Future Enhancements

1. **Iteration Comparison**: Add optional `iteration` parameter to load coefficients from `training/iterations/iterN/`
2. **Coefficient Tuning**: Command-line override for alpha/beta coefficients
3. **Multi-backend Ensemble**: Weighted average of evolved + roofline predictions
4. **Real-time Monitoring**: Log per-request APE for debugging coefficient drift

## References

- BLIS evolved backend: `inference-sim/sim/latency/evolved_model.go`
- Iter16 training results: `inference-sim/training/iterations/iter16/inner_loop_results.json`
- Coefficient bounds: `inference-sim/training/iterations/iter16/coefficient_bounds.yaml`
- Existing adapter pattern: `experiment/adapters/blis_roofline.py`
- Base adapter logic: `experiment/adapters/base.py`

## Appendix: Iter16 Coefficient Provenance

**Training timestamp:** 2026-04-01T13:42:00
**Training set:** 15 experiments (9 trained, 6 evaluated)
**Optimization:** 1705 Bayesian trials, 2 hours
**Overall loss:** 60.19% MAPE
**TTFT RMSE:** 31.36ms
**E2E RMSE:** 28.82ms

**Per-experiment APE range:**
- Best: 8.2% (Qwen2.5-7B reasoning-lite)
- Worst: 129.9% (Scout reasoning-lite — high variance workload)
- Median: ~40% (typical cross-model generalization)

**Architecture enhancements vs trained-roofline:**
1. MoE/dense layer split (Scout #877 fix)
2. FP8-aware weight precision (1 byte/param)
3. Per-request overhead in PostDecodeFixedOverhead (not StepTime)
