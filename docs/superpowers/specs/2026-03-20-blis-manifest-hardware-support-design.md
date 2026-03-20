# BLIS Adapter Manifest Hardware Support

**Date:** 2026-03-20
**Status:** Approved
**Author:** Claude Sonnet 4.5

## Problem

The BLIS adapters (`blis_roofline.py`, `blis_crossmodel.py`, `blis_blackbox.py`, `blis_trained_roofline.py`) hardcode `--hardware "H100"` in their CLI arguments, preventing them from simulating the full range of experiments in the vllm_data ground truth dataset.

The manifest (`vllm_data/ground_truth/experiments.json`) contains 56 experiments spanning:
- **Hardware:** H100, A100-80GB
- **Precision:** FP16, FP8
- **GPU memory utilization:** 0.9, 0.95
- **Data parallelism:** null, 1, 2, 4

Currently, only H100 experiments can be accurately simulated.

## Analysis

### Current State

`experiment/adapters/base.py:70` hardcodes hardware:
```python
def _build_common_args(...) -> list[str]:
    return [
        self.blis_binary, "run",
        "--model", experiment.model,
        "--tp", str(experiment.tp),
        "--hardware", "H100",  # ❌ hardcoded
        "--max-num-running-reqs", str(experiment.max_num_seqs),
        # ...
    ]
```

### Available Data

The `Experiment` dataclass (in `experiment/data_model.py`) already includes:
```python
@dataclass
class Experiment:
    # ... existing fields
    hardware: str = "H100"
    precision: str = "FP16"
    gpu_mem_util: float = 0.9
    dp: int | None = None
    # ...
```

These fields are populated by `parse_experiment()` from the manifest.

### BLIS CLI Capabilities

From BLIS PR #790 (`blis run --help`):
```
--hardware string                    GPU type
--gpu-memory-utilization float       Fraction of GPU memory to use for KV cache (default 0.9)
--total-kv-blocks int                Total number of KV cache blocks (default 1000000)
```

**Precedence for KV capacity** (from `inference-sim/cmd/root.go:374-702`):
1. `--total-kv-blocks` (CLI flag) — takes absolute precedence
2. `defaults.yaml` match
3. Auto-calculate using `--gpu-memory-utilization`

**What we pass:**
- ✅ `--total-kv-blocks` — extracted from `vllm.log` (ground truth)
- ✅ `--hardware` — should come from manifest, not hardcoded

**What BLIS doesn't support:**
- ❌ `--precision` (no such flag)
- ❌ `--data-parallelism` (no such flag)

### Why Not Pass `--gpu-memory-utilization`?

The `--gpu-memory-utilization` flag is **only used by BLIS for auto-calculating KV blocks** when `--total-kv-blocks` is not provided. Since we extract `total_kv_blocks` from vllm.log as ground truth, BLIS skips the auto-calculation entirely.

Passing `--gpu-memory-utilization` would:
- ✅ Document the experiment config in trace exports
- ❌ Have zero effect on simulation behavior (skipped due to explicit `--total-kv-blocks`)

Decision: **Don't pass it** — keeps the change minimal and focused.

## Design

### Change Summary

**File:** `experiment/adapters/base.py`
**Line:** 70
**Change:** Replace hardcoded `"H100"` with `experiment.hardware`

```python
# Before
"--hardware", "H100",

# After
"--hardware", experiment.hardware,
```

### Files Modified

1. **`experiment/adapters/base.py`**
   - Line 70: Use `experiment.hardware` instead of hardcoded string

2. **`tests/test_blis_adapters.py`**
   - `TestBLISCLIArgs.test_kv_offloading_flags_present` (line ~246-268)
   - Update assertion to check for `experiment.hardware` instead of hardcoded "H100"
   - Optionally add test for A100-80GB to verify hardware is respected

### Behavior

**Before:**
- All BLIS simulations use H100 hardware calibration regardless of actual experiment hardware
- A100-80GB experiments are simulated with incorrect hardware parameters
- Leads to systematic error for non-H100 experiments

**After:**
- BLIS uses the correct hardware calibration (H100 or A100-80GB) from the manifest
- Simulations match the actual hardware used in ground truth experiments
- Covers full parameter space of the 56-experiment dataset

### What's Not Changing

**Fields that remain unused:**
- `experiment.precision` — BLIS has no precision flag (affects model loading, not a CLI parameter)
- `experiment.gpu_mem_util` — only used for KV auto-calculation, which we skip
- `experiment.dp` — BLIS doesn't support data parallelism

These fields are stored in the dataclass for completeness but not passed to BLIS.

### Backward Compatibility

The change is fully backward compatible:
- `Experiment.hardware` has a default value of `"H100"`
- Existing code that constructs `Experiment` without a manifest will continue to work
- Tests that don't use manifest entries will get the H100 default

## Implementation Plan

1. **Modify `BaseBLISAdapter._build_common_args()`**
   - Replace hardcoded `"H100"` with `experiment.hardware`

2. **Update test in `test_blis_adapters.py`**
   - `test_kv_offloading_flags_present`: Change assertion from checking for "H100" to checking for `experiment.hardware`
   - `test_model_and_tp_in_args`: Already tests dynamic values, no change needed

3. **Verify with integration test**
   - Run BLIS adapter against an A100-80GB experiment from manifest
   - Verify `--hardware A100-80GB` appears in BLIS subprocess args

## Testing Strategy

### Unit Tests (already exist)

1. `TestBLISCLIArgs.test_kv_offloading_flags_present`
   - Update to verify hardware is passed from `experiment.hardware`
   - No longer check for hardcoded "H100"

2. `TestBLISCLIArgs.test_model_and_tp_in_args`
   - Already tests that CLI args use experiment fields dynamically
   - Demonstrates the pattern we're following for hardware

### Integration Test (manual verification)

Run adapter against A100-80GB experiment:
```bash
python -m experiment.run \
  --gt-dir vllm_data/ground_truth \
  --adapter blis-roofline \
  --blis-binary inference-sim/blis \
  --exp-id 36  # Qwen3-14B on A100-80GB
```

Verify BLIS command includes `--hardware A100-80GB`.

## Risks

### Low Risk

**Hardware calibration mismatch:**
- Risk: BLIS might not have calibration data for A100-80GB
- Mitigation: BLIS has `hardware_config.json` with both H100 and A100-80GB
- Verified: Both GPU types are supported in PR #790

**Test brittleness:**
- Risk: Test might need manual value updates
- Mitigation: Test will check for `experiment.hardware` value, not a hardcoded string

## Alternatives Considered

### Alternative 1: Pass `--gpu-memory-utilization` too

**Why not:** It's only used for auto-calculating KV blocks, which we skip since we provide explicit `--total-kv-blocks` from vllm.log ground truth. Adds complexity with zero benefit to simulation accuracy.

### Alternative 2: Pass all manifest fields to BLIS

**Why not:** BLIS doesn't have flags for precision or data-parallelism. Would require BLIS changes first.

### Alternative 3: Validate hardware against supported list

**Why not:** Adds unnecessary complexity. BLIS will error clearly if hardware is unsupported. Let BLIS own its validation logic.

## Open Questions

None. Design is ready for implementation.
