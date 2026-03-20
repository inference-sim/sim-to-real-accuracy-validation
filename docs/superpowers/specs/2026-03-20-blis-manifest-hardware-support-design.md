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

### Additional Issues Discovered

#### Issue 2: Hardware Name Mismatch

**Manifest** (`experiments.json`) uses: `"A100-80GB"`, `"H100"`, `"L40S"`
**BLIS** (`hardware_config.json`) expects: `"A100-80"`, `"H100"`, `"L40S"`
**defaults.yaml** uses: `"A100-80"`, `"H100"`

The manifest uses `"A100-80GB"` but BLIS and its defaults.yaml use `"A100-80"`.

**Impact:** Passing `--hardware "A100-80GB"` to BLIS will fail because it won't match any hardware config.

**Solution:** Add a hardware normalization function that maps manifest names to BLIS names.

#### Issue 3: BLISBlackboxAdapter.can_run() Doesn't Check Hardware

`blis_blackbox.py` lines 32-48 check model and TP but NOT GPU:
```python
def can_run(self, experiment: Experiment) -> bool:
    model_lower = experiment.model.lower()
    for entry in (data or {}).get("models", []):
        if (
            entry.get("id", "").lower() == model_lower
            and entry.get("tensor_parallelism") == experiment.tp
            # ❌ Missing: and entry.get("GPU") == normalized_hardware
            and any(c != 0 for c in entry.get("alpha_coeffs", []))
        ):
            return True
```

**Problem:** defaults.yaml has separate entries for H100 and A100-80 (same model, same TP, different GPU). Without checking GPU, `can_run()` returns True even when coefficients exist only for the wrong hardware.

**Example:**
- defaults.yaml has coefficients for `Llama-3.1-8B / H100 / TP=1`
- Experiment is `Llama-3.1-8B / A100-80GB / TP=1`
- `can_run()` returns True (wrong!)
- BLIS either fails or uses wrong coefficients

**Solution:** Add GPU field check to `can_run()` using normalized hardware names.

## Design

### Change Summary

**Three coordinated changes:**

1. **Add hardware normalization helper** (`base.py`)
   - Maps manifest hardware names to BLIS-compatible names
   - `"A100-80GB"` → `"A100-80"`, others pass through

2. **Use normalized hardware in CLI args** (`base.py`)
   - Replace hardcoded `"H100"` with normalized `experiment.hardware`

3. **Fix blackbox adapter can_run()** (`blis_blackbox.py`)
   - Add GPU field check using normalized hardware

### Files Modified

1. **`experiment/adapters/base.py`**
   - Add `_normalize_hardware()` static method
   - Line 70: Use `_normalize_hardware(experiment.hardware)` instead of `"H100"`

2. **`experiment/adapters/blis_blackbox.py`**
   - Update `can_run()` to check `entry.get("GPU")` matches normalized hardware

3. **`tests/test_blis_adapters.py`**
   - Update `test_kv_offloading_flags_present` to check for dynamic hardware
   - Add test for A100-80GB normalization
   - Add test for blackbox `can_run()` hardware matching

### Behavior Changes

**Before:**
- All BLIS simulations hardcode `--hardware "H100"` regardless of experiment
- A100-80GB experiments fail or use wrong hardware calibration
- BLISBlackboxAdapter.can_run() returns True for A100 experiments even when only H100 coefficients exist
- Manifest hardware name `"A100-80GB"` would fail if passed to BLIS

**After:**
- BLIS uses correct hardware calibration from manifest (`"H100"`, `"A100-80"`, `"L40S"`)
- Hardware names are normalized (`"A100-80GB"` → `"A100-80"`)
- BLISBlackboxAdapter.can_run() correctly checks if trained coefficients exist for the experiment's hardware
- Covers full parameter space of 56 experiments (H100 and A100-80GB)

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

### Step 1: Add Hardware Normalization

`experiment/adapters/base.py` - Add before `_build_common_args()`:
```python
@staticmethod
def _normalize_hardware(hardware: str) -> str:
    """Normalize manifest hardware names to BLIS-compatible names.

    Manifest uses "A100-80GB" but BLIS expects "A100-80".
    """
    if hardware == "A100-80GB":
        return "A100-80"
    return hardware  # H100, L40S, etc. pass through unchanged
```

### Step 2: Use Normalized Hardware in CLI Args

`experiment/adapters/base.py:70` - Update `_build_common_args()`:
```python
# Before
"--hardware", "H100",

# After
"--hardware", self._normalize_hardware(experiment.hardware),
```

### Step 3: Fix Blackbox Adapter can_run()

`experiment/adapters/blis_blackbox.py:32-48` - Add GPU check:
```python
def can_run(self, experiment: Experiment) -> bool:
    """True if defaults.yaml has trained coefficients for (model, GPU, TP) tuple."""
    try:
        with open(self._defaults_yaml) as fh:
            data = yaml.safe_load(fh)
    except (FileNotFoundError, PermissionError, yaml.YAMLError):
        return False

    model_lower = experiment.model.lower()
    normalized_hw = self._normalize_hardware(experiment.hardware)  # ← New

    for entry in (data or {}).get("models", []):
        if (
            entry.get("id", "").lower() == model_lower
            and entry.get("tensor_parallelism") == experiment.tp
            and entry.get("GPU") == normalized_hw  # ← New check
            and any(c != 0 for c in entry.get("alpha_coeffs", []))
        ):
            return True
    return False
```

### Step 4: Update Tests

`tests/test_blis_adapters.py`:

1. **test_kv_offloading_flags_present** - Verify hardware is passed dynamically
2. **test_hardware_normalization** - New test for A100-80GB → A100-80 mapping
3. **test_blackbox_can_run_checks_hardware** - New test verifying GPU field is checked

## Testing Strategy

### Unit Tests

1. **test_hardware_normalization** (new)
   ```python
   def test_hardware_normalization():
       adapter = BLISRooflineAdapter("/tmp/blis")
       assert adapter._normalize_hardware("A100-80GB") == "A100-80"
       assert adapter._normalize_hardware("H100") == "H100"
       assert adapter._normalize_hardware("L40S") == "L40S"
   ```

2. **test_kv_offloading_flags_present** (update existing)
   - Change assertion to verify `experiment.hardware` is passed (not hardcoded "H100")
   - Test with both H100 and A100-80GB experiments

3. **test_blackbox_hardware_matching** (new)
   ```python
   def test_blackbox_can_run_checks_hardware(tmp_path):
       # defaults.yaml with H100 coeffs only
       defaults = _write_defaults_yaml(
           str(tmp_path),
           [("meta-llama/Llama-2-7b-hf", "H100", 1)]
       )
       adapter = BLISBlackboxAdapter("/tmp/blis", defaults_yaml=defaults)

       # H100 experiment should match
       exp_h100 = _make_experiment(model="meta-llama/Llama-2-7b-hf", hardware="H100")
       assert adapter.can_run(exp_h100) is True

       # A100-80GB experiment should NOT match (coeffs don't exist)
       exp_a100 = _make_experiment(model="meta-llama/Llama-2-7b-hf", hardware="A100-80GB")
       assert adapter.can_run(exp_a100) is False
   ```

4. **test_model_and_tp_in_args** (existing, no change)
   - Already verifies CLI args use experiment fields dynamically

### Integration Test (manual verification)

Run BLIS against A100-80GB experiment:
```bash
python -m experiment.run \
  --gt-dir vllm_data/ground_truth \
  --adapter blis-roofline \
  --blis-binary inference-sim/blis \
  --exp-id 38  # Llama-3.1-8B on A100-80GB
```

Verify BLIS command includes `--hardware A100-80` (normalized from A100-80GB).

## Risks

### Low Risk

**Hardware calibration mismatch:**
- Risk: BLIS might not have calibration data for A100-80GB
- Mitigation: BLIS has `hardware_config.json` with H100, A100-80, and L40S
- Verified: All manifest hardware types are supported

**Hardware normalization incomplete:**
- Risk: Future hardware types might need normalization but we don't handle them
- Mitigation: Normalization function has a pass-through default (returns input unchanged)
- New hardware types will fail clearly with BLIS error if not in hardware_config.json

**defaults.yaml might not have coefficients for all hardware:**
- Risk: BLISBlackboxAdapter might not have trained coefficients for A100-80GB experiments
- Mitigation: can_run() will correctly return False, adapter won't be used
- Other adapters (roofline, crossmodel) don't require trained coefficients

**Test brittleness:**
- Risk: Tests need to know about hardware normalization mapping
- Mitigation: Tests explicitly verify the normalization mapping is correct

## Alternatives Considered

### Alternative 1: Pass `--gpu-memory-utilization` too

**Why not:** It's only used for auto-calculating KV blocks, which we skip since we provide explicit `--total-kv-blocks` from vllm.log ground truth. Adds complexity with zero benefit to simulation accuracy.

### Alternative 2: Pass all manifest fields to BLIS

**Why not:** BLIS doesn't have flags for precision or data-parallelism. Would require BLIS changes first.

### Alternative 3: Validate hardware against supported list

**Why not:** Adds unnecessary complexity. BLIS will error clearly if hardware is unsupported. Let BLIS own its validation logic.

## Open Questions

None. All three issues have been identified with clear solutions:
1. Hardware normalization for A100-80GB → A100-80
2. Using normalized hardware in BLIS CLI args
3. Fixing blackbox adapter can_run() to check GPU field

Design is ready for implementation.
