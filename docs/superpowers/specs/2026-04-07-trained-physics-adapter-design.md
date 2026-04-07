# BLIS-Trained-Physics Adapter Design

**Date:** 2026-04-07
**Status:** Approved
**Approach:** Minimal Change (Approach 1)

## Overview

Add a new `BLISTrainedPhysicsAdapter` to wrap inference-sim's `--latency-model trained-physics` mode and update figure generation to use trained-physics instead of evolved. The trained-physics latency model uses 13 coefficients (10 beta + 3 alpha) with physics-informed roofline basis functions and learned corrections, designed to generalize across model architectures, workloads, and TP configurations without per-model calibration.

## Requirements

1. Create new `BLISTrainedPhysicsAdapter` that uses `--latency-model trained-physics`
2. Update `experiment/figures.py` to replace all `"blis-evolved"` references with `"blis-trained-physics"`
3. Add new adapter to orchestrator and exports
4. **Preserve `BLISEvolvedAdapter` entirely** - zero code changes
5. **Preserve `compare_iterations.py`** - continues using only `"blis-evolved"`
6. Both adapters remain available for backwards compatibility

## Architecture

### Component Overview

```
experiment/
├── adapters/
│   ├── blis_evolved.py           [PRESERVE - no changes]
│   ├── blis_trained_physics.py   [NEW - create this file]
│   ├── blis_trained_roofline.py  [reference pattern]
│   └── __init__.py                [UPDATE - add new export]
├── figures.py                     [UPDATE - replace evolved → trained-physics]
└── run.py                         [UPDATE - add to ALL_ADAPTER_NAMES]

compare_iterations.py              [PRESERVE - no changes, uses only blis-evolved]
check_saturation.py                [PRESERVE - no changes]
```

### BLISTrainedPhysicsAdapter

New adapter file: `experiment/adapters/blis_trained_physics.py`

**Interface:**
```python
class BLISTrainedPhysicsAdapter(BaseBLISAdapter):
    """BLIS simulator with --latency-model trained-physics.

    Uses globally-fitted roofline basis functions with architecture-aware
    corrections from defaults.yaml. Generalizes across model architectures,
    workloads, and TP configurations without per-model calibration.

    The trained-physics model uses 13 coefficients (10 beta + 3 alpha):
    - Beta coefficients: prefill compute/memory split, decode compute/memory split,
      weight loading, TP communication, layer overhead, batch overhead, step overhead,
      MoE-layer overhead (architecture-aware)
    - Alpha coefficients: API queueing, post-decode fixed overhead, per-token overhead

    Coefficients are automatically loaded from inference-sim/defaults.yaml.
    No per-model calibration required.
    """

    @property
    def name(self) -> str:
        return "blis-trained-physics"

    def run(self, experiment: Experiment) -> SimulatorResult:
        # Build command: ./blis run --latency-model trained-physics ...
        # No --alpha-coeffs or --beta-coeffs flags needed
        # BLIS binary loads coefficients from defaults.yaml automatically
```

**Key differences from BLISEvolvedAdapter:**
- No `iteration` parameter (uses single global coefficient set)
- No `_format_coeffs()` method (coefficients from defaults.yaml)
- No coefficient passing via `--alpha-coeffs` / `--beta-coeffs`
- Simpler implementation (~44 lines vs ~232 lines)

**Pattern followed:** `BLISTrainedRooflineAdapter` (lines 1-44 of `blis_trained_roofline.py`)

## Data Flow

### Request Flow

1. **Experiment input** → Adapter receives `Experiment` object (model, workload, TP config)
2. **Workload spec** → Write YAML spec to temp directory (inherited from `BaseBLISAdapter`)
3. **BLIS invocation** → Execute:
   ```bash
   ./blis run \
     --latency-model trained-physics \
     --hardware H100 \
     --tp N \
     --workload-spec /tmp/workload_spec.yaml \
     --results-file /tmp/results.json
   ```
4. **Coefficient resolution** → BLIS binary loads `trained_physics_coefficients` from `inference-sim/defaults.yaml`
5. **Simulation** → BLIS computes step times using roofline basis functions × learned corrections
6. **Results parsing** → Parse JSON results (TTFT, ITL, E2E metrics) into `SimulatorResult`
7. **Return** → Adapter returns results to orchestrator

### Figure Generation Flow

1. **CSV input** → Read `runtime.csv` with columns: `[simulator, model, ttft_p50, itl_p50, e2e_p50, ...]`
2. **Filter** → Select rows where `simulator == "blis-trained-physics"` (was `"blis-evolved"`)
3. **Plot** → Generate bar charts, scatter plots, Pareto curves using trained-physics data
4. **Style** → Apply purple color (`#D946EF`), hatch pattern (`||`), marker (`*`) - same as evolved
5. **Labels** → Display "BLIS-Trained-Physics" in legends/titles
6. **Save** → Output PDF figures to `results/figures/`

## Implementation Details

### 1. Figure Configuration Updates

File: `experiment/figures.py`

**Configuration dictionaries to update:**

```python
# Line 35: SIMULATOR_ORDER
SIMULATOR_ORDER = [
    "llmservingsim",
    "vidur",
    "blis-trained-physics",  # was: "blis-evolved"
    "blis-trained-roofline",
    # ...
]

# Line 45: SIMULATOR_LABELS
SIMULATOR_LABELS = {
    "blis-trained-physics": "BLIS-Trained-Physics",  # was: "blis-evolved": "BLIS-Evolved"
    # ...
}

# Line 55: SIMULATOR_COLORS (reuse evolved color for consistency)
SIMULATOR_COLORS = {
    "blis-trained-physics": "#D946EF",  # was: "blis-evolved": "#D946EF"
    # ...
}

# Line 65: SIMULATOR_HATCHES
SIMULATOR_HATCHES = {
    "blis-trained-physics": "||",  # was: "blis-evolved": "||"
    # ...
}

# Line 75: SIMULATOR_MARKERS
SIMULATOR_MARKERS = {
    "blis-trained-physics": "*",  # was: "blis-evolved": "*"
    # ...
}
```

**Figure functions to update (~10 occurrences):**

- Line 1053: Docstring example `["blis-roofline", "blis-evolved"]` → `["blis-roofline", "blis-trained-physics"]`
- Line 1231-1246: `figure_1()` function - filter and title
- Line 1641: Annotation offset dictionary key
- Line 2131-2134: Comparison tuples in figure generation loop

**Pattern:** Find all `"blis-evolved"` string literals and replace with `"blis-trained-physics"` (except in comments about historical data).

### 2. Orchestrator Updates

File: `experiment/run.py`

**Line 32-42: ALL_ADAPTER_NAMES list**

```python
ALL_ADAPTER_NAMES = [
    "blis-blackbox",
    "blis-roofline",
    "blis-crossmodel",
    "blis-evolved",              # KEEP - preserve for backwards compatibility
    "blis-trained-physics",      # ADD - new adapter
    "blis-trained-roofline",
    "vidur",
    "llm-optimizer-estimate",
    "aiconfigurator-estimate",
    "llmservingsim",
]
```

**Adapter factory (add case to existing if/elif chain):**

```python
def create_adapter(name: str, blis_binary: str, **kwargs) -> SimulatorAdapter:
    # ... existing cases ...
    elif name == "blis-trained-physics":
        return BLISTrainedPhysicsAdapter(blis_binary)
    # ... rest of cases ...
```

**Import statement (line 18-27):**

```python
from experiment.adapters.blis_trained_physics import BLISTrainedPhysicsAdapter
```

### 3. Export Updates

File: `experiment/adapters/__init__.py`

```python
from experiment.adapters.blis_evolved import BLISEvolvedAdapter          # KEEP
from experiment.adapters.blis_trained_physics import BLISTrainedPhysicsAdapter  # ADD

__all__ = [
    # ...
    "BLISEvolvedAdapter",          # KEEP
    "BLISTrainedPhysicsAdapter",   # ADD
    # ...
]
```

## Scope Constraints

### Files to CREATE

- ✅ `experiment/adapters/blis_trained_physics.py` - New adapter (separate from blis_evolved.py)

### Files to UPDATE

- ✅ `experiment/figures.py` - Replace `"blis-evolved"` → `"blis-trained-physics"` in config dicts and functions
- ✅ `experiment/run.py` - Add `"blis-trained-physics"` to `ALL_ADAPTER_NAMES` (keep `"blis-evolved"`)
- ✅ `experiment/adapters/__init__.py` - Add `BLISTrainedPhysicsAdapter` export (keep `BLISEvolvedAdapter`)

### Files to PRESERVE (zero changes)

- 🔒 `experiment/adapters/blis_evolved.py` - **NO CHANGES** - Keep entire file unchanged
- 🔒 `compare_iterations.py` - **NO CHANGES** - Uses only `"blis-evolved"`, never `"blis-trained-physics"`
- 🔒 `check_saturation.py` - **NO CHANGES** - Ground truth validation script
- 🔒 `tests/test_blis_adapters.py` - Keep existing BLIS-Evolved tests (will add new tests for trained-physics)
- 🔒 All `results_iter*/` directories - Historical CSV data unchanged

## Error Handling

### Adapter Error Cases

1. **Missing BLIS binary**
   - Inherited from `BaseBLISAdapter.__init__()`
   - Raises `FileNotFoundError` with clear path message

2. **Subprocess failure (BLIS execution error)**
   - Caught in `subprocess.run(..., check=True)`
   - Raises `RuntimeError` with stderr and model context:
     ```python
     raise RuntimeError(
         f"BLIS trained-physics failed (rc={exc.returncode}) for "
         f"{experiment.model}: {stderr}"
     ) from exc
     ```

3. **Missing defaults.yaml coefficients**
   - BLIS binary errors if `trained_physics_coefficients` section missing in `defaults.yaml`
   - Stderr captured and included in RuntimeError
   - Clear error message from BLIS: "no trained_physics_coefficients found"

4. **Invalid results JSON**
   - Parsing errors in `_parse_blis_results()` (inherited from base)
   - Propagated with file path context

### Figure Generation Error Cases

1. **Missing simulator data in CSV**
   - Filter returns empty DataFrame
   - Figure functions skip or show empty plot with warning

2. **Invalid metric values**
   - Handled by existing figure code (matplotlib error handling)
   - NaN/inf values filtered before plotting

## Testing

### Unit Tests

File: `tests/test_blis_adapters.py`

**New test case:**

```python
def test_blis_trained_physics_adapter(tmp_path, mock_blis_binary):
    """Test BLIS trained-physics adapter invocation."""
    adapter = BLISTrainedPhysicsAdapter(str(mock_blis_binary))

    # Verify adapter name
    assert adapter.name == "blis-trained-physics"

    # Create test experiment
    experiment = create_test_experiment(...)

    # Mock BLIS execution
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = ...
        result = adapter.run(experiment)

    # Verify command construction
    args = mock_run.call_args[0][0]
    assert "--latency-model" in args
    assert "trained-physics" in args
    # Verify NO coefficient flags
    assert "--alpha-coeffs" not in args
    assert "--beta-coeffs" not in args

    # Verify results parsing
    assert result.ttft_p50 > 0
    assert result.model == experiment.model
```

**Tests to preserve:**
- `test_blis_evolved_adapter_iter16()` - unchanged
- `test_blis_evolved_adapter_iter24()` - unchanged
- `test_blis_evolved_adapter_iter26()` - unchanged
- All other adapter tests - unchanged

### Integration Testing

**Manual validation steps:**

1. **compare_iterations.py still works**
   ```bash
   python compare_iterations.py
   # Should run without modification
   # Should only reference "blis-evolved"
   ```

2. **BLIS-Evolved adapter still works**
   ```bash
   python -m experiment.run --adapters blis-evolved --blis-evolved-iteration 29
   # Should run successfully
   ```

3. **New BLIS-Trained-Physics adapter works**
   ```bash
   python -m experiment.run --adapters blis-trained-physics
   # Should run successfully
   # Should use defaults.yaml coefficients
   ```

4. **Figure regeneration shows new labels**
   ```bash
   python -m experiment.figures --input results/runtime.csv --output-dir results/figures
   # Generated PDFs should show "BLIS-Trained-Physics" labels
   # Should use purple color, || hatch, * marker
   ```

### Verification Checklist

After implementation:
- [ ] `compare_iterations.py` runs without modification
- [ ] `grep -r "blis-trained-physics" compare_iterations.py` returns 0 matches
- [ ] `git diff experiment/adapters/blis_evolved.py` shows 0 changes
- [ ] `ALL_ADAPTER_NAMES` includes both `"blis-evolved"` and `"blis-trained-physics"`
- [ ] `__init__.py` exports both `BLISEvolvedAdapter` and `BLISTrainedPhysicsAdapter`
- [ ] Running `--adapters blis-evolved` still works
- [ ] Running `--adapters blis-trained-physics` works
- [ ] New figures show "BLIS-Trained-Physics" (not "BLIS-Evolved")
- [ ] Unit tests pass for both adapters

## Trade-offs

### Why Approach 1 (Minimal Change)?

**Advantages:**
- Simple, low-risk implementation
- Minimal code changes reduce chance of breaking existing functionality
- Fast to implement and test
- BLISTrainedPhysicsAdapter simpler than BLISEvolvedAdapter (no iteration parameter)
- User can regenerate figures when ready without blocking code changes

**Disadvantages:**
- Doesn't automatically regenerate figures (user must re-run experiments)
- Some old results directories reference `blis-evolved` (acceptable - historical data)

**Why not Approach 2 (Comprehensive Migration)?**
- More time-consuming (regenerate all figures, update all docs)
- Requires re-running all experiments (could take hours)
- More changes = more review overhead
- inference-sim already validates trained-physics extensively

**Why not Approach 3 (Phased Rollout)?**
- Unnecessary complexity - inference-sim already recommends trained-physics
- Temporary figure crowding (both evolved and trained-physics lines)
- Multi-phase implementation delays delivery

## References

- **inference-sim trained-physics docs**: `inference-sim/docs/guide/latency-models.md` (lines 203-286)
- **defaults.yaml coefficients**: `inference-sim/defaults.yaml` (lines 1759-1778)
- **Pattern to follow**: `experiment/adapters/blis_trained_roofline.py` (44 lines, simple implementation)
- **inference-sim recommendation**: "Recommended for new models" (line 291 of latency-models.md)

## Success Criteria

1. ✅ New `BLISTrainedPhysicsAdapter` created and tested
2. ✅ All figure generation uses `"blis-trained-physics"` instead of `"blis-evolved"`
3. ✅ `BLISEvolvedAdapter` preserved with zero changes
4. ✅ `compare_iterations.py` works without modification
5. ✅ Both adapters available in `ALL_ADAPTER_NAMES`
6. ✅ Unit tests pass for both adapters
7. ✅ Manual validation checklist complete
