# BLIS-Trained-Physics Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add BLISTrainedPhysicsAdapter that wraps inference-sim's `--latency-model trained-physics` and update figures to use trained-physics instead of evolved.

**Architecture:** Create new adapter following BLISTrainedRooflineAdapter pattern (simple pass-through to BLIS binary with no coefficient passing), update figure configuration dictionaries to replace blis-evolved with blis-trained-physics, preserve BLISEvolvedAdapter entirely for backwards compatibility.

**Tech Stack:** Python 3.11+, subprocess, pytest, matplotlib (figures)

---

## File Structure

**Files to CREATE:**
- `experiment/adapters/blis_trained_physics.py` - New adapter (44 lines, pattern from blis_trained_roofline.py)

**Files to MODIFY:**
- `experiment/adapters/__init__.py` - Add export (1 import + 1 __all__ entry)
- `experiment/run.py` - Add to ALL_ADAPTER_NAMES and factory (1 list entry + 1 import + 2 lines in factory)
- `experiment/figures.py` - Replace blis-evolved → blis-trained-physics (5 dicts + ~10 function occurrences)
- `tests/test_blis_adapters.py` - Add test for new adapter (~40 lines)

**Files to PRESERVE (verify no changes):**
- `experiment/adapters/blis_evolved.py` - Zero changes
- `compare_iterations.py` - Zero changes

---

## Task 1: Create BLISTrainedPhysicsAdapter

**Files:**
- Create: `experiment/adapters/blis_trained_physics.py`
- Reference: `experiment/adapters/blis_trained_roofline.py` (pattern to follow)
- Reference: `experiment/adapters/base.py` (BaseBLISAdapter interface)

- [ ] **Step 1: Write the failing test**

Create test first to define interface:

```python
# Add to tests/test_blis_adapters.py at end of file

def test_blis_trained_physics_adapter(tmp_path):
    """Test BLIS trained-physics adapter invocation."""
    # Create mock BLIS binary
    blis_binary = tmp_path / "inference-sim" / "blis"
    blis_binary.parent.mkdir(parents=True)
    blis_binary.write_text("#!/bin/bash\necho 'mock'")
    blis_binary.chmod(0o755)

    # Create adapter
    from experiment.adapters.blis_trained_physics import BLISTrainedPhysicsAdapter
    adapter = BLISTrainedPhysicsAdapter(str(blis_binary))

    # Verify adapter name
    assert adapter.name == "blis-trained-physics"

    # Create test experiment
    from experiment.data_model import Experiment
    experiment = Experiment(
        model="qwen/qwen3-14b",
        tp=1,
        workload="chatbot",
        mbt=512,
        rate=10.0,
        num_requests=100,
        stages=[],
    )

    # Mock BLIS execution to test command construction
    import subprocess
    from unittest.mock import patch, MagicMock

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = b""
    mock_result.stderr = b""

    # Create mock results file
    results_file_content = {
        "metrics": {
            "ttft_p50": 0.05,
            "itl_p50": 0.01,
            "e2e_p50": 0.5,
        }
    }

    with patch("subprocess.run", return_value=mock_result) as mock_run, \
         patch("builtins.open", create=True) as mock_open, \
         patch("json.load", return_value=results_file_content):

        result = adapter.run(experiment)

        # Verify command construction
        args = mock_run.call_args[0][0]
        assert any("--latency-model" in str(arg) for arg in args)
        assert any("trained-physics" in str(arg) for arg in args)

        # Verify NO coefficient flags (uses defaults.yaml)
        args_str = " ".join(str(arg) for arg in args)
        assert "--alpha-coeffs" not in args_str
        assert "--beta-coeffs" not in args_str

    # Verify result object
    assert result.ttft_p50 == 0.05
    assert result.model == experiment.model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_blis_adapters.py::test_blis_trained_physics_adapter -v`

Expected: FAIL with "cannot import name 'BLISTrainedPhysicsAdapter'"

- [ ] **Step 3: Create adapter file with minimal implementation**

```python
# Create: experiment/adapters/blis_trained_physics.py

"""BLIS trained-physics adapter — roofline basis functions with learned corrections."""

from __future__ import annotations

import os
import subprocess
import tempfile

from experiment.adapters.base import BaseBLISAdapter
from experiment.data_model import Experiment, SimulatorResult


class BLISTrainedPhysicsAdapter(BaseBLISAdapter):
    """BLIS simulator with ``--latency-model trained-physics``.

    Uses globally-fitted roofline basis functions with architecture-aware
    corrections from ``defaults.yaml`` (loaded by the BLIS binary automatically).
    Generalizes across model architectures, workloads, and TP configurations
    without per-model calibration.

    The trained-physics model uses 13 coefficients (10 beta + 3 alpha):
    - Beta coefficients: prefill compute/memory split, decode compute/memory split,
      weight loading, TP communication, layer overhead, batch overhead, step overhead,
      MoE-layer overhead (architecture-aware)
    - Alpha coefficients: API queueing, post-decode fixed overhead, per-token overhead
    """

    @property
    def name(self) -> str:
        return "blis-trained-physics"

    def run(self, experiment: Experiment) -> SimulatorResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "workload_spec.yaml")
            self._write_workload_spec(experiment, spec_path)

            results_path = os.path.join(tmpdir, "results.json")
            args = self._build_common_args(experiment, spec_path, results_path)
            args.extend(["--latency-model", "trained-physics"])

            try:
                subprocess.run(args, capture_output=True, check=True, cwd=self._blis_dir)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"BLIS trained-physics failed (rc={exc.returncode}) for "
                    f"{experiment.model}: {stderr}"
                ) from exc

            return self._parse_blis_results(results_path, experiment)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_blis_adapters.py::test_blis_trained_physics_adapter -v`

Expected: PASS (test may need minor adjustments for mocking)

- [ ] **Step 5: Commit**

```bash
git add experiment/adapters/blis_trained_physics.py tests/test_blis_adapters.py
git commit -m "feat: add BLISTrainedPhysicsAdapter

Add new adapter for inference-sim's --latency-model trained-physics mode.
Uses globally-fitted roofline basis functions with architecture-aware
corrections from defaults.yaml. No per-model calibration required.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Update Adapter Exports

**Files:**
- Modify: `experiment/adapters/__init__.py`

- [ ] **Step 1: Add import statement**

```python
# Add to experiment/adapters/__init__.py after existing imports (around line 3)

from experiment.adapters.blis_trained_physics import BLISTrainedPhysicsAdapter
```

- [ ] **Step 2: Add to __all__ list**

```python
# Update __all__ list in experiment/adapters/__init__.py (around line 6-12)

__all__ = [
    "AIConfiguratorEstimateAdapter",
    "BaseBLISAdapter",
    "BLISEvolvedAdapter",
    "BLISTrainedPhysicsAdapter",  # ADD THIS LINE
    "LLMServingSimAdapter",
    "SimulatorAdapter",
]
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from experiment.adapters import BLISTrainedPhysicsAdapter; print(BLISTrainedPhysicsAdapter)"`

Expected: `<class 'experiment.adapters.blis_trained_physics.BLISTrainedPhysicsAdapter'>`

- [ ] **Step 4: Commit**

```bash
git add experiment/adapters/__init__.py
git commit -m "feat: export BLISTrainedPhysicsAdapter

Add new adapter to public exports.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update Orchestrator

**Files:**
- Modify: `experiment/run.py`

- [ ] **Step 1: Add import statement**

```python
# Add to experiment/run.py after existing adapter imports (around line 22)

from experiment.adapters.blis_trained_physics import BLISTrainedPhysicsAdapter
```

- [ ] **Step 2: Add to ALL_ADAPTER_NAMES list**

```python
# Update ALL_ADAPTER_NAMES in experiment/run.py (around line 32-42)

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

- [ ] **Step 3: Add to adapter factory function**

Find the `create_adapter` function or the adapter factory logic (search for "elif name == "). Add new case:

```python
# Add to adapter factory in experiment/run.py (search for "blis-trained-roofline" case and add after it)

    elif name == "blis-trained-physics":
        return BLISTrainedPhysicsAdapter(blis_binary)
```

- [ ] **Step 4: Verify orchestrator recognizes adapter**

Run: `python -m experiment.run --help | grep blis-trained-physics`

Expected: Should show adapter in help text or no error when parsing

- [ ] **Step 5: Commit**

```bash
git add experiment/run.py
git commit -m "feat: add blis-trained-physics to orchestrator

Add new adapter to ALL_ADAPTER_NAMES list and factory function.
Both blis-evolved and blis-trained-physics are now available.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Update Figure Configuration Dictionaries

**Files:**
- Modify: `experiment/figures.py` (lines 35, 45, 55, 65, 75)

- [ ] **Step 1: Update SIMULATOR_ORDER (line ~35)**

```python
# In experiment/figures.py around line 35

SIMULATOR_ORDER = [
    "llmservingsim",
    "vidur",
    "blis-trained-physics",  # CHANGED: was "blis-evolved"
    "blis-trained-roofline",
    "blis-crossmodel",
    "blis-roofline",
    "blis-blackbox",
    "llm-optimizer-estimate",
    "aiconfigurator-estimate",
]
```

- [ ] **Step 2: Update SIMULATOR_LABELS (line ~45)**

```python
# In experiment/figures.py around line 45

SIMULATOR_LABELS = {
    "llmservingsim": "LLMServingSim",
    "vidur": "Vidur",
    "blis-trained-physics": "BLIS-Trained-Physics",  # CHANGED: was "blis-evolved": "BLIS-Evolved"
    "blis-trained-roofline": "BLIS-Trained-Roofline",
    "blis-crossmodel": "BLIS-CrossModel",
    "blis-roofline": "BLIS-Roofline",
    "blis-blackbox": "BLIS-Blackbox",
    "llm-optimizer-estimate": "LLM-Optimizer-Est",
    "aiconfigurator-estimate": "AIConfigurator-Est",
}
```

- [ ] **Step 3: Update SIMULATOR_COLORS (line ~55)**

```python
# In experiment/figures.py around line 55

SIMULATOR_COLORS = {
    "llmservingsim": "#3B82F6",
    "vidur": "#10B981",
    "blis-trained-physics": "#D946EF",  # CHANGED: was "blis-evolved": "#D946EF"
    "blis-trained-roofline": "#8B5CF6",
    "blis-crossmodel": "#F59E0B",
    "blis-roofline": "#EF4444",
    "blis-blackbox": "#6B7280",
    "llm-optimizer-estimate": "#EC4899",
    "aiconfigurator-estimate": "#14B8A6",
}
```

- [ ] **Step 4: Update SIMULATOR_HATCHES (line ~65)**

```python
# In experiment/figures.py around line 65

SIMULATOR_HATCHES = {
    "llmservingsim": "//",
    "vidur": "\\\\",
    "blis-trained-physics": "||",  # CHANGED: was "blis-evolved": "||"
    "blis-trained-roofline": "--",
    "blis-crossmodel": "++",
    "blis-roofline": "xx",
    "blis-blackbox": "..",
    "llm-optimizer-estimate": "oo",
    "aiconfigurator-estimate": "**",
}
```

- [ ] **Step 5: Update SIMULATOR_MARKERS (line ~75)**

```python
# In experiment/figures.py around line 75

SIMULATOR_MARKERS = {
    "llmservingsim": "o",
    "vidur": "s",
    "blis-trained-physics": "*",  # CHANGED: was "blis-evolved": "*"
    "blis-trained-roofline": "^",
    "blis-crossmodel": "D",
    "blis-roofline": "v",
    "blis-blackbox": "p",
    "llm-optimizer-estimate": "h",
    "aiconfigurator-estimate": "X",
}
```

- [ ] **Step 6: Verify no syntax errors**

Run: `python -c "import experiment.figures; print('OK')"`

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add experiment/figures.py
git commit -m "feat: update figure config for trained-physics

Replace blis-evolved with blis-trained-physics in all configuration
dictionaries (SIMULATOR_ORDER, LABELS, COLORS, HATCHES, MARKERS).
Reuse purple color and || hatch pattern for consistency.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update Figure Function References

**Files:**
- Modify: `experiment/figures.py` (multiple functions, ~10 occurrences)

- [ ] **Step 1: Update docstring example (line ~1053)**

Find line with docstring example showing `["blis-roofline", "blis-evolved"]`:

```python
# In experiment/figures.py around line 1053

"""
    Single simulator or list of simulators to compare (e.g., ["blis-roofline", "blis-trained-physics"])
"""
```

- [ ] **Step 2: Update figure_1 function (lines ~1231-1246)**

```python
# In experiment/figures.py around line 1231-1246

def figure_1(df: pd.DataFrame, output_dir: str) -> None:
    """Figure 1: BLIS-Roofline vs BLIS-Trained-Physics MAPE across 7 model architectures on H100, default config."""
    # Filter to BLIS-Roofline and BLIS-Trained-Physics
    df = df[df["simulator"].isin(["blis-roofline", "blis-trained-physics"])]

    # ... rest of function ...

    plot_per_model_error_comparison(
        df,
        metrics=["ttft_mape", "itl_mape", "e2e_mape"],
        output_path=os.path.join(output_dir, "figure_1.pdf"),
        title="BLIS-Roofline vs BLIS-Trained-Physics: Prediction Error Across Model Architectures ↓",
        # ... rest of args ...
    )
```

- [ ] **Step 3: Update annotation offset dictionary (line ~1641)**

Find annotation offset dictionary with `"blis-evolved"` key:

```python
# In experiment/figures.py around line 1641

offset_dict = {
    "blis-roofline": (15, 20),
    "blis-trained-physics": (15, -20),  # CHANGED: was "blis-evolved": (15, -20)
    # ... other simulators ...
}
```

- [ ] **Step 4: Update comparison tuples (lines ~2131-2134)**

Find the comparison tuples for figure generation:

```python
# In experiment/figures.py around lines 2131-2134

comparisons = [
    (["blis-roofline", "blis-trained-physics"], "vidur", "blis_vs_vidur.pdf"),  # CHANGED
    (["blis-roofline", "blis-trained-physics"], "llm-optimizer-estimate", "blis_vs_llm_optimizer.pdf"),  # CHANGED
    (["blis-roofline", "blis-trained-physics"], "aiconfigurator-estimate", "blis_vs_aiconfigurator.pdf"),  # CHANGED
    (["blis-roofline", "blis-trained-physics"], "llmservingsim", "blis_vs_llmservingsim.pdf"),  # CHANGED
]
```

- [ ] **Step 5: Search for remaining occurrences**

Run: `grep -n "blis-evolved" experiment/figures.py`

Expected: Only comments or historical references (no code references)

If any code references found, update them to `"blis-trained-physics"`

- [ ] **Step 6: Verify no syntax errors**

Run: `python -c "import experiment.figures; print('OK')"`

Expected: `OK`

- [ ] **Step 7: Run figure tests**

Run: `pytest tests/test_figures.py -v`

Expected: All tests pass (may need to update test fixtures if they hardcode blis-evolved)

- [ ] **Step 8: Commit**

```bash
git add experiment/figures.py tests/test_figures.py
git commit -m "feat: replace blis-evolved with trained-physics in figures

Update all figure functions to use blis-trained-physics:
- figure_1() filter and title
- Docstring examples
- Annotation offset dictionary
- Comparison tuples for simulator comparisons

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Verification and Testing

**Files:**
- Verify: All modified files
- Verify: Preserved files unchanged

- [ ] **Step 1: Verify BLISEvolvedAdapter unchanged**

Run: `git diff HEAD~6 -- experiment/adapters/blis_evolved.py`

Expected: Empty output (no changes)

- [ ] **Step 2: Verify compare_iterations.py unchanged**

Run: `git diff HEAD~6 -- compare_iterations.py`

Expected: Empty output (no changes)

- [ ] **Step 3: Verify both adapters in ALL_ADAPTER_NAMES**

Run: `python -c "from experiment.run import ALL_ADAPTER_NAMES; assert 'blis-evolved' in ALL_ADAPTER_NAMES; assert 'blis-trained-physics' in ALL_ADAPTER_NAMES; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Verify both adapters can be imported**

Run: `python -c "from experiment.adapters import BLISEvolvedAdapter, BLISTrainedPhysicsAdapter; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Run all adapter tests**

Run: `pytest tests/test_blis_adapters.py -v`

Expected: All tests pass (including new test_blis_trained_physics_adapter)

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`

Expected: All tests pass

- [ ] **Step 7: Verify compare_iterations.py still works**

Run: `python compare_iterations.py 2>&1 | head -5`

Expected: No import errors (may error on missing data files, but should not have import/code errors)

- [ ] **Step 8: Final grep check for blis-evolved in code**

Run: `grep -r "blis-evolved" experiment/ --include="*.py" | grep -v "# " | grep -v compare_iterations`

Expected: Only in:
- `experiment/run.py`: in ALL_ADAPTER_NAMES list (preserved)
- `experiment/adapters/blis_evolved.py`: adapter name property (preserved)

Not in:
- `experiment/figures.py`: should be replaced with blis-trained-physics

---

## Task 7: Documentation Update

**Files:**
- Create: Add usage example to adapter docstring

- [ ] **Step 1: Verify adapter docstring is complete**

Read: `experiment/adapters/blis_trained_physics.py`

Verify docstring includes:
- What the adapter does
- What latency model it uses
- Coefficient information
- That it uses defaults.yaml

Expected: Already complete from Task 1

- [ ] **Step 2: Add usage note to README or adapter README if exists**

Check: `experiment/adapters/README.md` exists

If yes, add entry for blis-trained-physics:

```markdown
## BLIS-Trained-Physics

Uses `--latency-model trained-physics` with globally-fitted roofline basis functions and architecture-aware corrections. Generalizes across model architectures without per-model calibration. Recommended for new models.

Usage:
```bash
python -m experiment.run --adapters blis-trained-physics
```
```

If README doesn't exist, skip this step.

- [ ] **Step 3: Commit**

```bash
git add experiment/adapters/README.md  # if modified
git commit -m "docs: add blis-trained-physics usage

Add documentation for new adapter.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

### Spec Coverage

From `docs/superpowers/specs/2026-04-07-trained-physics-adapter-design.md`:

- [x] **Requirement 1**: Create BLISTrainedPhysicsAdapter → Task 1
- [x] **Requirement 2**: Update figures.py → Task 4, Task 5
- [x] **Requirement 3**: Add to orchestrator → Task 3
- [x] **Requirement 4**: Preserve BLISEvolvedAdapter → Verified in Task 6
- [x] **Requirement 5**: Preserve compare_iterations.py → Verified in Task 6
- [x] **Requirement 6**: Both adapters available → Task 3, verified in Task 6

### Placeholder Scan

- [x] No "TBD", "TODO", "implement later", "fill in details"
- [x] No "add appropriate error handling" without showing code
- [x] No "write tests for the above" without actual test code
- [x] No "similar to Task N" without repeating code
- [x] All code steps include actual code blocks
- [x] All types/functions referenced are defined in tasks

### Type Consistency

- [x] Adapter name: `"blis-trained-physics"` used consistently
- [x] Class name: `BLISTrainedPhysicsAdapter` used consistently
- [x] Method signatures match base class (BaseBLISAdapter)
- [x] Config dict keys: `"blis-trained-physics"` used consistently

### Completeness

- [x] All files have exact paths
- [x] All commands have expected output
- [x] All commits have messages
- [x] All tests have assertions
- [x] Verification steps included

---

## Execution Notes

**Test-Driven Development:** This plan follows strict TDD:
1. Task 1 writes test first (Step 1), verifies failure (Step 2), implements (Step 3), verifies pass (Step 4)
2. All other tasks modify existing code with verification steps

**Frequent Commits:** Each task ends with a commit (Tasks 1-7)

**DRY Principle:** Adapter implementation (~44 lines) follows pattern from blis_trained_roofline.py, reusing all BaseBLISAdapter methods

**YAGNI Principle:** No extra features - only what spec requires:
- No iteration parameter (unlike evolved)
- No coefficient formatting (uses defaults.yaml)
- No extra configuration options

**Preservation Verified:** Task 6 explicitly verifies:
- blis_evolved.py unchanged (git diff)
- compare_iterations.py unchanged (git diff)
- Both adapters available (Python import check)

**Expected Duration:** ~30-45 minutes for all 7 tasks
