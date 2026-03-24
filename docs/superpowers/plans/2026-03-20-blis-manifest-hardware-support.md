# BLIS Manifest Hardware Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable BLIS adapters to simulate experiments across all hardware types in the manifest (H100, A100-80GB, L40S) by using experiment hardware instead of hardcoded "H100", normalizing hardware names for BLIS compatibility, and fixing blackbox adapter's can_run() to check GPU field.

**Architecture:** Add hardware normalization helper in BaseBLISAdapter to map manifest names (A100-80GB) to BLIS names (A100-80). Update _build_common_args() to use normalized hardware from experiment. Fix BLISBlackboxAdapter.can_run() to check GPU field in defaults.yaml entries.

**Tech Stack:** Python 3.10+, pytest, subprocess, YAML parsing

**Spec:** `docs/superpowers/specs/2026-03-20-blis-manifest-hardware-support-design.md`

---

## File Structure

**Modified:**
- `experiment/adapters/base.py` - Add `_normalize_hardware()`, update `_build_common_args()` line 70
- `experiment/adapters/blis_blackbox.py` - Update `can_run()` to check GPU field (lines 32-48)
- `tests/test_blis_adapters.py` - Update existing test, add 2 new tests

**Context Files to Review:**
- `experiment/data_model.py` - Understand `Experiment` dataclass structure
- `vllm_data/ground_truth/experiments.json` - See hardware values in manifest
- `inference-sim/defaults.yaml` - See GPU field structure in trained coefficients
- `inference-sim/hardware_config.json` - See BLIS-supported hardware names

---

## Task 1: Add Hardware Normalization Helper

**Files:**
- Modify: `experiment/adapters/base.py` (add method before line 59)
- Test: `tests/test_blis_adapters.py` (new test class)

**Goal:** Add static method to normalize manifest hardware names to BLIS-compatible names.

- [ ] **Step 1: Write the failing test for hardware normalization**

Add to `tests/test_blis_adapters.py` after line 174 (after `TestTrainedRooflineCanRun`):

```python


# ---------------------------------------------------------------------------
# Tests: hardware normalization
# ---------------------------------------------------------------------------


class TestHardwareNormalization:
    def test_a100_80gb_normalization(self):
        """A100-80GB from manifest should normalize to A100-80 for BLIS."""
        adapter = BLISRooflineAdapter("/tmp/blis")
        assert adapter._normalize_hardware("A100-80GB") == "A100-80"

    def test_h100_passes_through(self):
        """H100 should pass through unchanged."""
        adapter = BLISRooflineAdapter("/tmp/blis")
        assert adapter._normalize_hardware("H100") == "H100"

    def test_l40s_passes_through(self):
        """L40S should pass through unchanged."""
        adapter = BLISRooflineAdapter("/tmp/blis")
        assert adapter._normalize_hardware("L40S") == "L40S"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_blis_adapters.py::TestHardwareNormalization -v`

Expected: FAIL with `AttributeError: 'BLISRooflineAdapter' object has no attribute '_normalize_hardware'`

- [ ] **Step 3: Implement hardware normalization in base.py**

Add to `experiment/adapters/base.py` after line 57:

```python
    @staticmethod
    def _normalize_hardware(hardware: str) -> str:
        """Normalize manifest hardware names to BLIS-compatible names.

        The manifest uses "A100-80GB" but BLIS expects "A100-80" in its
        hardware_config.json and defaults.yaml. Other hardware types
        (H100, L40S) pass through unchanged.
        """
        if hardware == "A100-80GB":
            return "A100-80"
        return hardware
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_blis_adapters.py::TestHardwareNormalization -v`

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add experiment/adapters/base.py tests/test_blis_adapters.py
git commit -m "feat: add hardware normalization helper for BLIS adapters

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Use Normalized Hardware in CLI Args

**Files:**
- Modify: `experiment/adapters/base.py:70`
- Test: `tests/test_blis_adapters.py` (update + add new test)

- [ ] **Step 1: Update test_kv_offloading_flags_present to check hardware**

In `tests/test_blis_adapters.py` at line ~268, add before closing:

```python
        # Check hardware is passed from experiment
        idx = called_args.index("--hardware")
        assert called_args[idx + 1] == "H100"
```

- [ ] **Step 2: Run test - should pass with current code**

Run: `pytest tests/test_blis_adapters.py::TestBLISCLIArgs::test_kv_offloading_flags_present -v`

- [ ] **Step 3: Add test for A100-80GB normalization**

Add after `test_kv_offloading_flags_present`:

```python
    @patch("experiment.adapters.blis_roofline.subprocess.run")
    def test_hardware_from_experiment_normalized(self, mock_run):
        """Hardware should come from experiment and be normalized."""
        mock_run.return_value = MagicMock()
        adapter = BLISRooflineAdapter("/usr/local/bin/blis")
        exp = _make_experiment()
        exp.hardware = "A100-80GB"
        
        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)
        
        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--hardware")
        assert called_args[idx + 1] == "A100-80"
```

- [ ] **Step 4: Run test - should fail**

Run: `pytest tests/test_blis_adapters.py::TestBLISCLIArgs::test_hardware_from_experiment_normalized -v`

Expected: FAIL showing "H100" != "A100-80"

- [ ] **Step 5: Update base.py line 70**

In `experiment/adapters/base.py`, find line 70:
```python
            "--hardware", "H100",
```

Replace with:
```python
            "--hardware", self._normalize_hardware(experiment.hardware),
```

- [ ] **Step 6: Run tests - should pass**

Run: `pytest tests/test_blis_adapters.py::TestBLISCLIArgs -v`

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add experiment/adapters/base.py tests/test_blis_adapters.py
git commit -m "feat: use normalized experiment hardware in BLIS CLI args

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Fix Blackbox can_run() GPU Check

**Files:**
- Modify: `experiment/adapters/blis_blackbox.py:32-48`
- Test: `tests/test_blis_adapters.py` (new test)

- [ ] **Step 1: Write failing test**

Add to `tests/test_blis_adapters.py` after line 153:

```python
    def test_gpu_mismatch_rejects(self, tmp_path):
        """Coefficients for H100 should not match A100-80GB experiment."""
        data = {
            "models": [{
                "id": "meta-llama/Llama-2-7b-hf",
                "GPU": "H100",
                "tensor_parallelism": 1,
                "alpha_coeffs": [1.0, 2.0, 3.0],
            }]
        }
        defaults_path = os.path.join(str(tmp_path), "defaults.yaml")
        with open(defaults_path, "w") as fh:
            yaml.dump(data, fh)
        
        adapter = BLISBlackboxAdapter("/tmp/blis", defaults_yaml=defaults_path)
        
        # H100 should match
        exp_h100 = _make_experiment(model="meta-llama/Llama-2-7b-hf", tp=1)
        exp_h100.hardware = "H100"
        assert adapter.can_run(exp_h100) is True
        
        # A100-80GB should NOT match
        exp_a100 = _make_experiment(model="meta-llama/Llama-2-7b-hf", tp=1)
        exp_a100.hardware = "A100-80GB"
        assert adapter.can_run(exp_a100) is False
```

- [ ] **Step 2: Run test - should fail**

Run: `pytest tests/test_blis_adapters.py::TestBlackboxCanRun::test_gpu_mismatch_rejects -v`

Expected: FAIL on second assertion

- [ ] **Step 3: Update can_run() in blis_blackbox.py**

Replace lines 32-48:

```python
    def can_run(self, experiment: Experiment) -> bool:
        """True if defaults.yaml has trained coefficients for (model, GPU, TP) tuple."""
        try:
            with open(self._defaults_yaml) as fh:
                data = yaml.safe_load(fh)
        except (FileNotFoundError, PermissionError, yaml.YAMLError):
            return False

        model_lower = experiment.model.lower()
        normalized_hw = self._normalize_hardware(experiment.hardware)

        for entry in (data or {}).get("models", []):
            if (
                entry.get("id", "").lower() == model_lower
                and entry.get("tensor_parallelism") == experiment.tp
                and entry.get("GPU") == normalized_hw
                and any(c != 0 for c in entry.get("alpha_coeffs", []))
            ):
                return True
        return False
```

- [ ] **Step 4: Run test - should pass**

Run: `pytest tests/test_blis_adapters.py::TestBlackboxCanRun -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add experiment/adapters/blis_blackbox.py tests/test_blis_adapters.py
git commit -m "feat: check GPU field in blackbox adapter can_run()

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Run Full Test Suite

- [ ] **Step 1: Run all BLIS tests**

Run: `pytest tests/test_blis_adapters.py -v`

Expected: All PASS

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`

Expected: All PASS

---

## Success Criteria

✅ Hardware normalization helper added
✅ CLI args use normalized hardware
✅ Blackbox checks GPU field  
✅ All tests pass
✅ 3 commits total
