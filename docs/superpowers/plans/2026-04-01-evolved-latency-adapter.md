# BLIS Evolved Latency Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a simulator adapter for BLIS's evolved latency backend with iter16 coefficients

**Architecture:** Extend `BaseBLISAdapter` with evolved-specific CLI arguments (--latency-model evolved, --alpha-coeffs, --beta-coeffs). Static iter16 coefficients (3 alpha + 7 beta) for MoE/FP8-enhanced roofline model.

**Tech Stack:** Python 3.10+, pytest, subprocess, tempfile

---

## File Structure

```
experiment/adapters/
├── blis_evolved.py          # New: 65 lines (adapter class + constants)
├── __init__.py              # Modified: Add BLISEvolvedAdapter export
└── README.md                # Modified: Document evolved adapter

tests/
└── test_blis_adapters.py    # Modified: Add evolved adapter tests
```

**Key decisions:**
- Static iter16 coefficients as class constants (no config files)
- 6 decimal precision for coefficient formatting
- Follow existing adapter pattern (roofline/crossmodel/trained-roofline)

---

### Task 1: Coefficient Formatting Helper Tests

**Files:**
- Test: `tests/test_blis_adapters.py:465-520`

- [ ] **Step 1: Write test for coefficient formatting**

Add to `tests/test_blis_adapters.py` after line 464:

```python
# ---------------------------------------------------------------------------
# Tests: BLISEvolvedAdapter coefficient formatting
# ---------------------------------------------------------------------------


class TestBLISEvolvedCoefficients:
    def test_format_coeffs_three_values(self):
        """Alpha coefficients (3 values) should format with 6 decimals."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        adapter = BLISEvolvedAdapter("/tmp/blis")
        result = adapter._format_coeffs([15569.495449697066, 815.0556502348827, 45.705744318725586])
        assert result == "15569.495450,815.055650,45.705744"

    def test_format_coeffs_seven_values(self):
        """Beta coefficients (7 values) should format with 6 decimals."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        adapter = BLISEvolvedAdapter("/tmp/blis")
        result = adapter._format_coeffs([
            0.20081681581824434, 1.6173961192042448, 1.3603417361920076,
            0.39579536655780084, 62.19421689224744, 2.937563498958273, 169.37780505091155
        ])
        assert result == "0.200817,1.617396,1.360342,0.395795,62.194217,2.937563,169.377805"

    def test_format_coeffs_no_trailing_zeros(self):
        """Formatted coefficients should maintain 6 decimal places."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        adapter = BLISEvolvedAdapter("/tmp/blis")
        result = adapter._format_coeffs([1.0, 2.5, 3.123456789])
        assert result == "1.000000,2.500000,3.123457"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_blis_adapters.py::TestBLISEvolvedCoefficients -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'experiment.adapters.blis_evolved'"

---

### Task 2: CLI Arguments Test

**Files:**
- Test: `tests/test_blis_adapters.py:520-580`

- [ ] **Step 1: Write test for evolved CLI arguments**

Add to `tests/test_blis_adapters.py` after the coefficient formatting tests:

```python
class TestBLISEvolvedCLIArgs:
    @patch("experiment.adapters.blis_evolved.subprocess.run")
    def test_evolved_latency_model_flag(self, mock_run):
        """Evolved adapter should pass --latency-model evolved."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        mock_run.return_value = MagicMock()
        adapter = BLISEvolvedAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--latency-model")
        assert called_args[idx + 1] == "evolved"

    @patch("experiment.adapters.blis_evolved.subprocess.run")
    def test_evolved_alpha_coeffs_flag(self, mock_run):
        """Evolved adapter should pass --alpha-coeffs with iter16 values."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        mock_run.return_value = MagicMock()
        adapter = BLISEvolvedAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--alpha-coeffs")
        alpha_str = called_args[idx + 1]

        # Should be 3 comma-separated values
        assert alpha_str.count(',') == 2
        parts = alpha_str.split(',')
        assert len(parts) == 3

        # Verify first alpha coefficient (QueueingTime ≈ 15569.5)
        assert parts[0].startswith("15569.")

    @patch("experiment.adapters.blis_evolved.subprocess.run")
    def test_evolved_beta_coeffs_flag(self, mock_run):
        """Evolved adapter should pass --beta-coeffs with iter16 values."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        mock_run.return_value = MagicMock()
        adapter = BLISEvolvedAdapter("/usr/local/bin/blis")
        exp = _make_experiment()

        with patch.object(adapter, "_parse_blis_results") as mock_parse:
            mock_parse.return_value = MagicMock()
            adapter.run(exp)

        called_args = mock_run.call_args[0][0]
        idx = called_args.index("--beta-coeffs")
        beta_str = called_args[idx + 1]

        # Should be 7 comma-separated values
        assert beta_str.count(',') == 6
        parts = beta_str.split(',')
        assert len(parts) == 7

        # Verify first beta coefficient (prefill roofline ≈ 0.2)
        assert parts[0].startswith("0.2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_blis_adapters.py::TestBLISEvolvedCLIArgs -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'experiment.adapters.blis_evolved'"

---

### Task 3: Implement BLISEvolvedAdapter

**Files:**
- Create: `experiment/adapters/blis_evolved.py`

- [ ] **Step 1: Create adapter file with iter16 coefficients**

Create `experiment/adapters/blis_evolved.py`:

```python
"""BLIS evolved adapter — physics-informed latency with learned corrections."""

from __future__ import annotations

import os
import subprocess
import tempfile

from experiment.adapters.base import BaseBLISAdapter
from experiment.data_model import Experiment, SimulatorResult


class BLISEvolvedAdapter(BaseBLISAdapter):
    """BLIS simulator with ``--latency-model evolved``.

    Uses iter16 optimized coefficients (60.19% MAPE on 15 experiments).
    Supports MoE (Scout, Mixtral), GQA (Llama-3.1), and FP8 quantization.

    Architecture: 7-term trained-roofline formula with dataset-specific
    enhancements for MoE/FP8.
    """

    # Iter16 optimized coefficients (2026-04-01T13:42:00)
    # Training: 15 experiments, 1705 trials, 60.19% overall MAPE
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

    @staticmethod
    def _format_coeffs(coeffs: list[float]) -> str:
        """Format coefficients as comma-separated string with 6 decimal places.

        Args:
            coeffs: List of coefficient values

        Returns:
            Comma-separated string (e.g., "15569.495450,815.055650,45.705744")
        """
        return ','.join(f"{c:.6f}" for c in coeffs)

    def run(self, experiment: Experiment) -> SimulatorResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "workload_spec.yaml")
            self._write_workload_spec(experiment, spec_path)

            results_path = os.path.join(tmpdir, "results.json")
            args = self._build_common_args(experiment, spec_path, results_path)

            # Add evolved-specific flags
            args.extend(["--latency-model", "evolved"])
            args.extend(["--alpha-coeffs", self._format_coeffs(self.ITER16_ALPHA)])
            args.extend(["--beta-coeffs", self._format_coeffs(self.ITER16_BETA)])

            try:
                subprocess.run(args, capture_output=True, check=True, cwd=self._blis_dir)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"BLIS evolved failed (rc={exc.returncode}) for "
                    f"{experiment.model}: {stderr}"
                ) from exc

            return self._parse_blis_results(results_path, experiment)
```

- [ ] **Step 2: Run coefficient formatting tests**

Run: `pytest tests/test_blis_adapters.py::TestBLISEvolvedCoefficients -v`

Expected: PASS (all 3 tests)

- [ ] **Step 3: Run CLI argument tests**

Run: `pytest tests/test_blis_adapters.py::TestBLISEvolvedCLIArgs -v`

Expected: PASS (all 3 tests)

- [ ] **Step 4: Commit adapter implementation**

```bash
git add experiment/adapters/blis_evolved.py tests/test_blis_adapters.py
git commit -m "feat: add BLIS evolved latency adapter with iter16 coefficients

Implement adapter for inference-sim evolved backend (60.19% MAPE).
Uses static iter16 coefficients: 3 alpha + 7 beta.

- Coefficient formatting helper (_format_coeffs) with 6 decimal precision
- CLI args: --latency-model evolved, --alpha-coeffs, --beta-coeffs
- Extends BaseBLISAdapter for common functionality

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Adapter Name and Error Handling Tests

**Files:**
- Test: `tests/test_blis_adapters.py:116-117,453-464`

- [ ] **Step 1: Add adapter name test**

Add to `tests/test_blis_adapters.py` in `TestAdapterNames` class after line 115:

```python
    def test_evolved_name(self):
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter
        adapter = BLISEvolvedAdapter("/tmp/blis")
        assert adapter.name == "blis-evolved"
```

- [ ] **Step 2: Add error handling test**

Add to `tests/test_blis_adapters.py` after line 453 (in `TestBLISSubprocessErrors` class):

```python
    @patch("experiment.adapters.blis_evolved.subprocess.run")
    def test_evolved_wraps_subprocess_error(self, mock_run):
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["blis"], stderr=b"evolved backend not compiled"
        )
        adapter = BLISEvolvedAdapter("/tmp/blis")
        exp = _make_experiment()
        with pytest.raises(RuntimeError, match="BLIS evolved failed.*evolved backend not compiled"):
            adapter.run(exp)
```

- [ ] **Step 3: Run new tests**

Run: `pytest tests/test_blis_adapters.py::TestAdapterNames::test_evolved_name tests/test_blis_adapters.py::TestBLISSubprocessErrors::test_evolved_wraps_subprocess_error -v`

Expected: PASS (both tests)

- [ ] **Step 4: Commit tests**

```bash
git add tests/test_blis_adapters.py
git commit -m "test: add evolved adapter name and error handling tests

- Verify adapter.name returns 'blis-evolved'
- Verify subprocess errors are wrapped with context

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 5: can_run() Test

**Files:**
- Test: `tests/test_blis_adapters.py:219-222`

- [ ] **Step 1: Add can_run test for evolved adapter**

Add to `tests/test_blis_adapters.py` after line 218:

```python
class TestEvolvedCanRun:
    def test_always_true(self):
        """Evolved adapter works for any model (iter16 coefficients are cross-model)."""
        from experiment.adapters.blis_evolved import BLISEvolvedAdapter

        adapter = BLISEvolvedAdapter("/tmp/blis")
        exp = _make_experiment()
        assert adapter.can_run(exp) is True
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_blis_adapters.py::TestEvolvedCanRun -v`

Expected: PASS (evolved adapter inherits `can_run()` from `BaseBLISAdapter` which returns True)

- [ ] **Step 3: Commit test**

```bash
git add tests/test_blis_adapters.py
git commit -m "test: add evolved adapter can_run test

Verify evolved adapter accepts any experiment (cross-model coefficients).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 6: Update __init__.py Export

**Files:**
- Modify: `experiment/adapters/__init__.py:1-10`

- [ ] **Step 1: Add evolved adapter import and export**

Modify `experiment/adapters/__init__.py`:

```python
from experiment.adapters.aiconfigurator_est import AIConfiguratorEstimateAdapter
from experiment.adapters.base import BaseBLISAdapter, SimulatorAdapter
from experiment.adapters.blis_evolved import BLISEvolvedAdapter
from experiment.adapters.llmservingsim import LLMServingSimAdapter

__all__ = [
    "SimulatorAdapter",
    "BaseBLISAdapter",
    "AIConfiguratorEstimateAdapter",
    "BLISEvolvedAdapter",
    "LLMServingSimAdapter",
]
```

- [ ] **Step 2: Test import**

Run: `python -c "from experiment.adapters import BLISEvolvedAdapter; print(BLISEvolvedAdapter.__name__)"`

Expected: Output "BLISEvolvedAdapter"

- [ ] **Step 3: Run all adapter tests**

Run: `pytest tests/test_blis_adapters.py -v`

Expected: PASS (all tests including evolved adapter tests)

- [ ] **Step 4: Commit export**

```bash
git add experiment/adapters/__init__.py
git commit -m "feat: export BLISEvolvedAdapter from adapters module

Make evolved adapter available for experiment runner.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Manual Validation with BLIS Binary

**Files:**
- Test manually with actual BLIS binary

- [ ] **Step 1: Verify BLIS binary has evolved backend**

Run: `/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/blis run --help 2>&1 | grep -A 5 "latency-model"`

Expected: Output should list "evolved" as a valid latency model option

If not found, check: `grep -n "evolved" /Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/sim/bundle.go`

Expected: Should show evolved in validLatencyBackends map

- [ ] **Step 2: Test adapter with single experiment**

Create test script `test_evolved_adapter.py`:

```python
#!/usr/bin/env python3
"""Quick validation of evolved adapter with actual BLIS binary."""

from experiment.adapters.blis_evolved import BLISEvolvedAdapter
from experiment.data_model import Experiment, StageMetrics, LatencyDistribution, ThroughputMetrics

def _zero_lat():
    return LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)

def _zero_tp():
    return ThroughputMetrics(0.0, 0.0, 0.0)

# Create minimal experiment
exp = Experiment(
    folder="/tmp/test-evolved",
    model="meta-llama/Llama-3.1-8b",
    tp=1,
    hardware="H100",
    workload="general",
    max_model_len=4096,
    max_num_batched_tokens=2048,
    max_num_seqs=128,
    total_kv_blocks=7463,
    cpu_kv_blocks=5,
    stages=[
        StageMetrics(
            stage_index=0, rate=1.0, duration=60.0, num_requests=60,
            e2e=_zero_lat(), ttft=_zero_lat(), itl=_zero_lat(), throughput=_zero_tp(),
        ),
    ],
    summary=StageMetrics(
        stage_index=-1, rate=0.0, duration=0.0, num_requests=60,
        e2e=_zero_lat(), ttft=_zero_lat(), itl=_zero_lat(), throughput=_zero_tp(),
    ),
    profile_config={
        "load": {"stages": [{"duration": 60, "rate": 1}]},
        "data": {
            "shared_prefix": {
                "num_unique_system_prompts": 1,
                "num_users_per_system_prompt": 1,
                "system_prompt_len": 0,
                "question_len": 512,
                "output_len": 512,
            },
        },
    },
)

# Run adapter
adapter = BLISEvolvedAdapter("/Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim/blis")

try:
    result = adapter.run(exp)
    print(f"✅ Success! Adapter ran without errors.")
    print(f"   Completed requests: {result.summary.num_requests}")
    print(f"   E2E mean: {result.summary.e2e.mean:.2f}ms")
    print(f"   TTFT mean: {result.summary.ttft.mean:.2f}ms")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
```

Run: `python test_evolved_adapter.py`

Expected:
- Output shows "✅ Success!"
- No RuntimeError about evolved backend
- Metrics are non-zero and reasonable (E2E > 0, TTFT > 0)

- [ ] **Step 3: Clean up test script**

Run: `rm test_evolved_adapter.py`

- [ ] **Step 4: Document validation**

Add comment to commit message noting manual validation passed.

---

### Task 8: Update README Documentation

**Files:**
- Modify: `experiment/adapters/README.md:43-end`

- [ ] **Step 1: Add evolved adapter documentation**

Add to `experiment/adapters/README.md` after the LLMServingSim section (around line 43):

```markdown
## BLIS Evolved Adapter

### Overview

The BLIS evolved adapter validates inference-sim's evolved latency backend against real vLLM experiments. The evolved backend uses physics-informed latency modeling with learned correction coefficients (iter16: 60.19% MAPE).

### Architecture

- **7-term step-time formula**: Roofline basis functions (prefill/decode compute, KV cache, weight loading, TP communication, per-layer/request/step overhead)
- **3 alpha coefficients**: Request-level overhead (QueueingTime, PostDecodeFixedOverhead, OutputTokenProcessingTime)
- **7 beta coefficients**: Roofline corrections (dimensionless multipliers + µs overhead terms)
- **Dataset-specific enhancements**: MoE/dense layer split, FP8-aware weight precision, proper per-request overhead placement

### Supported Configurations

- **Hardware**: H100, A100-80GB, L40S
- **Models**: Scout (interleaved MoE/FP8), Llama-2-7b, Llama-3.1-70B, Qwen2.5-7B, Yi-34B, Mistral-Nemo-12B
- **Precision**: FP16, FP8 (FP8-aware weight loading)
- **Features**: TP, MoE, GQA

### Usage

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters blis-evolved \
  --blis-binary inference-sim/blis
```

### Requirements

- BLIS binary compiled with evolved backend (`sim/latency/evolved_model.go`)
- Ground-truth data with `per_request_lifecycle_metrics.json`

### How It Works

1. **Coefficient Selection**: Uses static iter16 coefficients (trained on 15 experiments)
2. **CLI Arguments**: Passes `--latency-model evolved --alpha-coeffs a0,a1,a2 --beta-coeffs b0,...,b6`
3. **Workload Generation**: Generates BLIS-compatible workload spec from experiment profile_config
4. **Execution**: Runs BLIS binary with evolved backend
5. **Parsing**: Converts JSON output to SimulatorResult with per-stage metrics

### Expected Accuracy

- **Overall**: 60.19% MAPE (cross-model average)
- **Best case**: 8% MAPE (Qwen2.5-7B reasoning-lite)
- **Worst case**: 130% MAPE (Scout reasoning-lite — high variance workload)
- **Typical**: 20-50% MAPE for most experiments

### Troubleshooting

**Error: "unknown latency model: evolved"**
- BLIS binary not compiled with evolved backend
- Check: `grep "evolved" inference-sim/sim/bundle.go` should show in validLatencyBackends

**Error: "AlphaCoeffs requires at least 3 elements"**
- Coefficient formatting issue (check `_format_coeffs()`)
- Verify comma-separated format with no spaces

**High MAPE (>100%)**
- Evolved backend optimized for iter16 training set
- Some workloads (reasoning-lite) have high inherent variance
- Consider falling back to roofline for out-of-distribution experiments
```

- [ ] **Step 2: Run all tests to ensure nothing broken**

Run: `pytest tests/test_blis_adapters.py -v`

Expected: PASS (all tests)

- [ ] **Step 3: Commit README update**

```bash
git add experiment/adapters/README.md
git commit -m "docs: document BLIS evolved adapter in README

Add comprehensive documentation:
- Architecture (7-term formula, iter16 coefficients)
- Supported configurations (models, hardware, features)
- Usage examples and requirements
- Expected accuracy ranges
- Troubleshooting guide

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ Adapter class structure (Task 3)
- ✅ Coefficient management (Task 3, iter16 constants)
- ✅ CLI argument construction (Task 2-3, --latency-model, --alpha-coeffs, --beta-coeffs)
- ✅ Result parsing (Task 3, inherited from BaseBLISAdapter)
- ✅ Error handling (Task 4)
- ✅ Export from __init__.py (Task 6)
- ✅ Manual validation (Task 7)
- ✅ Documentation (Task 8)

**2. Placeholder scan:**
- ✅ No TBD/TODO markers
- ✅ All code blocks are complete
- ✅ All test assertions are specific
- ✅ All commands have expected output
- ✅ All file paths are absolute or relative to project root

**3. Type consistency:**
- ✅ `_format_coeffs()` signature consistent across all uses
- ✅ `ITER16_ALPHA` and `ITER16_BETA` are `list[float]`
- ✅ Return type `SimulatorResult` matches base class
- ✅ CLI args are `list[str]` (extends base class args)

**4. CLI argument correctness (user emphasis):**
- ✅ `--latency-model evolved` (exact backend name)
- ✅ `--alpha-coeffs` followed by comma-separated string (no spaces)
- ✅ `--beta-coeffs` followed by comma-separated string (no spaces)
- ✅ 6 decimal precision matches iter16 JSON format
- ✅ Coefficients passed after common args (order matters for subprocess)

---

## Execution Summary

**Total tasks:** 8
**Total steps:** 28
**Estimated time:** 45-60 minutes

**Critical path:**
1. Task 1-2: Write tests (10 min)
2. Task 3: Implement adapter (15 min)
3. Task 4-6: Additional tests + export (10 min)
4. Task 7: Manual validation (10 min)
5. Task 8: Documentation (10 min)

**Key validation points:**
- After Task 3 Step 2: Coefficient formatting works
- After Task 3 Step 3: CLI args constructed correctly
- After Task 7 Step 2: Adapter runs with real BLIS binary
- After Task 8 Step 2: All tests still pass
