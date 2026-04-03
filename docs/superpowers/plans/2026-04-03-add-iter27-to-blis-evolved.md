# Add Iter27 to BLIS Evolved Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add iter27 coefficients to BLIS evolved adapter while keeping iter26 as default

**Architecture:** Add ITER27_ALPHA and ITER27_BETA class constants, update constructor validation to accept 27, add iter27 handling in run() method, and update documentation. Iter27 achieves 34.61% loss through CMA-ES joint optimization.

**Tech Stack:** Python, pytest, YAML

---

## Task 1: Add Test for Iter27 Instantiation

**Files:**
- Modify: `tests/test_blis_adapters.py:117-121`

- [ ] **Step 1: Write failing test for iter27 instantiation**

Add this test after the existing `test_evolved_name` test in the `TestAdapterNames` class:

```python
def test_evolved_name_iter27(self):
    """Iter27 adapter should still report name as blis-evolved."""
    from experiment.adapters.blis_evolved import BLISEvolvedAdapter
    adapter = BLISEvolvedAdapter("/tmp/blis", iteration=27)
    assert adapter.name == "blis-evolved"
    assert adapter.iteration == 27
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_blis_adapters.py::TestAdapterNames::test_evolved_name_iter27 -v`

Expected: FAIL with "ValueError: iteration must be 16, 24, or 26, got 27"

- [ ] **Step 3: Add test for invalid iteration value**

Add this test after the previous one:

```python
def test_evolved_rejects_invalid_iteration(self):
    """Evolved adapter should reject invalid iteration values."""
    from experiment.adapters.blis_evolved import BLISEvolvedAdapter
    import pytest

    with pytest.raises(ValueError, match="iteration must be 16, 24, 26, or 27"):
        BLISEvolvedAdapter("/tmp/blis", iteration=99)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_blis_adapters.py::TestAdapterNames::test_evolved_rejects_invalid_iteration -v`

Expected: FAIL with "AssertionError: DID NOT RAISE ValueError" (because current code checks for 16, 24, or 26)

- [ ] **Step 5: Commit test additions**

```bash
git add tests/test_blis_adapters.py
git commit -m "test: add iter27 instantiation tests for blis-evolved

Add tests for iter27 parameter and invalid iteration rejection.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Iter27 Coefficient Constants

**Files:**
- Modify: `experiment/adapters/blis_evolved.py:104-122`

- [ ] **Step 1: Add ITER27_ALPHA constant**

After the ITER26_BETA constant (line 122), add:

```python
# Iter27 optimised coefficients (10 betas with CMA-ES joint optimization)
ITER27_ALPHA: list[float] = [
    15563.199579,        # α₀: QueueingTime (~15.6ms fixed API overhead)
    777.3455,            # α₁: PostDecodeFixedOverhead (~0.8ms per-request)
    45.907545,           # α₂: OutputTokenProcessingTime (µs/token streaming)
]
```

- [ ] **Step 2: Add ITER27_BETA constant**

After ITER27_ALPHA, add:

```python
ITER27_BETA: list[float] = [
    0.152128,            # β₁ₐ: Prefill compute (7.2× FlashAttention, +9% vs iter26)
    0.000721,            # β₂ₐ: Decode compute (near-zero — memory-bound)
    1.363621,            # β₃: Weight loading (36% overhead, stable)
    0.752037,            # β₄: TP All-Reduce (+83% vs iter26 — better comm capture)
    32.394131,           # β₅: Per-layer overhead (µs/layer, -35% as β₄ absorbed)
    2.805128,            # β₆: Per-request scheduling (µs/req)
    126.024825,          # β₇: Per-step constant (µs/step, -26% reduction)
    505.508488,          # β₈: Per-MoE-layer overhead (µs/MoE-layer, +18%)
    0.000746,            # β₁ᵦ: Prefill memory (near-zero — compute-bound)
    1.922366,            # β₂ᵦ: Decode memory (+52% vs iter26 — stronger correction)
]
```

- [ ] **Step 3: Verify coefficient precision**

Double-check that the values exactly match `inference-sim/training/iterations/iter27/inner_loop_results.json`:
- Alpha: [15563.199579, 777.3455, 45.907545]
- Beta: [0.152128, 0.000721, 1.363621, 0.752037, 32.394131, 2.805128, 126.024825, 505.508488, 0.000746, 1.922366]

- [ ] **Step 4: Commit coefficient constants**

```bash
git add experiment/adapters/blis_evolved.py
git commit -m "feat: add iter27 coefficient constants to blis-evolved

Add ITER27_ALPHA and ITER27_BETA with exact values from CMA-ES training.
Iter27 achieves 34.61% loss (vs 37.42% for iter26).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update Constructor to Accept Iter27

**Files:**
- Modify: `experiment/adapters/blis_evolved.py:124-137`

- [ ] **Step 1: Update constructor validation**

Find the line (currently 136):
```python
if iteration not in (16, 24, 26):
    raise ValueError(f"iteration must be 16, 24, or 26, got {iteration}")
```

Replace with:
```python
if iteration not in (16, 24, 26, 27):
    raise ValueError(f"iteration must be 16, 24, 26, or 27, got {iteration}")
```

- [ ] **Step 2: Update constructor docstring**

Find the docstring section (around line 132):
```python
iteration : int, default=26
    Which iteration coefficients to use (16, 24, or 26)
```

Replace with:
```python
iteration : int, default=26
    Which iteration coefficients to use (16, 24, 26, or 27)
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_blis_adapters.py::TestAdapterNames::test_evolved_name_iter27 tests/test_blis_adapters.py::TestAdapterNames::test_evolved_rejects_invalid_iteration -v`

Expected: PASS (both tests pass now)

- [ ] **Step 4: Commit constructor changes**

```bash
git add experiment/adapters/blis_evolved.py
git commit -m "feat: update constructor to accept iteration=27

Update validation and docstring to support iter27 parameter.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add Iter27 Handling in run() Method

**Files:**
- Modify: `experiment/adapters/blis_evolved.py:152-162`

- [ ] **Step 1: Write failing test for iter27 CLI args**

Add this test to the `TestBLISEvolvedCLIArgs` class in `tests/test_blis_adapters.py`:

```python
@patch("experiment.adapters.blis_evolved.subprocess.run")
def test_evolved_iter27_coefficients(self, mock_run):
    """Iter27 adapter should pass iter27-specific coefficients."""
    from experiment.adapters.blis_evolved import BLISEvolvedAdapter

    mock_run.return_value = MagicMock()
    adapter = BLISEvolvedAdapter("/usr/local/bin/blis", iteration=27)
    exp = _make_experiment()

    with patch.object(adapter, "_parse_blis_results") as mock_parse:
        mock_parse.return_value = MagicMock()
        adapter.run(exp)

    called_args = mock_run.call_args[0][0]

    # Check alpha coefficients
    idx = called_args.index("--alpha-coeffs")
    alpha_str = called_args[idx + 1]
    assert alpha_str.startswith("15563.199")

    # Check beta coefficients (should have 10 values)
    idx = called_args.index("--beta-coeffs")
    beta_str = called_args[idx + 1]
    assert beta_str.count(",") == 9
    parts = beta_str.split(",")

    # Verify key iter27 coefficients
    assert parts[0].startswith("0.152")  # β₁ₐ
    assert parts[3].startswith("0.752")  # β₄ (TP All-Reduce, +83%)
    assert parts[4].startswith("32.39")  # β₅ (per-layer, -35%)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_blis_adapters.py::TestBLISEvolvedCLIArgs::test_evolved_iter27_coefficients -v`

Expected: FAIL (no iter27 handling in run() method yet)

- [ ] **Step 3: Add iter27 case to run() method**

In the `run()` method, find the coefficient selection section (lines 154-162):

```python
# Select coefficients based on iteration
if self.iteration == 16:
    alpha = self.ITER16_ALPHA
    beta = self.ITER16_BETA
elif self.iteration == 24:
    alpha = self.ITER24_ALPHA
    beta = self.ITER24_BETA
else:  # iteration == 26
    alpha = self.ITER26_ALPHA
    beta = self.ITER26_BETA
```

Replace with:

```python
# Select coefficients based on iteration
if self.iteration == 16:
    alpha = self.ITER16_ALPHA
    beta = self.ITER16_BETA
elif self.iteration == 24:
    alpha = self.ITER24_ALPHA
    beta = self.ITER24_BETA
elif self.iteration == 26:
    alpha = self.ITER26_ALPHA
    beta = self.ITER26_BETA
else:  # iteration == 27
    alpha = self.ITER27_ALPHA
    beta = self.ITER27_BETA
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_blis_adapters.py::TestBLISEvolvedCLIArgs::test_evolved_iter27_coefficients -v`

Expected: PASS

- [ ] **Step 5: Run all evolved adapter tests**

Run: `pytest tests/test_blis_adapters.py -k "Evolved" -v`

Expected: All tests pass

- [ ] **Step 6: Commit run() method changes**

```bash
git add experiment/adapters/blis_evolved.py tests/test_blis_adapters.py
git commit -m "feat: add iter27 coefficient selection in run() method

Add iter27 case to coefficient selection logic with CLI args test.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update Class Docstring with Iter27 Summary

**Files:**
- Modify: `experiment/adapters/blis_evolved.py:18-65`

- [ ] **Step 1: Add iter27 training summary section**

After the "Iter26 training summary" section (after line 45), add:

```python
Iter27 training summary
-----------------------
* Dataset  : 15 experiments (H100 / FP16)
* Best MAPE: 34.61 % (overall loss: TTFT=22.81%, E2E=11.79%)
* Method   : CMA-ES joint optimization of 6 parameters (β₁ₐ,β₄,β₅,β₇,β₈,β₂ᵦ)
* Trials   : 141 total, best at trial 62
* Betas    : 10 (same architecture as iter24/iter26)
* Key finding: Strong coefficient interactions - β₄ increased 83% to 0.752, allowing β₅/β₇ to decrease
* Improvement: 37.42% → 34.61% (-2.81 points, -7.5% relative)
* Notable improvements: Mistral TP=2 TTFT 27.4%→20.0%, Llama-3.1 TP=4 E2E 9.4%→3.2%
```

- [ ] **Step 2: Update coefficient semantics table**

In the "Coefficient semantics (iter24/iter26)" section (line 47), update the header to:

```python
Coefficient semantics (iter24/iter26/iter27)
--------------------------------------------
```

- [ ] **Step 3: Update beta coefficient descriptions**

Update the beta descriptions to include iter27 values where they changed significantly:

```python
Beta (10 values):
    β₁ₐ : Prefill compute correction (iter24/26: 0.139, iter27: 0.152 — +9% refinement)
    β₂ₐ : Decode compute correction (0.0 — dropped, decode is memory-bound)
    β₃  : Weight loading correction (1.363 — 36% overhead above roofline)
    β₄  : TP communication correction (iter24: 0.396, iter26: 0.410, iter27: 0.752 — +83% in iter27)
    β₅  : Per-layer overhead (iter24: 62.3, iter26: 49.6, iter27: 32.4 µs/layer — β₄ absorbed overhead)
    β₆  : Per-request scheduling (2.8 µs/req)
    β₇  : Per-step constant (iter24/26: 169.4, iter27: 126.0 µs/step — -26% reduction)
    β₈  : Per-MoE-layer overhead (iter24/26: 427.3, iter27: 505.5 µs/MoE-layer — +18% increase)
    β₁ᵦ : Prefill memory correction (0.0 — dropped, prefill is compute-bound)
    β₂ᵦ : Decode memory correction (iter24/26: 1.263, iter27: 1.922 — +52% stronger correction)
```

- [ ] **Step 4: Read the updated docstring**

Run: `python -c "from experiment.adapters.blis_evolved import BLISEvolvedAdapter; print(BLISEvolvedAdapter.__doc__[:1000])"`

Expected: Docstring should include iter27 training summary

- [ ] **Step 5: Commit docstring updates**

```bash
git add experiment/adapters/blis_evolved.py
git commit -m "docs: update blis-evolved docstring with iter27 summary

Add iter27 training summary and update coefficient semantics table.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update README with Iter27 Documentation

**Files:**
- Modify: `experiment/adapters/README.md:44-167`

- [ ] **Step 1: Update overview section**

Find the line (around line 48):
```
The latest iteration (iter26) achieves **37.42% overall MAPE**
```

Replace with:
```
The latest iteration (iter27) achieves **34.61% overall MAPE** (TTFT: 22.81%, E2E: 11.79%) across 15 experiments on H100/FP16, with CMA-ES joint optimization discovering strong coefficient interactions.
```

- [ ] **Step 2: Add iter27 column to beta coefficients table**

Find the beta coefficients table (around line 64). Update the header row:

```markdown
| Index | Name | iter24 | iter26 | iter27 | Semantics |
|-------|------|--------|--------|--------|-----------|
```

Then update the coefficient rows:

```markdown
| `β₁ₐ` | Prefill compute correction | 0.139 | 0.139 | 0.152 | FlashAttention reduces effective FLOPs by 7.2× |
| `β₂ₐ` | Decode compute correction | **0.0** | **0.0** | **0.0** | **Dropped — decode is memory-bound** |
| `β₃` | Weight loading correction | 1.363 | 1.363 | 1.364 | 36% overhead above roofline weight bandwidth |
| `β₄` | TP communication correction | 0.396 | **0.410** | **0.752** | **Iter27: +83% vs iter26, captures comm better** |
| `β₅` | Per-layer overhead | 62.3 µs | **49.6 µs** | **32.4 µs** | **Iter27: β₄ absorbed overhead** |
| `β₆` | Per-request scheduling | 2.8 µs/req | 2.8 µs/req | 2.8 µs/req | Per-request scheduling in batch |
| `β₇` | Per-step constant | 169.4 µs/step | 169.4 µs/step | **126.0 µs/step** | **Iter27: -26% reduction** |
| `β₈` | Per-MoE-layer overhead | 427.3 µs | 427.3 µs | **505.5 µs** | **Iter27: +18% increase** |
| `β₁ᵦ` | Prefill memory correction | **0.0** | **0.0** | **0.0** | **Dropped — prefill is compute-bound** |
| `β₂ᵦ` | Decode memory correction | 1.263 | 1.263 | **1.922** | **Iter27: +52% stronger correction** |
```

- [ ] **Step 3: Update key insights section**

After the table, update the "Key insights" section:

```markdown
**Key insights:**
- **Iter24**: Clean physical split — prefill uses only compute (β₁ₐ), decode uses only memory (β₂ᵦ)
- **Iter26**: TP All-Reduce activated (β₄: 0.396 → 0.410), per-layer overhead decreased (β₅: 62.3 → 49.6 µs)
- **Iter27**: CMA-ES joint optimization — β₄ increased 83% to 0.752, allowing β₅ (-35%) and β₇ (-26%) to decrease
```

- [ ] **Step 4: Add iter27 usage example**

Find the usage section (around line 92). After the iter24 example, add:

```bash
# Using iter27 coefficients (10-beta BLIS with CMA-ES optimization, 34.61% loss)
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters blis-evolved \
  --blis-binary /path/to/blis \
  --blis-evolved-iteration 27
```

- [ ] **Step 5: Update iteration requirements section**

Find the "Iteration requirements" section (around line 126). Add iter27:

```markdown
**Iteration requirements:**
- **Iter16** (7 betas): Requires BLIS that supports 7-beta mode (60.19% MAPE)
- **Iter24** (10 betas): Requires BLIS with decode-split support (10-beta mode, 39.18% MAPE)
- **Iter26** (10 betas): Requires BLIS with TP All-Reduce support (10-beta mode, 37.42% MAPE)
- **Iter27** (10 betas): Requires BLIS with TP All-Reduce support (10-beta mode, 34.61% MAPE)
```

- [ ] **Step 6: Update expected accuracy section**

Find the "Expected Accuracy" section (around line 142). Update to reflect iter27:

```markdown
### Expected Accuracy

Based on iter27 training results across 15 H100/FP16 experiments:

- **Overall MAPE**: 34.61% (TTFT: 22.81%, E2E: 11.79%)
- **Improvement over iter26**: -2.81 points (-7.5% relative)
- **Best case**: Llama-3.1-8B TP=4 E2E (~3.2%), Yi general-lite TTFT (~2.7%)
- **Worst case**: Scout reasoning-lite (TTFT: ~59%, E2E: ~11% — long 934-token prefill)
- **Typical**: 12 of 15 experiments below 30% TTFT MAPE

**Training journey** (iter16 → iter27):
- Iter16: 60.19% MAPE (trained-roofline architecture)
- Iter20: 40.58% MAPE (β₈·nMoELayers breakthrough — 19.5pt improvement)
- Iter21: 39.86% MAPE (prefill compute-only split)
- Iter24: 39.18% MAPE (decode memory-only split)
- Iter26: 37.42% MAPE (TP All-Reduce activation)
- Iter27: 34.61% MAPE (CMA-ES joint 6-parameter optimization)

Total improvement: **42.5% relative reduction** (60.19% → 34.61%)
```

- [ ] **Step 7: Commit README updates**

```bash
git add experiment/adapters/README.md
git commit -m "docs: update README with iter27 documentation

Add iter27 to coefficients table, usage examples, and accuracy metrics.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Final Verification

**Files:**
- N/A (testing only)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/test_blis_adapters.py -v`

Expected: All tests pass

- [ ] **Step 2: Test iter27 instantiation interactively**

Run:
```bash
python -c "
from experiment.adapters.blis_evolved import BLISEvolvedAdapter
adapter = BLISEvolvedAdapter('/tmp/blis', iteration=27)
print(f'Adapter name: {adapter.name}')
print(f'Iteration: {adapter.iteration}')
print(f'Alpha coeffs: {len(adapter.ITER27_ALPHA)} values')
print(f'Beta coeffs: {len(adapter.ITER27_BETA)} values')
print(f'First alpha: {adapter.ITER27_ALPHA[0]}')
print(f'TP coeff β₄: {adapter.ITER27_BETA[3]}')
"
```

Expected output:
```
Adapter name: blis-evolved
Iteration: 27
Alpha coeffs: 3 values
Beta coeffs: 10 values
First alpha: 15563.199579
TP coeff β₄: 0.752037
```

- [ ] **Step 3: Verify coefficient formatting**

Run:
```bash
python -c "
from experiment.adapters.blis_evolved import BLISEvolvedAdapter
adapter = BLISEvolvedAdapter('/tmp/blis', iteration=27)
print('Formatted alpha:', adapter._format_coeffs(adapter.ITER27_ALPHA))
print('Formatted beta:', adapter._format_coeffs(adapter.ITER27_BETA))
"
```

Expected: Both should be comma-separated strings with 6 decimal places

- [ ] **Step 4: Test default iteration is still 26**

Run:
```bash
python -c "
from experiment.adapters.blis_evolved import BLISEvolvedAdapter
adapter = BLISEvolvedAdapter('/tmp/blis')
print(f'Default iteration: {adapter.iteration}')
"
```

Expected output:
```
Default iteration: 26
```

- [ ] **Step 5: Create final summary commit**

```bash
git add -A
git commit -m "feat: complete iter27 integration for blis-evolved adapter

Summary of changes:
- Added ITER27_ALPHA and ITER27_BETA constants with exact CMA-ES values
- Updated constructor to accept iteration=27 (default remains 26)
- Added iter27 case to coefficient selection in run() method
- Updated docstring with iter27 training summary
- Updated README with iter27 documentation and usage examples
- Added comprehensive tests for iter27 functionality

Iter27 achieves 34.61% loss (TTFT: 22.81%, E2E: 11.79%), a 7.5%
relative improvement over iter26 through joint optimization.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Add ITER27_ALPHA constant with exact values
- ✅ Add ITER27_BETA constant with exact values
- ✅ Update constructor to accept iteration=27
- ✅ Keep iteration=26 as default for backward compatibility
- ✅ Update run() method to select iter27 coefficients
- ✅ Update class docstring with iter27 training summary
- ✅ Update README.md with iter27 documentation
- ✅ Add tests for iter27 functionality

**Placeholder scan:**
- ✅ No "TBD", "TODO", or "implement later"
- ✅ All code blocks contain actual implementation
- ✅ All coefficients are exact values from training
- ✅ All test assertions are specific

**Type consistency:**
- ✅ ITER27_ALPHA is `list[float]` matching ITER16/24/26
- ✅ ITER27_BETA is `list[float]` matching ITER16/24/26
- ✅ iteration parameter remains `int`
- ✅ All method signatures unchanged

**Coefficient accuracy verification:**
- ✅ Alpha: [15563.199579, 777.3455, 45.907545] from inner_loop_results.json
- ✅ Beta: [0.152128, 0.000721, 1.363621, 0.752037, 32.394131, 2.805128, 126.024825, 505.508488, 0.000746, 1.922366] from inner_loop_results.json
- ✅ All values match training output exactly
