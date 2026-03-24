# --no-dp-scaling Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--no-dp-scaling` CLI flag to `experiment.run` to exclude experiments with data parallelism > 1

**Architecture:** Add boolean flag to argparse, thread through `run_pipeline()`, filter experiments after parsing but before execution. Treat `dp: null` as equivalent to `dp: 1`.

**Tech Stack:** Python 3.10+, pytest, argparse

**Spec:** `docs/superpowers/specs/2026-03-20-no-dp-scaling-flag-design.md`

---

## File Structure

**Modified files:**
- `experiment/run.py` - Add flag argument, update `run_pipeline()` signature, add filtering logic
- `tests/test_run.py` - Unit tests for argparse and filtering
- `README.md` - Document the new CLI flag

**No new files created.**

---

## Task 1: Test --no-dp-scaling Flag Parsing

**Files:**
- Modify: `tests/test_run.py:61-80` (TestParseArgs class)

- [ ] **Step 1: Write test for --no-dp-scaling flag default**

Add to `TestParseArgs` class after `test_custom_values()`:

```python
def test_no_dp_scaling_default_false(self):
    args = parse_args([])
    assert args.no_dp_scaling is False

def test_no_dp_scaling_flag_present(self):
    args = parse_args(["--no-dp-scaling"])
    assert args.no_dp_scaling is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run.py::TestParseArgs::test_no_dp_scaling_default_false tests/test_run.py::TestParseArgs::test_no_dp_scaling_flag_present -v`

Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'no_dp_scaling'`

- [ ] **Step 3: Implement the argparse flag**

In `experiment/run.py`, add to `parse_args()` function after line 180 (after `--adapters` argument):

```python
parser.add_argument(
    "--no-dp-scaling",
    action="store_true",
    help="Exclude experiments with data parallelism > 1 (multi-replica).",
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run.py::TestParseArgs::test_no_dp_scaling_default_false tests/test_run.py::TestParseArgs::test_no_dp_scaling_flag_present -v`

Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_run.py experiment/run.py
git commit -m "test: add --no-dp-scaling flag parsing

Add argparse flag for excluding multi-replica experiments.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Test DP Filtering Logic

**Files:**
- Modify: `tests/test_run.py:117-304` (TestRunPipeline class)

- [ ] **Step 1: Write test for no filtering when flag is False**

Add to `TestRunPipeline` class after `test_no_experiments_returns_empty()`:

```python
@patch("experiment.run.generate_report")
@patch("experiment.run.parse_experiment")
@patch("experiment.run.discover_experiments")
@patch("experiment.run.build_adapter_registry")
def test_no_dp_scaling_false_runs_all_experiments(
    self, mock_registry, mock_discover, mock_parse, mock_report
):
    """When no_dp_scaling=False, all experiments should run."""
    exp_dp_null = _make_experiment(folder="/tmp/exp1")
    exp_dp_null.dp = None
    exp_dp_1 = _make_experiment(folder="/tmp/exp2")
    exp_dp_1.dp = 1
    exp_dp_2 = _make_experiment(folder="/tmp/exp3")
    exp_dp_2.dp = 2

    mock_discover.return_value = [
        (_MANIFEST_STUB, "/tmp/exp1"),
        (_MANIFEST_STUB, "/tmp/exp2"),
        (_MANIFEST_STUB, "/tmp/exp3"),
    ]
    mock_parse.side_effect = lambda path, manifest_entry=None: {
        "/tmp/exp1": exp_dp_null,
        "/tmp/exp2": exp_dp_1,
        "/tmp/exp3": exp_dp_2,
    }[path]
    mock_registry.return_value = {}

    run_pipeline(
        data_dir="/data",
        blis_binary="/bin/blis",
        vidur_dir="/opt/vidur",
        output_dir="/out",
        adapter_names=[],
        no_dp_scaling=False,
    )

    # All 3 experiments should be parsed
    assert mock_parse.call_count == 3
```

- [ ] **Step 2: Write test for filtering when flag is True**

Add after the previous test:

```python
@patch("experiment.run.generate_report")
@patch("experiment.run.parse_experiment")
@patch("experiment.run.discover_experiments")
@patch("experiment.run.build_adapter_registry")
def test_no_dp_scaling_true_filters_dp_gt_1(
    self, mock_registry, mock_discover, mock_parse, mock_report
):
    """When no_dp_scaling=True, only dp<=1 experiments should run."""
    exp_dp_null = _make_experiment(folder="/tmp/exp1")
    exp_dp_null.dp = None
    exp_dp_1 = _make_experiment(folder="/tmp/exp2")
    exp_dp_1.dp = 1
    exp_dp_2 = _make_experiment(folder="/tmp/exp3")
    exp_dp_2.dp = 2
    exp_dp_4 = _make_experiment(folder="/tmp/exp4")
    exp_dp_4.dp = 4

    mock_discover.return_value = [
        (_MANIFEST_STUB, "/tmp/exp1"),
        (_MANIFEST_STUB, "/tmp/exp2"),
        (_MANIFEST_STUB, "/tmp/exp3"),
        (_MANIFEST_STUB, "/tmp/exp4"),
    ]
    mock_parse.side_effect = lambda path, manifest_entry=None: {
        "/tmp/exp1": exp_dp_null,
        "/tmp/exp2": exp_dp_1,
        "/tmp/exp3": exp_dp_2,
        "/tmp/exp4": exp_dp_4,
    }[path]

    # Mock adapter that runs on all experiments
    mock_adapter = MagicMock()
    mock_adapter.name = "mock-sim"
    mock_adapter.can_run.return_value = True
    mock_adapter.run.return_value = SimulatorResult(
        adapter_name="mock-sim",
        experiment_folder="/tmp/exp",
        stages=[_make_stage(0)],
        summary=_make_stage(-1),
    )
    mock_registry.return_value = {"mock-sim": mock_adapter}

    run_pipeline(
        data_dir="/data",
        blis_binary="/bin/blis",
        vidur_dir="/opt/vidur",
        output_dir="/out",
        adapter_names=["mock-sim"],
        no_dp_scaling=True,
    )

    # All 4 experiments should be parsed
    assert mock_parse.call_count == 4
    # But only 2 should run through adapters (dp=null and dp=1)
    assert mock_adapter.run.call_count == 2
    # Verify the filtered experiments are the ones with dp<=1
    run_folders = {call.args[0].folder for call in mock_adapter.run.call_args_list}
    assert run_folders == {"/tmp/exp1", "/tmp/exp2"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_run.py::TestRunPipeline::test_no_dp_scaling_false_runs_all_experiments tests/test_run.py::TestRunPipeline::test_no_dp_scaling_true_filters_dp_gt_1 -v`

Expected: FAIL with `TypeError: run_pipeline() got an unexpected keyword argument 'no_dp_scaling'`

- [ ] **Step 4: Implement the filtering logic**

In `experiment/run.py`:

**4a. Update `run_pipeline()` signature (line 64):**

```python
def run_pipeline(
    data_dir: str,
    blis_binary: str,
    vidur_dir: str,
    output_dir: str,
    adapter_names: list[str] | None = None,
    no_dp_scaling: bool = False,
) -> tuple[list[ErrorRecord], list[RuntimeRecord]]:
```

**4b. Add filtering logic after line 96 (after "Parsed ... experiments successfully" print):**

```python
print(f"Parsed {len(experiments)} experiments successfully")
if discovered and not experiments:
    logger.warning("All %d experiments failed to parse", len(discovered))

# Filter by DP if requested
if no_dp_scaling:
    before_count = len(experiments)
    experiments = [exp for exp in experiments
                   if exp.dp is None or exp.dp <= 1]
    filtered_count = before_count - len(experiments)
    print(f"Filtered to {len(experiments)} single-replica experiments "
          f"(excluded {filtered_count} with DP > 1)")

# 3. Build adapter registry (only requested adapters)
adapters = build_adapter_registry(blis_binary, vidur_dir, adapter_names)
```

**4c. Update `main()` to pass the flag (line 188):**

```python
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_pipeline(
        data_dir=args.data_dir,
        blis_binary=args.blis_binary,
        vidur_dir=args.vidur_dir,
        output_dir=args.output_dir,
        adapter_names=args.adapters,
        no_dp_scaling=args.no_dp_scaling,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_run.py::TestRunPipeline::test_no_dp_scaling_false_runs_all_experiments tests/test_run.py::TestRunPipeline::test_no_dp_scaling_true_filters_dp_gt_1 -v`

Expected: PASS (both tests)

- [ ] **Step 6: Run all existing tests to verify no regressions**

Run: `pytest tests/test_run.py -v`

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_run.py experiment/run.py
git commit -m "feat: implement --no-dp-scaling filtering logic

Filter experiments with DP > 1 when flag is specified. Treat dp: null
as single-replica (equivalent to dp: 1).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update README Documentation

**Files:**
- Modify: `README.md:158-166` (Pipeline CLI options table)

- [ ] **Step 1: Add flag documentation to README**

In `README.md`, find the "Pipeline CLI options" table (around line 160) and add a new row after the `--adapters` entry:

```markdown
| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `vllm_data/ground_truth` | Directory containing ground-truth experiment folders |
| `--blis-binary` | `inference-sim/blis` | Path to compiled BLIS binary |
| `--vidur-dir` | `vidur` | Path to cloned Vidur repository |
| `--output-dir` | `results` | Where reports and CSV are saved |
| `--adapters` | all 7 | Space-separated list of adapters to run |
| `--no-dp-scaling` | *(disabled)* | Exclude experiments with data parallelism > 1 |
```

- [ ] **Step 2: Verify README renders correctly**

Run: `head -n 180 README.md | tail -n 30`

Expected: Table formatting looks correct, new row is present

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document --no-dp-scaling flag in README

Add CLI flag documentation to Pipeline CLI options table.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Integration Test (Optional but Recommended)

**Files:**
- Modify: `tests/test_integration.py:54-59` (after TestGroundTruthParsing class)

- [ ] **Step 1: Write integration test for CLI with flag**

Add new test class after `TestGroundTruthParsing`:

```python
class TestNoDPScalingIntegration:
    def test_no_dp_scaling_flag_integration(self):
        """End-to-end test: --no-dp-scaling should filter experiments."""
        from experiment.run import run_pipeline

        # Run pipeline with flag (no adapters to avoid external dependencies)
        error_records, runtime_records = run_pipeline(
            data_dir=_DATA_DIR,
            blis_binary="nonexistent",  # Won't be used with no adapters
            vidur_dir="nonexistent",
            output_dir="/tmp/test_output",
            adapter_names=[],
            no_dp_scaling=True,
        )

        # Should complete without errors
        assert error_records == []
        assert runtime_records == []
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py::TestNoDPScalingIntegration::test_no_dp_scaling_flag_integration -v`

Expected: PASS (or SKIP if ground-truth data not available)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for --no-dp-scaling

Verify end-to-end behavior with real data directory structure.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Verification and Manual Testing

**Files:**
- None (manual verification)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`

Expected: All tests PASS

- [ ] **Step 2: Test CLI help output**

Run: `python -m experiment.run --help`

Expected: Output includes `--no-dp-scaling` flag with description:
```
--no-dp-scaling       Exclude experiments with data parallelism > 1 (multi-replica).
```

- [ ] **Step 3: Manual smoke test with real data (if available)**

Run:
```bash
python -m experiment.run --data-dir vllm_data/ground_truth --no-dp-scaling --output-dir /tmp/test_results
```

Expected output should include:
```
Found X experiments
Parsed Y experiments successfully
Filtered to Z single-replica experiments (excluded W with DP > 1)
```

Where Z + W = Y and Z includes only experiments with dp=null or dp<=1.

Note: This will run all adapters. To test filtering only without running simulators, add `--adapters` at the end (will fail with argparse error, confirming the flag is parsed correctly) or check the experiment count in the output.

- [ ] **Step 4: Verify backward compatibility (no flag)**

Run:
```bash
python -m experiment.run --data-dir vllm_data/ground_truth --output-dir /tmp/test_results_all
```

Expected: No filtering message, all experiments run (current behavior)

- [ ] **Step 5: Final commit message verification**

Run: `git log --oneline -5`

Expected: See all 4 commits with clear, descriptive messages

---

## Completion Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass (or skip if data unavailable)
- [ ] CLI flag is documented in README
- [ ] Manual verification confirms filtering behavior
- [ ] Backward compatibility verified (default behavior unchanged)
- [ ] All commits have clear messages with Co-Authored-By tag

---

## Expected Outcome

After completing this plan:

1. **CLI flag works:** `python -m experiment.run --no-dp-scaling` filters experiments
2. **Tests pass:** All existing tests continue to pass, new tests verify filtering
3. **Documentation updated:** README includes the new flag
4. **Backward compatible:** Without flag, all experiments run as before
5. **Clean git history:** 4 focused commits with descriptive messages

## Usage Examples

**Before (current behavior):**
```bash
python -m experiment.run --data-dir vllm_data/ground_truth
# Runs all 59 experiments (including dp=2, dp=4, etc.)
```

**After (with flag):**
```bash
python -m experiment.run --data-dir vllm_data/ground_truth --no-dp-scaling
# Runs only single-replica experiments (dp=null or dp=1)
# Output: "Filtered to 45 single-replica experiments (excluded 14 with DP > 1)"
```
