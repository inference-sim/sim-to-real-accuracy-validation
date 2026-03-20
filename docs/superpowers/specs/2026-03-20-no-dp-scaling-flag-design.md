# Design: --no-dp-scaling Flag for experiment.run

**Date:** 2026-03-20
**Status:** Approved
**Author:** Claude Sonnet 4.5

## Problem

Users need the ability to exclude experiments with data parallelism (DP) > 1 from the experiment pipeline. Currently, `experiment.run` processes all discovered experiments regardless of their DP configuration. Some analyses (e.g., comparing simulators on single-replica workloads) need to filter out multi-replica experiments.

While `experiment.figures` has hardcoded DP ≤ 1 filters for specific publication figures (Figures 2 and 3), there's no CLI control over which experiments run through the pipeline.

## Goals

1. Add a CLI flag to `experiment.run` that excludes experiments with DP > 1
2. Treat `dp: null` as equivalent to `dp: 1` (single-replica)
3. Maintain backward compatibility (flag is optional, defaults to running all experiments)
4. Keep hardcoded filters in `experiment.figures` unchanged (they serve a different analytical purpose)

## Non-Goals

- Adding the flag to `experiment.figures` (not needed — figures automatically show only what's in the CSV)
- Making the flag configurable with arbitrary DP thresholds (just a boolean for "no scaling")
- Filtering at discovery time (cleaner to filter after parsing)

## Solution

### CLI Interface

Add a new boolean flag `--no-dp-scaling` to `experiment.run`:

```python
parser.add_argument(
    "--no-dp-scaling",
    action="store_true",
    help="Exclude experiments with data parallelism > 1 (multi-replica).",
)
```

**Flag behavior:**
- **Not specified** (default): Run all experiments regardless of DP
- **Specified**: Run only experiments where `dp` is `None` or `<= 1`

### Implementation

**File:** `experiment/run.py`

**1. Update `run_pipeline()` signature (line 64):**

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

**2. Add filtering logic after parsing (after line 94):**

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

**3. Update `main()` to pass the flag (line 185):**

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

### Filtering Logic Details

**DP Value Handling:**

| DP Value | Interpretation | With `--no-dp-scaling` |
|----------|---------------|----------------------|
| `null` | No DP configured (single replica) | **Included** |
| `1` | Explicit single replica | **Included** |
| `2`, `4`, etc. | Multi-replica | **Excluded** |

**Filter Expression:**

```python
exp.dp is None or exp.dp <= 1
```

This keeps experiments where:
- `dp` is `None` (null in experiments.json)
- `dp` is 1 (single replica)

And excludes only `dp > 1` (multi-replica).

### Example Usage

**Run all experiments (current behavior):**
```bash
python -m experiment.run --data-dir vllm_data/ground_truth
```

**Run only single-replica experiments:**
```bash
python -m experiment.run --data-dir vllm_data/ground_truth --no-dp-scaling
```

**Expected output with flag:**
```
Found 59 experiments
Parsed 59 experiments successfully
Filtered to 45 single-replica experiments (excluded 14 with DP > 1)
```

## Testing

### Unit Tests

**File:** `tests/test_run.py`

1. **Test filter logic:**
   - Create mock experiments with `dp: null`, `dp: 1`, `dp: 2`, `dp: 4`
   - Verify filtering keeps null and 1, excludes 2 and 4

2. **Test backward compatibility:**
   - Verify `no_dp_scaling=False` (default) keeps all experiments

### Integration Tests

**File:** `tests/test_integration.py`

1. **Test CLI flag parsing:**
   - Verify `--no-dp-scaling` flag is correctly parsed
   - Verify default behavior without flag

2. **Test end-to-end with real data:**
   - Run pipeline with `--no-dp-scaling` on test dataset
   - Verify output CSV contains only DP ≤ 1 experiments

### Manual Verification

Run on full dataset and verify:
```bash
python -m experiment.run --no-dp-scaling --output-dir results_single_replica
grep "dp" results_single_replica/runtime.csv | cut -d',' -f7 | sort -u
# Should show only empty/""/1, no 2/4
```

## Edge Cases

1. **No experiments after filtering:**
   - Pipeline continues with empty list (existing behavior)
   - Report generation handles empty data gracefully

2. **All experiments already single-replica:**
   - Filter is no-op, prints "excluded 0 with DP > 1"

3. **Null/empty DP values in CSV:**
   - Consistent with existing figures.py handling: `.replace("", 1).fillna(1)`
   - Null and empty treated as single-replica

## Documentation Updates

**File:** `README.md`

Update the "Pipeline CLI options" table (line 160):

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

## Alternatives Considered

### 1. Add flag to experiment.figures

**Rejected:** Not needed. If `experiment.run` filters out DP > 1, those experiments won't be in the CSV files, so `experiment.figures` won't visualize them. The hardcoded filters in figures.py serve a different purpose (ensuring clean baseline comparisons for specific publication figures).

### 2. Filter during experiment discovery

**Rejected:** Filtering after parsing is cleaner. Experiments are still validated (parse errors logged), but not run through adapters. This matches the existing `adapter.can_run()` pattern.

### 3. Make threshold configurable (e.g., --max-dp=2)

**Rejected:** YAGNI. The use case is specifically "no scaling" (single replica only). If future needs arise, the flag can be extended, but starting simple is better.

### 4. Filter in the main run loop alongside can_run()

**Rejected:** Filtering before the loop is clearer separation of concerns. Experiment-level filtering (DP threshold) is conceptually different from adapter-level compatibility (`can_run()`).

## Impact on Existing Figures

The hardcoded DP ≤ 1 filters in `experiment/figures.py` (lines 446, 477) remain **unchanged** because:

1. **Different purpose:** Those filters ensure Figures 2 (hardware portability) and 3 (workload sensitivity) show clean comparisons without DP as a confounding variable
2. **Defensive:** They work even if someone runs the full dataset then generates figures
3. **Isolated to specific figures:** Only Figures 2 and 3 need this constraint

The new CLI flag serves a different purpose: controlling which experiments run through the pipeline at all.

## Migration Path

None required. The flag is optional and defaults to existing behavior (run all experiments).

## Future Enhancements

If needed later, the filter could be extended to:
- Support arbitrary DP thresholds: `--max-dp=2`
- Support other experiment filters: `--only-fp16`, `--max-tp=4`, `--hardware=H100`
- Use a reusable filter factory pattern

However, these are out of scope for the current requirement.
