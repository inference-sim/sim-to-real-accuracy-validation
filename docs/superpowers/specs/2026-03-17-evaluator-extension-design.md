# Evaluator Extension Design: Full Ground Truth Support

**Date:** 2026-03-17
**Goal:** Extend the `experiment.run` pipeline to discover, parse, and evaluate all experiments in `vllm_data/ground_truth/` using the manifest-driven approach with `experiments.json` as the source of truth.
**Prerequisite:** [Publication Figures Spec](./2026-03-16-publication-figures-design.md), [Data Collection Plan (Discussion #598)](https://github.com/inference-sim/inference-sim/discussions/598)

### Experiment Counts

| Subset | Count | Notes |
|--------|------:|-------|
| Manifest entries (`experiments.json`) | 55 | IDs 1–59, gaps at 37, 39, 43, 45 |
| Directories on disk | 53 | IDs 47, 52 have `done=false` and no directory |
| `done=true` | 53 | IDs 47, 52 are `done=false` |
| `safe="safe"` AND `done=true` | 49 | **Default filter** — what the pipeline runs |
| Unsafe but done | 4 | IDs 1, 4, 5, 8 |

---

## Problem Statement

The current evaluator only supports 16 legacy experiments in `vllm_data/other_gt/` (timestamp-named, with `inference-perf-data/` subfolder). The primary `ground_truth/` directory contains 53 numbered experiment directories covering 8 models, 3 GPUs, 5 workloads, and 5 config knobs — but the evaluator can't discover or parse any of them.

### Six Gaps

| # | Gap | File | Impact |
|---|-----|------|--------|
| 1 | Discovery regex only matches timestamp naming | `ground_truth.py:32` | 0 of 53 numbered experiments found |
| 2 | Hardcoded `inference-perf-data/` subfolder | `ground_truth.py:76`, `vidur.py:58` | Parse fails for numbered experiments (use `results/`) |
| 3 | `extract_cpu_kv_blocks()` called unconditionally | `ground_truth.py:98` | FileNotFoundError for experiments 13–59 |
| 4 | No metadata enrichment (hw, dp, precision, safe) | `data_model.py` | Figures 2 & 4 can't filter; unsafe experiments included |
| 5 | Hardware hardcoded in all adapters | `base.py:70`, `vidur.py:90`, `llm_optimizer_est.py:92`, `aiconfigurator_est.py:132` | Wrong predictions for A100/L40S experiments |
| 6 | CSV output lacks metadata columns | `metrics.py`, `report.py` | Figures code can't derive grouping dimensions |

---

## Approach: Manifest-Driven Discovery

Replace filesystem regex scanning with `experiments.json` as the authoritative source of truth. The manifest provides experiment metadata (hardware, DP, precision, safe flag) and the `id` field maps to directory prefixes on disk.

---

## Design

### 1. Discovery & Manifest Loading (`ground_truth.py`)

**Replace** the timestamp-regex discovery with manifest-driven discovery.

```python
def load_manifest(base_dir: str) -> list[dict]:
    """Load experiments.json from base_dir."""
    path = os.path.join(base_dir, "experiments.json")
    with open(path) as fh:
        return json.load(fh)

def resolve_experiment_dir(base_dir: str, exp_id: int) -> str | None:
    """Find the directory matching '<id>-*' in base_dir."""
    prefix = f"{exp_id}-"
    for entry in os.listdir(base_dir):
        if entry.startswith(prefix) and os.path.isdir(os.path.join(base_dir, entry)):
            return os.path.abspath(os.path.join(base_dir, entry))
    return None

def discover_experiments(
    base_dir: str,
    *,
    safe_only: bool = True,
) -> list[tuple[dict, str]]:
    """Return (manifest_entry, dir_path) pairs for runnable experiments.

    Filters by safe="safe" by default. Experiments without a directory
    on disk are skipped with a warning. Each manifest_entry is a dict
    with keys: id, model, precision, hw, workload, mbt, cpu_offload,
    gpu_mem, tp, dp, safe, done, notes.
    """
    manifest = load_manifest(base_dir)
    results = []
    for entry in manifest:
        if safe_only and entry.get("safe") != "safe":
            continue
        dir_path = resolve_experiment_dir(base_dir, entry["id"])
        if dir_path is not None:
            results.append((entry, dir_path))
        else:
            logger.warning("No directory found for experiment id=%d", entry["id"])
    results.sort(key=lambda x: x[0]["id"])
    return results
```

**Return type change:** `list[str]` → `list[tuple[dict, str]]`. Metadata travels with the path.

### 2. Parsing Fixes (`ground_truth.py`)

**2a. Auto-detect results subfolder:**
```python
perf_dir = os.path.join(folder_path, "results")
if not os.path.isdir(perf_dir):
    perf_dir = os.path.join(folder_path, "inference-perf-data")
```

**2b. Optional `kv_events.jsonl`:**
```python
kv_events_path = os.path.join(folder_path, "kv_events.jsonl")
cpu_kv_blocks = extract_cpu_kv_blocks(kv_events_path) if os.path.exists(kv_events_path) else 0
```

**2c. Accept manifest metadata:**
```python
def parse_experiment(folder_path: str, manifest_entry: dict | None = None) -> Experiment:
```
When `manifest_entry` is provided, populate `Experiment` fields from the manifest:

| `Experiment` field | Source | Notes |
|--------------------|--------|-------|
| `model` | `exp-config.yaml` `model` field | **Always** the HuggingFace ID (e.g., `codellama/CodeLlama-34b-Instruct-hf`). Never from manifest — the manifest `model` is a short display name only. |
| `workload` | `manifest_entry["workload"]` | **Must** come from manifest, not folder-name parsing. Folder names have suffixes like `-1`, `-2-2` that corrupt the workload (e.g., `general-1` instead of `general`). |
| `tp` | `exp-config.yaml` `tensor_parallelism` | Unchanged — already parsed from YAML |
| `exp_id` | `manifest_entry["id"]` | Primary key for the experiment |
| `hardware` | `manifest_entry["hw"]` | `"H100"`, `"A100-80GB"`, `"L40S"` |
| `dp` | `manifest_entry["dp"]` | `null` → `None` |
| `cpu_offload` | `manifest_entry["cpu_offload"]` | |
| `gpu_mem_util` | `manifest_entry["gpu_mem"]` | |
| `precision` | `manifest_entry["precision"]` | `"FP16"` or `"FP8"` |
| `safe` | `manifest_entry["safe"]` | `"safe"`, `"unsafe"`, `"uncalibrated"` |

Legacy `other_gt/` support is out of scope for this change.

### 3. Enriched Data Model (`data_model.py`)

```python
@dataclass
class Experiment:
    # Existing fields (unchanged)
    folder: str
    model: str                  # HuggingFace ID from exp-config.yaml
    tp: int
    workload: str
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    total_kv_blocks: int
    cpu_kv_blocks: int          # 0 when kv_events.jsonl is absent
    stages: list[StageMetrics]
    summary: StageMetrics
    profile_config: dict

    # New fields from experiments.json
    exp_id: int = 0             # Experiment number (1-59)
    hardware: str = "H100"      # "H100", "A100-80GB", "L40S"
    dp: int | None = None       # Data parallelism degree
    cpu_offload: bool = False
    gpu_mem_util: float = 0.9
    precision: str = "FP16"     # "FP16", "FP8"
    safe: str = "safe"          # "safe", "unsafe", "uncalibrated"
```

Defaults ensure backwards compatibility — existing code that constructs `Experiment` without the new fields still works.

### 4. Pipeline Changes (`run.py`)

```python
def run_pipeline(data_dir, blis_binary, vidur_dir, output_dir, adapter_names=None):
    # 1. Discover (now returns metadata + paths)
    discovered = discover_experiments(data_dir)

    # 2. Parse (now receives manifest_entry)
    experiments = []
    for manifest_entry, dir_path in discovered:
        try:
            experiments.append(parse_experiment(dir_path, manifest_entry))
        except Exception:
            print(f"  SKIP (parse error): {dir_path}")
            traceback.print_exc()

    # 3-5. Build adapters, run, report (same structure)
```

### 5. CSV Schema Extension (`metrics.py`, `report.py`)

Add to `ErrorRecord` and `RuntimeRecord`:

```python
exp_id: int
hardware: str
dp: int | None
cpu_offload: bool
gpu_mem_util: float
precision: str
```

These propagate into `error_records.csv` and `runtime.csv` as new columns.

---

## Adapter Changes

### BLIS Adapters (`adapters/base.py` — shared by all 4 variants)

**Hardware mapping:**
```python
_HW_TO_BLIS: dict[str, str | None] = {
    "H100": "H100",
    "A100-80GB": "A100-80",    # BLIS hardware_config.json key
    "L40S": "L40S",            # Forward-compatible; profile pending
}
```

**Changes to `_build_common_args()`:**
```python
blis_hw = _HW_TO_BLIS.get(experiment.hardware)
# Replace hardcoded "H100":
"--hardware", blis_hw,
```

**Multi-instance support (DP and EP):**
```python
if experiment.dp and experiment.dp > 1:
    args.extend(["--num-instances", str(experiment.dp)])
```
BLIS has a single `--num-instances` flag — no separate DP or EP flags. It models N identical instances with cluster-level request routing. For EP experiments (e.g., 32, 33, 34 where vLLM uses `--enable-expert-parallel`), BLIS still uses `--num-instances` since it has no expert-distribution awareness. The EP dimension is approximated as DP replicas only.

**`can_run()` guard for L40S** (until BLIS adds the profile):
```python
def can_run(self, experiment: Experiment) -> bool:
    blis_hw = _HW_TO_BLIS.get(experiment.hardware)
    if blis_hw is None:
        return False
    # Check if hardware profile exists in hardware_config.json
    # For now, skip L40S (profile pending)
    if experiment.hardware == "L40S":
        return False
    return True
```

Remove the L40S guard when `hardware_config.json` adds the L40S entry.

**Coverage:** All H100 + A100 experiments (~47 of 49 safe+done; 2 L40S blocked until profile added). Applies to all 4 BLIS adapters: blackbox, crossmodel, roofline, trained-roofline.

### Vidur Adapter (`adapters/vidur.py`)

**Hardware mapping:**
```python
_HW_TO_VIDUR: dict[str, str] = {"H100": "h100", "A100-80GB": "a100"}
# L40S: not in DeviceSKUType enum, cannot be added without Vidur code changes
```

**Updated `can_run()`:**
```python
def can_run(self, experiment: Experiment) -> bool:
    return (experiment.model in _SUPPORTED_MODELS
            and experiment.hardware in _HW_TO_VIDUR)
```

**Updated `_build_args()`:**
```python
"--replica_config_device", _HW_TO_VIDUR[experiment.hardware],
```

**Auto-detect per-request metrics path:**
```python
perf_dir = os.path.join(experiment.folder, "results")
if not os.path.isdir(perf_dir):
    perf_dir = os.path.join(experiment.folder, "inference-perf-data")
per_req_path = os.path.join(perf_dir, "per_request_lifecycle_metrics.json")
```

**No model expansion possible** — Vidur requires pre-profiled GPU kernel timings per (model, device) pair. Current profiles exist only for Llama-2-7b, Llama-2-70b, CodeLlama-34b on h100/a100.

**Coverage:** ~9 experiments (3 models × {H100, A100} × available workloads).

### LLM-Optimizer Adapter (`adapters/llm_optimizer_est.py`)

**GPU mapping:**
```python
_HW_TO_LLM_OPT: dict[str, str] = {"H100": "H100", "A100-80GB": "A100"}
# L40S: GPU_SPECS has "L40" but NOT "L40S" (different GPU)
```

**Updated `can_run()`:**
```python
def can_run(self, experiment: Experiment) -> bool:
    if experiment.hardware not in _HW_TO_LLM_OPT:
        return False
    # FP8 not supported on A100 (A100 has FP8_TFLOPS=None)
    if experiment.precision == "FP8" and experiment.hardware == "A100-80GB":
        return False
    # ... existing profile_config check ...
```

**Updated `run()`:**
```python
gpu_name = _HW_TO_LLM_OPT[experiment.hardware]
precision = experiment.precision.lower()  # "fp16" or "fp8"
# Use experiment.precision instead of model_config.inferred_precision
```

**MoE note:** LLM-Optimizer has no MoE awareness — its roofline model treats all models as dense. MoE experiments will produce inaccurate results but won't crash. The figures spec already expects varying accuracy across simulators, so letting MoE experiments run and documenting the limitation is acceptable.

**Coverage:** ~46 experiments (H100 + A100, excluding A100 FP8).

### AIConfigurator Adapter (`adapters/aiconfigurator_est.py`)

**System mapping:**
```python
_HW_TO_AICONFIG: dict[str, str] = {"H100": "h100_sxm"}
# A100: "a100_sxm" exists but has NO vllm backend perf data
# L40S: "l40s" exists but has NO vllm backend perf data
```

**Updated `_MOE_MODELS`:** The current set uses `Mixtral-8x22B-v0.1` (base variant) but the ground truth experiments use the Instruct variant. Update to match actual HuggingFace IDs from `exp-config.yaml`:
```python
_MOE_MODELS: frozenset[str] = frozenset({
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",   # Was Mixtral-8x22B-v0.1 (wrong)
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",  # New: MoE with 16 experts
})
```

**Updated `can_run()`:**
```python
def can_run(self, experiment: Experiment) -> bool:
    if experiment.model in _MOE_MODELS:
        return False
    if experiment.hardware not in _HW_TO_AICONFIG:
        return False
    # ... existing profile_config check ...
```

**Updated `run()`:**
```python
system_name = _HW_TO_AICONFIG[experiment.hardware]
```

**Coverage:** ~20-25 experiments (H100, dense models only).

---

## Adapter Coverage Matrix

All 4 BLIS adapters (blackbox, crossmodel, roofline, trained-roofline) inherit from `BaseBLISAdapter` and share the same hardware routing in `_build_common_args()`. Hardware fix in `base.py` covers all 4.

Counts below are of the 49 safe+done experiments (default filter).

| Adapter | H100 Dense | H100 MoE | A100 Dense | A100 FP8 | L40S | Total (approx) |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| BLIS (all 4) | Yes | Yes | Yes | Yes | Pending | ~47 |
| Vidur | 3 models | No | 3 models | No | No | ~9 |
| LLM-Optimizer | Yes | Yes* | Yes | No | No | ~46 |
| AIConfigurator | Dense only | No | No | No | No | ~20 |

*MoE results will be inaccurate (dense-only roofline).

---

## Files Changed

| File | Changes |
|------|---------|
| `experiment/ground_truth.py` | Manifest-driven discovery; auto-detect results subfolder; handle missing `kv_events.jsonl`; accept manifest metadata |
| `experiment/data_model.py` | Add `exp_id`, `hardware`, `dp`, `cpu_offload`, `gpu_mem_util`, `precision`, `safe` fields with defaults |
| `experiment/adapters/base.py` | Hardware mapping; `--num-instances` for dp > 1; `can_run()` L40S guard |
| `experiment/adapters/vidur.py` | Hardware mapping; auto-detect perf_dir; `can_run()` checks hardware |
| `experiment/adapters/llm_optimizer_est.py` | Hardware mapping; precision from experiment; `can_run()` skips L40S and A100 FP8 |
| `experiment/adapters/aiconfigurator_est.py` | Hardware mapping; extend `_MOE_MODELS`; `can_run()` skips non-H100 |
| `experiment/metrics.py` | Add metadata fields to `ErrorRecord` and `RuntimeRecord` |
| `experiment/report.py` | Write new columns to CSVs |
| `experiment/run.py` | Update `run_pipeline()` for manifest-driven discovery flow |
| `tests/test_ground_truth.py` | Manifest discovery tests; auto-detect perf_dir tests; missing kv_events tests |
| `tests/test_adapter_base.py` | Hardware mapping tests; DP routing tests; L40S guard tests |
| `tests/test_vidur_adapter.py` | Hardware + model `can_run()` tests; perf_dir auto-detection |
| `tests/test_llm_optimizer_adapter.py` | Hardware mapping; FP8/A100 exclusion tests |
| `tests/test_aiconfigurator_adapter.py` | MoE exclusion update; non-H100 exclusion tests |
| `tests/test_metrics.py` | New fields in ErrorRecord |
| `tests/test_report.py` | CSV column output tests |
| `tests/test_run.py` | Pipeline integration with manifest |

---

## Out of Scope

- Adding L40S profile to BLIS `hardware_config.json` (separate inference-sim PR)
- Adding new Vidur GPU kernel profiles (requires running profiling on target hardware)
- Fixing LLM-Optimizer's MoE roofline model (separate llm-optimizer effort)
- Adding vllm backend data to AIConfigurator for A100/L40S (separate aiconfigurator effort)
- Changes to the `figures.py` module (already designed in the publication figures spec)
