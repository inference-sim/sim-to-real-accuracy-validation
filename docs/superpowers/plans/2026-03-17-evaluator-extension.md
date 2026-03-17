# Evaluator Extension Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the evaluator pipeline to discover, parse, and evaluate all 49 safe ground-truth vLLM experiments using manifest-driven discovery with `experiments.json`.

**Architecture:** Replace filesystem regex scanning with manifest-driven discovery. Enrich the `Experiment` dataclass with hardware/precision/dp metadata from the manifest. Update all three non-BLIS adapters (Vidur, LLM-Optimizer, AIConfigurator) to use experiment metadata for hardware, precision, and dp — respecting simulator limitations documented in `docs/simulator-limitations.md`.

**Tech Stack:** Python 3.10+, pytest, pyyaml, dataclasses

**Spec:** `docs/superpowers/specs/2026-03-17-evaluator-extension-design.md`
**Limitations:** `docs/simulator-limitations.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `experiment/data_model.py` | Modify | Add 7 new fields to `Experiment` dataclass |
| `experiment/ground_truth.py` | Modify | Manifest-driven discovery, auto-detect results dir, optional kv_events, accept manifest metadata |
| `experiment/run.py` | Modify | Wire new discovery return type into pipeline |
| `experiment/adapters/vidur.py` | Modify | Hardware mapping, network_device, dp via num_replicas, auto-detect perf_dir |
| `experiment/adapters/llm_optimizer_est.py` | Modify | Hardware mapping, experiment precision, can_run guards |
| `experiment/adapters/aiconfigurator_est.py` | Modify | Hardware mapping, _MOE_MODELS fix, precision via profiles, can_run guards |
| `experiment/metrics.py` | Modify | Add metadata fields to ErrorRecord and RuntimeRecord |
| `experiment/report.py` | Modify | Write new columns to CSVs |
| `tests/test_ground_truth.py` | Modify | Manifest discovery, auto-detect perf_dir, optional kv_events tests |
| `tests/test_vidur_adapter.py` | Modify | Hardware + dp can_run tests, build_args verification |
| `tests/test_llm_optimizer_adapter.py` | Modify | Hardware, precision, can_run guards |
| `tests/test_aiconfigurator_adapter.py` | Modify | MoE update, hardware, precision profiles |
| `tests/test_metrics.py` | Modify | New fields in ErrorRecord/RuntimeRecord |
| `tests/test_report.py` | Modify | CSV column output |
| `tests/test_run.py` | Modify | Pipeline integration with manifest discovery |

---

## Task 1: Enrich `Experiment` dataclass

**Files:**
- Modify: `experiment/data_model.py:32-45`
- Test: `tests/test_data_model.py`

- [ ] **Step 1: Write failing test — new fields exist with defaults**

```python
# tests/test_data_model.py — add to existing file

def test_experiment_new_fields_have_defaults():
    """New metadata fields should have defaults for backward compat."""
    from experiment.data_model import LatencyDistribution, StageMetrics, ThroughputMetrics
    stage = StageMetrics(
        stage_index=0, rate=5.0, duration=600.0, num_requests=100,
        e2e=LatencyDistribution(mean=100.0),
        ttft=LatencyDistribution(mean=10.0),
        itl=LatencyDistribution(mean=5.0),
        throughput=ThroughputMetrics(100.0, 50.0, 5.0),
    )
    exp = Experiment(
        folder="/tmp/test", model="test/model", tp=1, workload="general",
        max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128,
        total_kv_blocks=1000, cpu_kv_blocks=0,
        stages=[stage], summary=stage, profile_config={},
    )
    # New fields should have defaults
    assert exp.exp_id == 0
    assert exp.hardware == "H100"
    assert exp.dp is None
    assert exp.cpu_offload is False
    assert exp.gpu_mem_util == 0.9
    assert exp.precision == "FP16"
    assert exp.safe == "safe"


def test_experiment_new_fields_settable():
    """New metadata fields should be settable."""
    from experiment.data_model import LatencyDistribution, StageMetrics, ThroughputMetrics
    stage = StageMetrics(
        stage_index=0, rate=5.0, duration=600.0, num_requests=100,
        e2e=LatencyDistribution(mean=100.0),
        ttft=LatencyDistribution(mean=10.0),
        itl=LatencyDistribution(mean=5.0),
        throughput=ThroughputMetrics(100.0, 50.0, 5.0),
    )
    exp = Experiment(
        folder="/tmp/test", model="test/model", tp=1, workload="general",
        max_model_len=4096, max_num_batched_tokens=2048, max_num_seqs=128,
        total_kv_blocks=1000, cpu_kv_blocks=0,
        stages=[stage], summary=stage, profile_config={},
        exp_id=42, hardware="A100-80GB", dp=2,
        cpu_offload=True, gpu_mem_util=0.95,
        precision="FP8", safe="unsafe",
    )
    assert exp.exp_id == 42
    assert exp.hardware == "A100-80GB"
    assert exp.dp == 2
    assert exp.cpu_offload is True
    assert exp.gpu_mem_util == 0.95
    assert exp.precision == "FP8"
    assert exp.safe == "unsafe"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_model.py -v -k "new_fields"`
Expected: FAIL — `Experiment` does not accept `exp_id` etc.

- [ ] **Step 3: Add new fields to Experiment**

In `experiment/data_model.py`, add after `profile_config: dict`:

```python
    # New fields from experiments.json manifest
    exp_id: int = 0             # Experiment number (1-59)
    hardware: str = "H100"      # "H100", "A100-80GB", "L40S"
    dp: int | None = None       # Data parallelism degree
    cpu_offload: bool = False
    gpu_mem_util: float = 0.9
    precision: str = "FP16"     # "FP16", "FP8"
    safe: str = "safe"          # "safe", "unsafe", "uncalibrated"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_model.py -v -k "new_fields"`
Expected: PASS

- [ ] **Step 5: Run full test suite to check backward compat**

Run: `pytest tests/ -v --tb=short`
Expected: All 295+ existing tests still pass (defaults ensure compat)

- [ ] **Step 6: Commit**

```bash
git add experiment/data_model.py tests/test_data_model.py
git commit -m "feat(data_model): add experiment metadata fields from manifest"
```

---

## Task 2: Manifest-driven discovery

**Files:**
- Modify: `experiment/ground_truth.py:32-48` (replace `discover_experiments`)
- Test: `tests/test_ground_truth.py`

- [ ] **Step 1: Write failing tests for manifest discovery**

Add to `tests/test_ground_truth.py`:

```python
class TestManifestDiscovery:
    """Tests for the new manifest-driven discover_experiments."""

    def test_discovers_safe_experiments(self, tmp_path):
        """Only safe experiments with directories should be returned."""
        manifest = [
            {"id": 13, "model": "Qwen3-14B", "precision": "FP16", "hw": "H100",
             "workload": "general", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "safe": "safe", "done": True, "notes": ""},
            {"id": 1, "model": "Codellama-34b", "precision": "FP16", "hw": "H100",
             "workload": "general", "mbt": 2048, "cpu_offload": True,
             "gpu_mem": 0.9, "tp": 2, "dp": None, "safe": "unsafe", "done": True, "notes": ""},
        ]
        (tmp_path / "experiments.json").write_text(json.dumps(manifest))
        (tmp_path / "13-qwen3-14b-tp1-general").mkdir()
        (tmp_path / "1-codellama-34b-tp2-general").mkdir()

        result = discover_experiments(str(tmp_path))
        assert len(result) == 1
        entry, path = result[0]
        assert entry["id"] == 13
        assert "13-qwen3-14b" in path

    def test_discovers_all_when_safe_only_false(self, tmp_path):
        """safe_only=False returns all experiments with directories."""
        manifest = [
            {"id": 1, "safe": "unsafe", "done": True, "model": "m", "precision": "FP16",
             "hw": "H100", "workload": "general", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "notes": ""},
            {"id": 2, "safe": "safe", "done": True, "model": "m", "precision": "FP16",
             "hw": "H100", "workload": "codegen", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "notes": ""},
        ]
        (tmp_path / "experiments.json").write_text(json.dumps(manifest))
        (tmp_path / "1-model-tp1-general").mkdir()
        (tmp_path / "2-model-tp1-codegen").mkdir()

        result = discover_experiments(str(tmp_path), safe_only=False)
        assert len(result) == 2

    def test_skips_missing_directories(self, tmp_path):
        """Experiments without directories are skipped with warning."""
        manifest = [
            {"id": 47, "safe": "safe", "done": False, "model": "m", "precision": "FP16",
             "hw": "H100", "workload": "general", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "notes": ""},
        ]
        (tmp_path / "experiments.json").write_text(json.dumps(manifest))
        # No directory for id=47

        result = discover_experiments(str(tmp_path))
        assert len(result) == 0

    def test_sorted_by_id(self, tmp_path):
        """Results are sorted by experiment id."""
        manifest = [
            {"id": 20, "safe": "safe", "done": True, "model": "m", "precision": "FP16",
             "hw": "H100", "workload": "a", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "notes": ""},
            {"id": 3, "safe": "safe", "done": True, "model": "m", "precision": "FP16",
             "hw": "H100", "workload": "b", "mbt": 2048, "cpu_offload": False,
             "gpu_mem": 0.9, "tp": 1, "dp": None, "notes": ""},
        ]
        (tmp_path / "experiments.json").write_text(json.dumps(manifest))
        (tmp_path / "20-model-tp1-a").mkdir()
        (tmp_path / "3-model-tp1-b").mkdir()

        result = discover_experiments(str(tmp_path))
        assert result[0][0]["id"] == 3
        assert result[1][0]["id"] == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ground_truth.py::TestManifestDiscovery -v`
Expected: FAIL — `discover_experiments` still returns `list[str]`

- [ ] **Step 3: Implement manifest-driven discovery**

Replace `discover_experiments` in `experiment/ground_truth.py` with the three functions from the spec: `load_manifest`, `resolve_experiment_dir`, and the new `discover_experiments`. Keep the old regex and `_extract_workload` for now (removed in Task 3).

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
    """Return (manifest_entry, dir_path) pairs for runnable experiments."""
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

- [ ] **Step 4: Run manifest discovery tests**

Run: `pytest tests/test_ground_truth.py::TestManifestDiscovery -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiment/ground_truth.py tests/test_ground_truth.py
git commit -m "feat(ground_truth): manifest-driven experiment discovery"
```

---

## Task 3: Parsing fixes — auto-detect results dir, optional kv_events, manifest metadata

**Files:**
- Modify: `experiment/ground_truth.py:51-113` (update `parse_experiment`)
- Test: `tests/test_ground_truth.py`

- [ ] **Step 1: Update `_make_exp_dir` helper to support `perf_subdir` and `include_kv_events`**

Modify the existing helper in `tests/test_ground_truth.py` **before** writing any tests that use the new parameters:

```python
def _make_exp_dir(
    tmp_path,
    *,
    folder_name="20250101-120000-llama2-7b-tp1-codegen",
    perf_subdir="inference-perf-data",  # NEW: "results" for numbered experiments
    include_kv_events=True,             # NEW: False to skip kv_events.jsonl
    kv_events_lines=None,
    ...  # keep existing params
):
```

Key changes:
1. Replace hardcoded `"inference-perf-data"` with `perf_subdir` parameter
2. When `include_kv_events=False`, skip writing `kv_events.jsonl` entirely
3. When `include_kv_events=True` and `kv_events_lines is None`, use existing default kv_events

- [ ] **Step 2: Write failing test — auto-detect `results/` subfolder**

```python
class TestParseExperimentFixes:

    def test_auto_detects_results_subfolder(self, tmp_path):
        """Numbered experiments use results/ not inference-perf-data/."""
        exp_dir = _make_exp_dir(tmp_path, folder_name="13-qwen3-14b-tp1-general",
                                perf_subdir="results")
        exp = parse_experiment(str(exp_dir))
        assert exp.model == "meta-llama/Llama-2-7b-hf"  # from exp-config.yaml

    def test_falls_back_to_inference_perf_data(self, tmp_path):
        """Legacy experiments still use inference-perf-data/."""
        exp_dir = _make_exp_dir(tmp_path)
        exp = parse_experiment(str(exp_dir))
        assert exp.model == "meta-llama/Llama-2-7b-hf"
```

- [ ] **Step 3: Write failing test — optional kv_events.jsonl**

```python
    def test_missing_kv_events_gives_zero_cpu_blocks(self, tmp_path):
        """When kv_events.jsonl is absent, cpu_kv_blocks should be 0."""
        exp_dir = _make_exp_dir(tmp_path, folder_name="13-qwen3-14b-tp1-general",
                                perf_subdir="results", include_kv_events=False)
        exp = parse_experiment(str(exp_dir))
        assert exp.cpu_kv_blocks == 0
```

- [ ] **Step 4: Write failing test — manifest metadata populates Experiment fields**

```python
    def test_manifest_metadata_populates_fields(self, tmp_path):
        """When manifest_entry is provided, new fields are populated."""
        exp_dir = _make_exp_dir(tmp_path, folder_name="13-qwen3-14b-tp1-general",
                                perf_subdir="results")
        manifest_entry = {
            "id": 13, "model": "Qwen3-14B", "precision": "FP16", "hw": "H100",
            "workload": "general", "mbt": 2048, "cpu_offload": False,
            "gpu_mem": 0.9, "tp": 1, "dp": None, "safe": "safe", "done": True, "notes": "",
        }
        exp = parse_experiment(str(exp_dir), manifest_entry=manifest_entry)
        assert exp.exp_id == 13
        assert exp.hardware == "H100"
        assert exp.workload == "general"  # From manifest, not folder name
        assert exp.dp is None
        assert exp.precision == "FP16"
        assert exp.safe == "safe"

    def test_workload_from_manifest_not_foldername(self, tmp_path):
        """Workload must come from manifest to avoid suffix corruption."""
        # Folder name has -1 suffix that would corrupt workload extraction
        exp_dir = _make_exp_dir(tmp_path, folder_name="9-mixtral-8x7b-tp2-general-1",
                                perf_subdir="results")
        manifest_entry = {
            "id": 9, "model": "Mixtral-8x7B", "precision": "FP16", "hw": "H100",
            "workload": "general", "mbt": 2048, "cpu_offload": True,
            "gpu_mem": 0.9, "tp": 2, "dp": 1, "safe": "safe", "done": True, "notes": "",
        }
        exp = parse_experiment(str(exp_dir), manifest_entry=manifest_entry)
        assert exp.workload == "general"  # NOT "general-1"
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/test_ground_truth.py::TestParseExperimentFixes -v`
Expected: FAIL

- [ ] **Step 6: Implement parsing fixes**

Update `parse_experiment` in `experiment/ground_truth.py`:

1. Add `manifest_entry: dict | None = None` parameter
2. Auto-detect `results/` vs `inference-perf-data/`
3. Make `kv_events.jsonl` optional
4. When `manifest_entry` provided: workload from manifest, metadata fields populated
5. Model ALWAYS from `exp-config.yaml` (never from manifest)

```python
def parse_experiment(folder_path: str, manifest_entry: dict | None = None) -> Experiment:
    """Parse a single experiment directory into an :class:`Experiment`."""
    folder_path = os.path.abspath(folder_path)
    folder_name = os.path.basename(folder_path)

    # 1. Parse exp-config.yaml
    with open(os.path.join(folder_path, "exp-config.yaml")) as fh:
        exp_cfg = yaml.safe_load(fh)

    model = exp_cfg["model"]
    tp = exp_cfg["tensor_parallelism"]
    max_model_len = exp_cfg["max_model_len"]
    max_num_batched_tokens = exp_cfg["max_num_batched_tokens"]
    max_num_seqs = exp_cfg["max_num_seqs"]

    # 2. Workload: from manifest if available, else from folder name
    if manifest_entry is not None:
        workload = manifest_entry["workload"]
    else:
        workload = _extract_workload(folder_name)

    # 3. Parse profile.yaml
    with open(os.path.join(folder_path, "profile.yaml")) as fh:
        profile_config = yaml.safe_load(fh)

    stages_config = profile_config["load"]["stages"]

    # 4. Auto-detect results subfolder
    perf_dir = os.path.join(folder_path, "results")
    if not os.path.isdir(perf_dir):
        perf_dir = os.path.join(folder_path, "inference-perf-data")

    # 5. Parse stage lifecycle metrics
    stage_files = sorted(glob.glob(os.path.join(perf_dir, "stage_*_lifecycle_metrics.json")))
    if len(stage_files) != len(stages_config):
        logger.warning(
            "Stage count mismatch in %s: %d files vs %d in profile.yaml",
            folder_name, len(stage_files), len(stages_config),
        )
    stages: list[StageMetrics] = []
    for i, stage_file in enumerate(stage_files):
        with open(stage_file) as fh:
            stage_data = json.load(fh)
        stage_cfg = stages_config[i] if i < len(stages_config) else {}
        stages.append(_parse_stage_metrics(stage_data, stage_index=i, stage_cfg=stage_cfg))

    # 6. Parse summary lifecycle metrics
    summary_path = os.path.join(perf_dir, "summary_lifecycle_metrics.json")
    with open(summary_path) as fh:
        summary_data = json.load(fh)
    summary = _parse_stage_metrics(summary_data, stage_index=-1, stage_cfg=None)

    # 7. KV blocks
    total_kv_blocks = extract_total_kv_blocks(os.path.join(folder_path, "vllm.log"))
    kv_events_path = os.path.join(folder_path, "kv_events.jsonl")
    cpu_kv_blocks = extract_cpu_kv_blocks(kv_events_path) if os.path.exists(kv_events_path) else 0

    # 8. Build Experiment with optional manifest metadata
    kwargs = {}
    if manifest_entry is not None:
        kwargs = dict(
            exp_id=manifest_entry["id"],
            hardware=manifest_entry["hw"],
            dp=manifest_entry["dp"],
            cpu_offload=manifest_entry["cpu_offload"],
            gpu_mem_util=manifest_entry["gpu_mem"],
            precision=manifest_entry["precision"],
            safe=manifest_entry["safe"],
        )

    return Experiment(
        folder=folder_path,
        model=model,
        tp=tp,
        workload=workload,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        total_kv_blocks=total_kv_blocks,
        cpu_kv_blocks=cpu_kv_blocks,
        stages=stages,
        summary=summary,
        profile_config=profile_config,
        **kwargs,
    )
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_ground_truth.py -v`
Expected: All tests pass (old + new)

- [ ] **Step 8: Commit**

```bash
git add experiment/ground_truth.py tests/test_ground_truth.py
git commit -m "feat(ground_truth): auto-detect results dir, optional kv_events, manifest metadata"
```

---

## Task 4: Wire manifest discovery into pipeline

**Files:**
- Modify: `experiment/run.py:74-91`
- Test: `tests/test_run.py`

- [ ] **Step 1: Write failing test**

```python
def test_run_pipeline_uses_manifest_discovery(tmp_path, monkeypatch):
    """Pipeline should call discover_experiments and pass manifest_entry to parse_experiment."""
    calls = []

    def mock_discover(base_dir, *, safe_only=True):
        return [
            ({"id": 13, "workload": "general", "hw": "H100", "precision": "FP16",
              "dp": None, "cpu_offload": False, "gpu_mem": 0.9, "safe": "safe"},
             str(tmp_path / "13-exp")),
        ]

    def mock_parse(folder_path, manifest_entry=None):
        calls.append(("parse", folder_path, manifest_entry))
        raise ValueError("stop here")  # Don't need full parse

    monkeypatch.setattr("experiment.run.discover_experiments", mock_discover)
    monkeypatch.setattr("experiment.run.parse_experiment", mock_parse)

    run_pipeline(str(tmp_path), "blis", "vidur", str(tmp_path / "out"), adapter_names=[])

    assert len(calls) == 1
    assert calls[0][1] == str(tmp_path / "13-exp")
    assert calls[0][2]["id"] == 13
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run.py -v -k "manifest_discovery"`
Expected: FAIL — `parse_experiment` called without `manifest_entry`

- [ ] **Step 3: Update `run_pipeline` in `experiment/run.py`**

Change the discovery + parse loop from:

```python
experiment_dirs = discover_experiments(data_dir)
for d in experiment_dirs:
    experiments.append(parse_experiment(d))
```

To:

```python
discovered = discover_experiments(data_dir)
if not discovered:
    print(f"No experiments found in {data_dir}")
    return [], []

print(f"Found {len(discovered)} experiments")

experiments = []
for manifest_entry, dir_path in discovered:
    try:
        experiments.append(parse_experiment(dir_path, manifest_entry=manifest_entry))
    except Exception:
        print(f"  SKIP (parse error): {dir_path}")
        traceback.print_exc()
```

Also update the `RuntimeRecord` creation to include new metadata fields (done in Task 8).

- [ ] **Step 4: Update existing `tests/test_run.py` mocks for new return type**

The existing `TestRunPipeline` tests mock `discover_experiments` to return `list[str]`. After Task 2, it returns `list[tuple[dict, str]]`. Update all 4 existing test methods:

Replace every occurrence of:
```python
mock_discover.return_value = ["/tmp/exp"]
```

With:
```python
_MANIFEST_STUB = {"id": 1, "hw": "H100", "dp": None, "cpu_offload": False,
                  "gpu_mem": 0.9, "precision": "FP16", "safe": "safe",
                  "workload": "codegen", "model": "m", "mbt": 2048, "done": True, "notes": ""}
mock_discover.return_value = [(_MANIFEST_STUB, "/tmp/exp")]
```

Also update every `mock_parse` setup — `parse_experiment` now accepts `manifest_entry=`:
```python
mock_parse.side_effect = lambda path, manifest_entry=None: exp
```

Apply this change in: `test_runs_matching_adapters`, `test_skips_when_cant_run`, `test_continues_on_adapter_failure`. The fourth test (`test_no_experiments_returns_empty`) already uses `mock_discover.return_value = []` which is correct for both return types.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_run.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add experiment/run.py tests/test_run.py
git commit -m "feat(run): wire manifest-driven discovery into pipeline"
```

---

## Task 5: Vidur adapter — hardware, network_device, dp, perf_dir

**Files:**
- Modify: `experiment/adapters/vidur.py`
- Test: `tests/test_vidur_adapter.py`

**Simulator limitations to respect (from `docs/simulator-limitations.md`):**
- No MoE (zero support) — already excluded by model list
- No FP8 (hardcoded FP16) — exclude via `can_run`
- No L40S (only A40, A100, H100 SKUs) — exclude via hardware map
- No cpu_offload (single-tier GPU memory) — **true limitation, cannot fix**
- No gpu_mem modeling (adapter passes explicit num_blocks) — **true limitation**
- No chunked prefill (vLLM scheduler) — **true limitation, affects TTFT accuracy**
- dp via num_replicas is an **approximation** (assumes round-robin, aggregate trace)

- [ ] **Step 1: Write failing tests for can_run with hardware and FP8 guards**

```python
def test_can_run_rejects_unsupported_hardware(self):
    exp = _make_experiment()
    exp.hardware = "L40S"
    adapter = VidurAdapter("/tmp/vidur")
    assert adapter.can_run(exp) is False

def test_can_run_rejects_fp8(self):
    exp = _make_experiment()
    exp.precision = "FP8"
    adapter = VidurAdapter("/tmp/vidur")
    assert adapter.can_run(exp) is False

def test_can_run_accepts_a100_supported_model(self):
    exp = _make_experiment(model="codellama/CodeLlama-34b-Instruct-hf")
    exp.hardware = "A100-80GB"
    adapter = VidurAdapter("/tmp/vidur")
    assert adapter.can_run(exp) is True
```

- [ ] **Step 2: Write failing tests for build_args with hardware + dp**

```python
def test_build_args_uses_experiment_hardware(self):
    adapter = VidurAdapter("/tmp/vidur")
    exp = _make_experiment()
    exp.hardware = "A100-80GB"
    args = adapter._build_args(exp, "/tmp/trace.csv", "/tmp/out")
    assert "--replica_config_device" in args
    idx = args.index("--replica_config_device")
    assert args[idx + 1] == "a100"
    # Network device should also match
    idx_net = args.index("--replica_config_network_device")
    assert args[idx_net + 1] == "a100_pairwise_nvlink"

def test_build_args_includes_dp_replicas(self):
    adapter = VidurAdapter("/tmp/vidur")
    exp = _make_experiment()
    exp.dp = 2
    args = adapter._build_args(exp, "/tmp/trace.csv", "/tmp/out")
    idx = args.index("--cluster_config_num_replicas")
    assert args[idx + 1] == "2"
    assert "--global_scheduler_config_type" in args
    # Must appear exactly once (not duplicated)
    assert args.count("--cluster_config_num_replicas") == 1

def test_build_args_no_dp_stays_single_replica(self):
    adapter = VidurAdapter("/tmp/vidur")
    exp = _make_experiment()
    exp.dp = None
    args = adapter._build_args(exp, "/tmp/trace.csv", "/tmp/out")
    idx = args.index("--cluster_config_num_replicas")
    assert args[idx + 1] == "1"
    assert "--global_scheduler_config_type" not in args
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_vidur_adapter.py -v -k "hardware or fp8 or dp or a100"`
Expected: FAIL

- [ ] **Step 4: Implement Vidur adapter changes**

In `experiment/adapters/vidur.py`:

1. Add hardware and network-device mappings:
```python
_HW_TO_VIDUR: dict[str, str] = {"H100": "h100", "A100-80GB": "a100"}
_HW_TO_VIDUR_NETWORK: dict[str, str] = {
    "H100": "h100_pairwise_nvlink",
    "A100-80GB": "a100_pairwise_nvlink",
}
```

2. Update `can_run`:
```python
def can_run(self, experiment: Experiment) -> bool:
    return (experiment.model in _SUPPORTED_MODELS
            and experiment.hardware in _HW_TO_VIDUR
            and experiment.precision != "FP8")
```

3. Update `_build_args` — replace hardcoded `"h100"` with hardware mapping, add network_device, add dp support:
```python
"--replica_config_device", _HW_TO_VIDUR[experiment.hardware],
"--replica_config_network_device", _HW_TO_VIDUR_NETWORK[experiment.hardware],
```

For dp — **remove** the existing hardcoded `"--cluster_config_num_replicas", "1"` from the base args and replace with conditional logic at the end:
```python
# Remove from base args: "--cluster_config_num_replicas", "1"
# Add at end of _build_args:
num_replicas = str(experiment.dp) if experiment.dp and experiment.dp > 1 else "1"
args += ["--cluster_config_num_replicas", num_replicas]
if experiment.dp and experiment.dp > 1:
    args += ["--global_scheduler_config_type", "round_robin"]
```
This avoids the arg appearing twice (once hardcoded in base, once in the dp block).

4. Update `run` — auto-detect perf_dir:
```python
perf_dir = os.path.join(experiment.folder, "results")
if not os.path.isdir(perf_dir):
    perf_dir = os.path.join(experiment.folder, "inference-perf-data")
per_req_path = os.path.join(perf_dir, "per_request_lifecycle_metrics.json")
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_vidur_adapter.py -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add experiment/adapters/vidur.py tests/test_vidur_adapter.py
git commit -m "feat(vidur): hardware mapping, network_device, dp via num_replicas"
```

---

## Task 6: LLM-Optimizer adapter — hardware, precision, can_run guards

**Files:**
- Modify: `experiment/adapters/llm_optimizer_est.py`
- Test: `tests/test_llm_optimizer_adapter.py`

**Simulator limitations to respect:**
- No L40S (`GPU_SPECS` has "L40" not "L40S") — exclude via `can_run`
- A100 has `FP8_TFLOPS=None` — exclude A100+FP8 via `can_run`
- Mean metrics only — **true limitation**, no P50/P90/P99
- No MoE awareness (dense-only roofline) — runs but inaccurate, documented
- `mbt`, `cpu_offload`, `gpu_mem`, `dp` not modeled — **true limitations**
- `precision` now correctly passed via `experiment.precision.lower()`

- [ ] **Step 1: Write failing tests**

```python
def test_can_run_rejects_l40s(self):
    exp = _make_experiment()
    exp.hardware = "L40S"
    assert LLMOptimizerEstimateAdapter().can_run(exp) is False

def test_can_run_rejects_a100_fp8(self):
    exp = _make_experiment()
    exp.hardware = "A100-80GB"
    exp.precision = "FP8"
    assert LLMOptimizerEstimateAdapter().can_run(exp) is False

def test_can_run_accepts_a100_fp16(self):
    exp = _make_experiment()
    exp.hardware = "A100-80GB"
    exp.precision = "FP16"
    assert LLMOptimizerEstimateAdapter().can_run(exp) is True

@patch("experiment.adapters.llm_optimizer_est.estimate_llm_performance")
@patch("experiment.adapters.llm_optimizer_est.get_model_config_from_hf")
def test_run_passes_experiment_precision(self, mock_get_cfg, mock_estimate):
    """Precision should come from experiment, not model_config.inferred_precision."""
    mock_get_cfg.return_value = MagicMock()
    mock_estimate.return_value = _FakePerformanceResult(
        ttft_ms=25.0, itl_ms=3.5, e2e_latency_s=1.8,
        output_throughput_tps=980.0, input_throughput_tps=2900.0,
        requests_per_sec=5.2, concurrency=9,
    )
    adapter = LLMOptimizerEstimateAdapter()
    exp = _make_experiment()
    exp.precision = "FP8"
    adapter.run(exp)
    # Check the precision kwarg passed to estimate_llm_performance
    _, kwargs = mock_estimate.call_args
    assert kwargs["precision"] == "fp8"

@patch("experiment.adapters.llm_optimizer_est.estimate_llm_performance")
@patch("experiment.adapters.llm_optimizer_est.get_model_config_from_hf")
def test_run_uses_hardware_gpu_name(self, mock_get_cfg, mock_estimate):
    """gpu_name should come from experiment hardware, not hardcoded H100."""
    mock_get_cfg.return_value = MagicMock()
    mock_estimate.return_value = _FakePerformanceResult(
        ttft_ms=25.0, itl_ms=3.5, e2e_latency_s=1.8,
        output_throughput_tps=980.0, input_throughput_tps=2900.0,
        requests_per_sec=5.2, concurrency=9,
    )
    adapter = LLMOptimizerEstimateAdapter()
    exp = _make_experiment()
    exp.hardware = "A100-80GB"
    adapter.run(exp)
    _, kwargs = mock_estimate.call_args
    assert kwargs["gpu_name"] == "A100"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_optimizer_adapter.py -v -k "l40s or a100 or precision or hardware"`
Expected: FAIL

- [ ] **Step 3: Implement changes**

In `experiment/adapters/llm_optimizer_est.py`:

1. Add hardware mapping:
```python
_HW_TO_LLM_OPT: dict[str, str] = {"H100": "H100", "A100-80GB": "A100"}
```

2. Update `can_run`:
```python
def can_run(self, experiment: Experiment) -> bool:
    if experiment.hardware not in _HW_TO_LLM_OPT:
        return False
    if experiment.precision == "FP8" and experiment.hardware == "A100-80GB":
        return False
    # ... existing profile_config check ...
```

3. Update `run` — use hardware mapping and experiment precision:
```python
gpu_name = _HW_TO_LLM_OPT[experiment.hardware]
precision = experiment.precision.lower()  # "fp16" or "fp8"
```

Pass these instead of the hardcoded values.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_llm_optimizer_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiment/adapters/llm_optimizer_est.py tests/test_llm_optimizer_adapter.py
git commit -m "feat(llm_optimizer): hardware mapping, experiment precision, can_run guards"
```

---

## Task 7: AIConfigurator adapter — hardware, MoE fix, precision profiles, can_run guards

**Files:**
- Modify: `experiment/adapters/aiconfigurator_est.py`
- Test: `tests/test_aiconfigurator_adapter.py`

**Simulator limitations to respect:**
- No A100/L40S vLLM perf data — exclude via `can_run`
- MoE excluded (vLLM backend unsupported) — already excluded, but fix `_MOE_MODELS`
- Mean metrics only — **true limitation**
- `mbt`, `cpu_offload`, `gpu_mem`, `dp` not modeled — **true limitations**
- Precision: FP8 auto-selected on H100 — **fixed** via `profiles=["float16_default"]` for FP16 experiments

- [ ] **Step 1: Write failing tests**

```python
def test_can_run_rejects_non_h100(self):
    exp = _make_experiment()
    exp.hardware = "A100-80GB"
    assert AIConfiguratorEstimateAdapter().can_run(exp) is False

def test_can_run_rejects_mixtral_8x22b_instruct(self):
    """_MOE_MODELS should include the Instruct variant."""
    exp = _make_experiment(model="mistralai/Mixtral-8x22B-Instruct-v0.1")
    assert AIConfiguratorEstimateAdapter().can_run(exp) is False

def test_can_run_rejects_llama4_scout(self):
    exp = _make_experiment(model="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic")
    assert AIConfiguratorEstimateAdapter().can_run(exp) is False

@patch("experiment.adapters.aiconfigurator_est._run_task")
@patch("experiment.adapters.aiconfigurator_est._create_task_config")
def test_run_passes_float16_profile_for_fp16(self, mock_create, mock_run):
    """FP16 experiments should get profiles=['float16_default'] to override FP8 auto-selection."""
    mock_create.return_value = MagicMock()
    mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

    adapter = AIConfiguratorEstimateAdapter()
    exp = _make_experiment()
    exp.precision = "FP16"
    adapter.run(exp)

    _, kwargs = mock_create.call_args
    assert kwargs["profiles"] == ["float16_default"]

@patch("experiment.adapters.aiconfigurator_est._run_task")
@patch("experiment.adapters.aiconfigurator_est._create_task_config")
def test_run_passes_no_profile_for_fp8(self, mock_create, mock_run):
    """FP8 experiments should let default auto-selection work (no profile override)."""
    mock_create.return_value = MagicMock()
    mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

    adapter = AIConfiguratorEstimateAdapter()
    exp = _make_experiment()
    exp.precision = "FP8"
    adapter.run(exp)

    _, kwargs = mock_create.call_args
    assert kwargs["profiles"] == []

@patch("experiment.adapters.aiconfigurator_est._run_task")
@patch("experiment.adapters.aiconfigurator_est._create_task_config")
def test_run_uses_hardware_system_name(self, mock_create, mock_run):
    """system_name should come from hardware mapping."""
    mock_create.return_value = MagicMock()
    mock_run.return_value = {"pareto_df": _make_pareto_df(), "pareto_frontier_df": None}

    adapter = AIConfiguratorEstimateAdapter()
    exp = _make_experiment()
    adapter.run(exp)

    _, kwargs = mock_create.call_args
    assert kwargs["system_name"] == "h100_sxm"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_aiconfigurator_adapter.py -v -k "non_h100 or mixtral_8x22b or scout or profile or system_name"`
Expected: FAIL

- [ ] **Step 3: Implement changes**

In `experiment/adapters/aiconfigurator_est.py`:

1. Add hardware mapping:
```python
_HW_TO_AICONFIG: dict[str, str] = {"H100": "h100_sxm"}
```

2. Fix `_MOE_MODELS`:
```python
_MOE_MODELS: frozenset[str] = frozenset({
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
})
```

3. Update `can_run`:
```python
def can_run(self, experiment: Experiment) -> bool:
    if experiment.model in _MOE_MODELS:
        return False
    if experiment.hardware not in _HW_TO_AICONFIG:
        return False
    # ... existing profile_config check ...
```

4. Update `run` — hardware mapping + precision profiles:
```python
system_name = _HW_TO_AICONFIG[experiment.hardware]
profiles = ["float16_default"] if experiment.precision == "FP16" else []

task_config = _create_task_config(
    serving_mode="agg",
    model_name=model_name,
    system_name=system_name,
    backend_name="vllm",
    total_gpus=experiment.tp,
    isl=input_length,
    osl=output_length,
    ttft=5000.0,
    tpot=200.0,
    profiles=profiles,
)
```

- [ ] **Step 4: Update existing `test_task_config_args` test**

The existing test at `tests/test_aiconfigurator_adapter.py:262-283` uses `assert_called_once_with` with exact kwargs. After adding `profiles` and `system_name` from hardware mapping, this test will fail. Update the expected kwargs:

```python
mock_create.assert_called_once_with(
    serving_mode="agg",
    model_name="LLAMA2_7B",
    system_name="h100_sxm",  # now from _HW_TO_AICONFIG mapping
    backend_name="vllm",
    total_gpus=1,
    isl=566,
    osl=247,
    ttft=5000.0,
    tpot=200.0,
    profiles=["float16_default"],  # NEW: default _make_experiment has precision="FP16"
)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_aiconfigurator_adapter.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add experiment/adapters/aiconfigurator_est.py tests/test_aiconfigurator_adapter.py
git commit -m "feat(aiconfigurator): hardware mapping, MoE fix, precision via profiles"
```

---

## Task 8: CSV schema extension — metadata in ErrorRecord and RuntimeRecord

**Files:**
- Modify: `experiment/metrics.py:26-51`
- Modify: `experiment/report.py:122-143`, `195-209`
- Modify: `experiment/run.py` (RuntimeRecord creation)
- Test: `tests/test_metrics.py`, `tests/test_report.py`

- [ ] **Step 1: Write failing tests for new ErrorRecord fields**

```python
def test_error_record_has_metadata_fields():
    rec = ErrorRecord(
        simulator="vidur", experiment_folder="/tmp/exp", model="m", workload="general",
        stage_index=0, metric_name="e2e_mean", predicted=100.0, actual=110.0,
        mape=9.09, mpe=-9.09, absolute_error=10.0,
        exp_id=13, hardware="H100", dp=None,
        cpu_offload=False, gpu_mem_util=0.9, precision="FP16",
    )
    assert rec.exp_id == 13
    assert rec.hardware == "H100"

def test_runtime_record_has_metadata_fields():
    rec = RuntimeRecord(
        simulator="vidur", experiment_folder="/tmp/exp", model="m", workload="general",
        wall_clock_seconds=1.5,
        exp_id=13, hardware="H100", dp=None,
        cpu_offload=False, gpu_mem_util=0.9, precision="FP16",
    )
    assert rec.exp_id == 13
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v -k "metadata_fields"`
Expected: FAIL

- [ ] **Step 3: Add fields to ErrorRecord and RuntimeRecord**

In `experiment/metrics.py`, add to both dataclasses:

```python
exp_id: int = 0
hardware: str = "H100"
dp: int | None = None
cpu_offload: bool = False
gpu_mem_util: float = 0.9
precision: str = "FP16"
```

- [ ] **Step 4: Update `_compare_stages` to propagate metadata**

In `experiment/metrics.py`, update `_compare_stages` to pass metadata from experiment:

```python
records.append(ErrorRecord(
    # ... existing fields ...
    exp_id=experiment.exp_id,
    hardware=experiment.hardware,
    dp=experiment.dp,
    cpu_offload=experiment.cpu_offload,
    gpu_mem_util=experiment.gpu_mem_util,
    precision=experiment.precision,
))
```

- [ ] **Step 5: Update `run_pipeline` RuntimeRecord creation**

In `experiment/run.py`, update RuntimeRecord creation:

```python
runtime_records.append(RuntimeRecord(
    simulator=adapter_name,
    experiment_folder=exp.folder,
    model=exp.model,
    workload=exp.workload,
    wall_clock_seconds=elapsed,
    exp_id=exp.exp_id,
    hardware=exp.hardware,
    dp=exp.dp,
    cpu_offload=exp.cpu_offload,
    gpu_mem_util=exp.gpu_mem_util,
    precision=exp.precision,
))
```

- [ ] **Step 6: Update CSV writers in `experiment/report.py`**

Add new fieldnames to `save_csv` and `save_runtime_csv`:

```python
# In save_csv fieldnames:
"exp_id", "hardware", "dp", "cpu_offload", "gpu_mem_util", "precision",

# In save_runtime_csv fieldnames:
"exp_id", "hardware", "dp", "cpu_offload", "gpu_mem_util", "precision",
```

And include them in the `writer.writerow` dicts.

- [ ] **Step 7: Write test for CSV output columns**

```python
def test_csv_includes_metadata_columns(tmp_path):
    records = [ErrorRecord(
        simulator="vidur", experiment_folder="/tmp/exp", model="m", workload="general",
        stage_index=0, metric_name="e2e_mean", predicted=100.0, actual=110.0,
        mape=9.09, mpe=-9.09, absolute_error=10.0,
        exp_id=13, hardware="A100-80GB", dp=2,
        cpu_offload=True, gpu_mem_util=0.95, precision="FP8",
    )]
    csv_path = str(tmp_path / "error_records.csv")
    save_csv(records, csv_path)

    import csv
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
        assert row["exp_id"] == "13"
        assert row["hardware"] == "A100-80GB"
        assert row["dp"] == "2"
        assert row["precision"] == "FP8"
```

- [ ] **Step 8: Run all tests**

Run: `pytest tests/test_metrics.py tests/test_report.py tests/test_run.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add experiment/metrics.py experiment/report.py experiment/run.py tests/test_metrics.py tests/test_report.py
git commit -m "feat(metrics,report): add experiment metadata to CSV output"
```

---

## Task 9: Full integration test

**Files:**
- Test: `tests/test_integration.py` or `tests/test_run.py`

- [ ] **Step 1: Run the full existing test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Verify against real data (manual smoke test)**

Run the pipeline against the actual ground truth directory to check discovery works:

```bash
python -c "
from experiment.ground_truth import discover_experiments
results = discover_experiments('vllm_data/ground_truth')
print(f'Discovered {len(results)} safe experiments')
for entry, path in results[:5]:
    print(f'  ID {entry[\"id\"]}: {entry[\"model\"]} on {entry[\"hw\"]} ({entry[\"workload\"]})')
"
```

Expected: `Discovered 49 safe experiments`

- [ ] **Step 3: Commit any final fixes**

```bash
git add tests/
git commit -m "test: verify full pipeline integration with manifest discovery"
```

---

## Limitations Reference

The following are **true simulator limitations** that this implementation cannot fix. The adapter code should be aware of these but does not attempt to work around them:

| Limitation | Vidur | LLM-Optimizer | AIConfigurator |
|------------|:-----:|:-------------:|:--------------:|
| `cpu_offload` not modeled | Yes | Yes | Yes |
| `gpu_mem` not modeled | Yes | Yes | Yes |
| `mbt` not modeled | — | Yes | Yes |
| `dp` not modeled | — | Yes | Yes |
| No MoE | Yes | Inaccurate* | Excluded |
| No FP8 | Yes | — | — |
| No L40S | Yes | Yes | Yes |
| No chunked prefill | Yes | — | — |
| Mean metrics only | — | Yes | Yes |

*MoE runs but with dense-only roofline (d_ff = 4 * d_model)

See `docs/simulator-limitations.md` for full details with affected experiment IDs.
