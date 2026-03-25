# LLMServingSim Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date**: 2026-03-24
**Spec**: docs/superpowers/specs/2026-03-24-llmservingsim-adapter-design.md
**Status**: Ready for Implementation

**Goal:** Implement LLMServingSim adapter to validate simulator predictions against vLLM ground-truth experiments

**Architecture:** Adapter follows SimulatorAdapter interface, generates cluster configs and workloads from ground-truth data, executes LLMServingSim via subprocess, parses CSV results into SimulatorResult with nested LatencyDistribution and ThroughputMetrics

**Tech Stack:** Python 3.10+, LLMServingSim (subprocess), pytest, numpy (percentiles)

---

## File Structure

**Create:**
- `experiment/adapters/llmservingsim.py` - Main adapter implementation (~500 lines)
- `tests/test_llmservingsim_adapter.py` - Unit and integration tests (~700 lines)

**Modify:**
- `experiment/adapters/__init__.py:1-5` - Add LLMServingSimAdapter export (preserve existing)
- `experiment/run.py:30-38` - Add "llmservingsim" to ALL_ADAPTER_NAMES list
- `experiment/run.py:10-15` - Add --llmservingsim-dir CLI argument
- `experiment/run.py:41-62` - Update build_adapter_registry signature and factories
- `experiment/run.py:64-71` - Update run_pipeline signature
- `experiment/run.py:bottom` - Update main() to pass llmservingsim_dir

---

## Task 1: Adapter Skeleton and Model Mapping

**Files:**
- Create: `experiment/adapters/llmservingsim.py`
- Test: `tests/test_llmservingsim_adapter.py`

- [ ] **Step 1: Write test for MODEL_MAP constant**

```python
# tests/test_llmservingsim_adapter.py
import pytest
from experiment.adapters.llmservingsim import MODEL_MAP


def test_model_map_llama():
    """Test Llama model mapping strips -Instruct suffix"""
    assert MODEL_MAP["meta-llama/Llama-3.1-8B-Instruct"] == "meta-llama/Llama-3.1-8B"


def test_model_map_mixtral():
    """Test Mixtral model mapping remains unchanged"""
    assert MODEL_MAP["mistralai/Mixtral-8x7B-v0.1"] == "mistralai/Mixtral-8x7B-v0.1"


def test_model_map_coverage():
    """Test MODEL_MAP contains exactly 2 supported models"""
    assert len(MODEL_MAP) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llmservingsim_adapter.py::test_model_map_llama -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'experiment.adapters.llmservingsim'"

- [ ] **Step 3: Create adapter skeleton with MODEL_MAP**

```python
# experiment/adapters/llmservingsim.py
"""LLMServingSim adapter for validation against vLLM ground-truth experiments."""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiment.data_model import Experiment, SimulatorResult

from experiment.adapters.base import SimulatorAdapter


# Map ground-truth model IDs (with suffixes) to LLMServingSim perf model names (without suffixes)
MODEL_MAP = {
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B",
    "mistralai/Mixtral-8x7B-v0.1": "mistralai/Mixtral-8x7B-v0.1",
}


class LLMServingSimAdapter(SimulatorAdapter):
    """Adapter for LLMServingSim simulator.

    Supports H100 hardware with Llama-3.1-8B and Mixtral-8x7B models.
    Generates workloads from ground-truth token counts with constant-rate arrivals.
    """

    def __init__(self, llmservingsim_dir: str):
        """Initialize adapter.

        Args:
            llmservingsim_dir: Path to LLMServingSim directory containing main.py
        """
        self.llmservingsim_dir = os.path.abspath(llmservingsim_dir)
        if not os.path.exists(os.path.join(self.llmservingsim_dir, "main.py")):
            raise ValueError(
                f"Invalid LLMServingSim directory: {llmservingsim_dir}. "
                "Must contain main.py"
            )

    @property
    def name(self) -> str:
        """Return adapter name for identification."""
        return "llmservingsim"

    def can_run(self, experiment: "Experiment") -> bool:
        """Check if experiment is runnable."""
        # TODO: implement eligibility checks
        return False

    def run(self, experiment: "Experiment") -> "SimulatorResult":
        """Run simulation and return results."""
        # TODO: implement main execution flow
        raise NotImplementedError("run() not yet implemented")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llmservingsim_adapter.py::test_model_map_llama -v`
Expected: PASS

- [ ] **Step 5: Commit skeleton**

```bash
git add experiment/adapters/llmservingsim.py tests/test_llmservingsim_adapter.py
git commit -m "feat: add LLMServingSim adapter skeleton with MODEL_MAP

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Eligibility Checking (can_run)

**Files:**
- Modify: `experiment/adapters/llmservingsim.py:35-50`
- Test: `tests/test_llmservingsim_adapter.py`

- [ ] **Step 1: Write test for can_run() with supported config**

```python
# tests/test_llmservingsim_adapter.py
import json
from pathlib import Path
from experiment.data_model import Experiment, LatencyDistribution, StageMetrics, ThroughputMetrics
from experiment.adapters.llmservingsim import LLMServingSimAdapter


def _make_experiment(**overrides):
    """Helper to create Experiment with all required fields."""
    zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)
    zero_stage = StageMetrics(
        stage_index=-1,
        rate=0.0,
        duration=0.0,
        num_requests=0,
        e2e=zero_lat,
        ttft=zero_lat,
        itl=zero_lat,
        throughput=ThroughputMetrics(
            input_tokens_per_sec=0.0,
            output_tokens_per_sec=0.0,
            requests_per_sec=0.0
        ),
    )
    defaults = {
        "folder": "dummy",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "tp": 1,
        "workload": "general",
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "total_kv_blocks": 1000,
        "cpu_kv_blocks": 0,
        "stages": [],
        "summary": zero_stage,
        "profile_config": {"load": {"stages": []}},
        "hardware": "H100",
        "precision": "FP16",
        "dp": 1,
        "cpu_offload": False,
    }
    defaults.update(overrides)
    return Experiment(**defaults)


@pytest.fixture
def adapter(tmp_path):
    """Create adapter with mock LLMServingSim directory"""
    llm_dir = tmp_path / "LLMServingSim"
    llm_dir.mkdir()
    (llm_dir / "main.py").touch()

    # Create mock perf model directories
    perf_base = llm_dir / "llm_profile" / "perf_models" / "H100"
    (perf_base / "meta-llama/Llama-3.1-8B" / "tp1").mkdir(parents=True)
    (perf_base / "meta-llama/Llama-3.1-8B" / "tp2").mkdir(parents=True)
    (perf_base / "mistralai/Mixtral-8x7B-v0.1" / "tp1").mkdir(parents=True)
    (perf_base / "mistralai/Mixtral-8x7B-v0.1" / "tp2").mkdir(parents=True)
    (perf_base / "mistralai/Mixtral-8x7B-v0.1" / "tp4").mkdir(parents=True)

    # Create mock cluster config template
    config_dir = llm_dir / "cluster_config"
    config_dir.mkdir()
    template = {
        "nodes": [{
            "num_instances": 1,
            "instances": [{
                "model_name": "placeholder",
                "npu_num": 1,
                "npu_group": 1,
                "npu_mem": {"mem_size": 40.0, "mem_bw": 3350, "mem_latency": 0},
            }]
        }]
    }
    with open(config_dir / "single_node_single_instance_H100.json", "w") as f:
        json.dump(template, f)

    return LLMServingSimAdapter(str(llm_dir))


def test_can_run_llama_h100_tp1(adapter):
    """Test can_run returns True for supported Llama config"""
    exp = _make_experiment(
        model="meta-llama/Llama-3.1-8B-Instruct",
        hardware="H100",
        tp=1,
        precision="FP16",
    )
    assert adapter.can_run(exp) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llmservingsim_adapter.py::test_can_run_llama_h100_tp1 -v`
Expected: FAIL with "AssertionError: assert False is True"

- [ ] **Step 3: Implement can_run() logic**

```python
# experiment/adapters/llmservingsim.py
def can_run(self, experiment: "Experiment") -> bool:
    """Check if experiment is runnable on LLMServingSim.

    Returns True only if:
    - Hardware is H100
    - Model is in MODEL_MAP
    - Performance model exists for the TP configuration
    - Precision is FP16
    """
    # Check hardware
    if experiment.hardware != "H100":
        return False

    # Check model mapping
    model_sim = MODEL_MAP.get(experiment.model)
    if not model_sim:
        return False

    # Check perf model exists
    perf_model_path = os.path.join(
        self.llmservingsim_dir,
        f"llm_profile/perf_models/H100/{model_sim}/tp{experiment.tp}"
    )
    if not os.path.exists(perf_model_path):
        return False

    # Check precision
    if experiment.precision != "FP16":
        return False

    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llmservingsim_adapter.py::test_can_run_llama_h100_tp1 -v`
Expected: PASS

- [ ] **Step 5: Write tests for unsupported configs**

```python
# tests/test_llmservingsim_adapter.py
def test_can_run_unsupported_hardware(adapter):
    """Test can_run returns False for non-H100 hardware"""
    exp = _make_experiment(hardware="A100-80GB")
    assert adapter.can_run(exp) is False


def test_can_run_unsupported_model(adapter):
    """Test can_run returns False for unsupported model"""
    exp = _make_experiment(model="codellama/CodeLlama-34b-Instruct-hf")
    assert adapter.can_run(exp) is False


def test_can_run_unsupported_precision(adapter):
    """Test can_run returns False for FP8 precision"""
    exp = _make_experiment(precision="FP8")
    assert adapter.can_run(exp) is False


def test_can_run_missing_tp_config(adapter):
    """Test can_run returns False when TP config doesn't exist"""
    exp = _make_experiment(tp=8)  # tp8 doesn't exist in mock
    assert adapter.can_run(exp) is False
```

- [ ] **Step 6: Run tests to verify they all pass**

Run: `pytest tests/test_llmservingsim_adapter.py -k can_run -v`
Expected: All PASS

- [ ] **Step 7: Commit eligibility checking**

```bash
git add experiment/adapters/llmservingsim.py tests/test_llmservingsim_adapter.py
git commit -m "feat: implement eligibility checking in LLMServingSim adapter

- Check H100 hardware, MODEL_MAP, perf model existence, FP16 precision
- Add comprehensive tests for supported and unsupported configs
- Include test helper _make_experiment() with proper data model

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Cluster Config Generator

**Files:**
- Modify: `experiment/adapters/llmservingsim.py:70-150`
- Test: `tests/test_llmservingsim_adapter.py`

- [ ] **Step 1: Write test for single-instance cluster config generation**

```python
# tests/test_llmservingsim_adapter.py
def test_generate_cluster_config_single_instance(adapter, tmp_path):
    """Test cluster config generation for single-instance (dp=1)"""
    exp = _make_experiment(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tp=2,
        dp=1,
    )

    output_path = tmp_path / "cluster.json"
    adapter._generate_cluster_config(exp, str(output_path))

    with open(output_path) as f:
        config = json.load(f)

    # Check model name
    assert config["nodes"][0]["instances"][0]["model_name"] == "meta-llama/Llama-3.1-8B"

    # Check TP config
    assert config["nodes"][0]["instances"][0]["npu_num"] == 2
    assert config["nodes"][0]["instances"][0]["npu_group"] == 1

    # Check GPU memory
    assert config["nodes"][0]["instances"][0]["npu_mem"]["mem_size"] == 80.0

    # Check single instance
    assert config["nodes"][0]["num_instances"] == 1
    assert len(config["nodes"][0]["instances"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llmservingsim_adapter.py::test_generate_cluster_config_single_instance -v`
Expected: FAIL with "AttributeError: 'LLMServingSimAdapter' object has no attribute '_generate_cluster_config'"

- [ ] **Step 3: Implement _generate_cluster_config() for single instance**

```python
# experiment/adapters/llmservingsim.py
import json
import copy


def _generate_cluster_config(self, experiment: "Experiment", output_path: str) -> None:
    """Generate cluster config JSON for the experiment.

    Args:
        experiment: Experiment configuration
        output_path: Where to write the cluster config JSON
    """
    # Load H100 template
    template_path = os.path.join(
        self.llmservingsim_dir,
        "cluster_config/single_node_single_instance_H100.json"
    )
    with open(template_path) as f:
        config = json.load(f)

    # Get LLMServingSim model name
    model_sim = MODEL_MAP[experiment.model]

    # Modify instance config
    instance = config["nodes"][0]["instances"][0]
    instance["model_name"] = model_sim
    instance["npu_num"] = experiment.tp
    instance["npu_group"] = 1  # npus_per_group = npu_num / npu_group = TP
    instance["npu_mem"]["mem_size"] = 80.0  # H100 HBM3

    # Write config
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llmservingsim_adapter.py::test_generate_cluster_config_single_instance -v`
Expected: PASS

- [ ] **Step 5: Write test for multi-instance cluster config**

```python
# tests/test_llmservingsim_adapter.py
def test_generate_cluster_config_multi_instance(adapter, tmp_path):
    """Test cluster config generation for multi-instance (dp>1)"""
    exp = _make_experiment(
        model="mistralai/Mixtral-8x7B-v0.1",
        tp=4,
        dp=2,
    )

    output_path = tmp_path / "cluster.json"
    adapter._generate_cluster_config(exp, str(output_path))

    with open(output_path) as f:
        config = json.load(f)

    # Check multi-instance setup
    assert config["nodes"][0]["num_instances"] == 2
    assert len(config["nodes"][0]["instances"]) == 2

    # Check both instances have correct config
    for instance in config["nodes"][0]["instances"]:
        assert instance["model_name"] == "mistralai/Mixtral-8x7B-v0.1"
        assert instance["npu_num"] == 4
        assert instance["npu_group"] == 1
        assert instance["npu_mem"]["mem_size"] == 80.0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_llmservingsim_adapter.py::test_generate_cluster_config_multi_instance -v`
Expected: FAIL with "AssertionError: assert 1 == 2"

- [ ] **Step 7: Add multi-instance logic to _generate_cluster_config()**

```python
# experiment/adapters/llmservingsim.py
def _generate_cluster_config(self, experiment: "Experiment", output_path: str) -> None:
    """Generate cluster config JSON for the experiment.

    Args:
        experiment: Experiment configuration
        output_path: Where to write the cluster config JSON
    """
    # Load H100 template
    template_path = os.path.join(
        self.llmservingsim_dir,
        "cluster_config/single_node_single_instance_H100.json"
    )
    with open(template_path) as f:
        config = json.load(f)

    # Get LLMServingSim model name
    model_sim = MODEL_MAP[experiment.model]

    # Modify instance config
    instance = config["nodes"][0]["instances"][0]
    instance["model_name"] = model_sim
    instance["npu_num"] = experiment.tp
    instance["npu_group"] = 1  # npus_per_group = npu_num / npu_group = TP
    instance["npu_mem"]["mem_size"] = 80.0  # H100 HBM3

    # Handle multi-instance (dp > 1) - use deep copy to avoid shared dicts
    if experiment.dp and experiment.dp > 1:
        config["nodes"][0]["num_instances"] = experiment.dp
        config["nodes"][0]["instances"] = [
            copy.deepcopy(instance) for _ in range(experiment.dp)
        ]

    # Write config
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_llmservingsim_adapter.py::test_generate_cluster_config_multi_instance -v`
Expected: PASS

- [ ] **Step 9: Commit cluster config generator**

```bash
git add experiment/adapters/llmservingsim.py tests/test_llmservingsim_adapter.py
git commit -m "feat: implement cluster config generator

- Generate single-instance configs (dp=1)
- Generate multi-instance configs (dp>1) with deep copy
- Use H100 template with correct TP and memory settings

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Workload Generator

**Files:**
- Modify: `experiment/adapters/llmservingsim.py:152-250`
- Test: `tests/test_llmservingsim_adapter.py`

- [ ] **Step 1: Write test for constant-rate arrival generation**

```python
# tests/test_llmservingsim_adapter.py
def test_generate_arrivals_single_stage():
    """Test constant-rate arrival times for single stage"""
    from experiment.adapters.llmservingsim import _generate_arrivals

    stages = [{"rate": 10, "duration": 5}]  # 10 req/s for 5 seconds
    arrivals = _generate_arrivals(stages)

    # Should have 50 requests
    assert len(arrivals) == 50

    # Check uniform spacing (0.1s = 100ms)
    for i in range(1, len(arrivals)):
        spacing = arrivals[i] - arrivals[i-1]
        assert abs(spacing - 0.1) < 1e-9  # Constant 100ms spacing


def test_generate_arrivals_multi_stage():
    """Test constant-rate arrivals for multiple stages"""
    from experiment.adapters.llmservingsim import _generate_arrivals

    stages = [
        {"rate": 8, "duration": 2},   # 16 requests, 0-2s
        {"rate": 12, "duration": 3},  # 36 requests, 2-5s
    ]
    arrivals = _generate_arrivals(stages)

    assert len(arrivals) == 52

    # Check stage 1 arrivals (0-2s)
    stage1 = [a for a in arrivals if a < 2.0]
    assert len(stage1) == 16
    for i in range(1, len(stage1)):
        assert abs((stage1[i] - stage1[i-1]) - 0.125) < 1e-9  # 1/8 = 0.125s

    # Check stage 2 arrivals (2-5s)
    stage2 = [a for a in arrivals if a >= 2.0]
    assert len(stage2) == 36
    for i in range(1, len(stage2)):
        assert abs((stage2[i] - stage2[i-1]) - (1.0/12)) < 1e-9  # 1/12 = 0.0833s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llmservingsim_adapter.py::test_generate_arrivals_single_stage -v`
Expected: FAIL with "ImportError: cannot import name '_generate_arrivals'"

- [ ] **Step 3: Implement _generate_arrivals() helper**

```python
# experiment/adapters/llmservingsim.py
def _generate_arrivals(stages: list[dict]) -> list[float]:
    """Generate constant-rate arrival times from stage config.

    Args:
        stages: List of {"rate": req/s, "duration": seconds}

    Returns:
        List of arrival times in seconds
    """
    arrivals = []
    t = 0.0

    for stage in stages:
        rate = stage["rate"]
        duration = stage["duration"]
        num_requests = round(rate * duration)
        inter_arrival = 1.0 / rate

        for _ in range(num_requests):
            arrivals.append(t)
            t += inter_arrival

    return arrivals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llmservingsim_adapter.py -k generate_arrivals -v`
Expected: All PASS

- [ ] **Step 5: Write test for workload file generation**

```python
# tests/test_llmservingsim_adapter.py
def test_generate_workload_file(adapter, tmp_path):
    """Test workload .jsonl generation from ground-truth"""
    # Create mock ground-truth data
    gt_dir = tmp_path / "test-exp"
    results_dir = gt_dir / "results"
    results_dir.mkdir(parents=True)

    metrics = [
        {"info": {"input_tokens": 100, "output_tokens": 50}},
        {"info": {"input_tokens": 120, "output_tokens": 60}},
        {"info": {"input_tokens": 110, "output_tokens": 55}},
    ]
    with open(results_dir / "per_request_lifecycle_metrics.json", "w") as f:
        json.dump(metrics, f)

    exp = _make_experiment(
        folder=str(gt_dir),
        profile_config={
            "load": {
                "stages": [{"rate": 1, "duration": 3}]  # 3 requests
            }
        },
    )

    output_path = tmp_path / "workload.jsonl"
    adapter._generate_workload(exp, str(output_path))

    # Read generated workload
    with open(output_path) as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 3

    # Check first request
    assert lines[0]["input_toks"] == 100
    assert lines[0]["output_toks"] == 50
    assert lines[0]["arrival_time_ns"] == 0
    assert len(lines[0]["input_tok_ids"]) == 100

    # Check arrivals are spaced at 1s intervals
    assert lines[1]["arrival_time_ns"] == 1_000_000_000
    assert lines[2]["arrival_time_ns"] == 2_000_000_000
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_llmservingsim_adapter.py::test_generate_workload_file -v`
Expected: FAIL with "AttributeError: 'LLMServingSimAdapter' object has no attribute '_generate_workload'"

- [ ] **Step 7: Implement _generate_workload() method**

```python
# experiment/adapters/llmservingsim.py
def _generate_workload(self, experiment: "Experiment", output_path: str) -> None:
    """Generate workload .jsonl file from ground-truth token counts.

    Args:
        experiment: Experiment configuration
        output_path: Where to write the workload .jsonl
    """
    # Import resolve_perf_dir from correct location
    from experiment.ground_truth import resolve_perf_dir

    # Read ground-truth metrics
    perf_dir = resolve_perf_dir(experiment.folder)
    metrics_path = os.path.join(perf_dir, "per_request_lifecycle_metrics.json")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Ground-truth metrics not found: {metrics_path}. "
            "Cannot generate workload without token counts."
        )

    with open(metrics_path) as f:
        requests = json.load(f)

    # Extract token counts
    token_pairs = [
        (req["info"]["input_tokens"], req["info"]["output_tokens"])
        for req in requests
    ]

    # Generate constant-rate arrivals
    stages = experiment.profile_config["load"]["stages"]
    arrivals = _generate_arrivals(stages)

    # Check for mismatch
    if len(token_pairs) < len(arrivals):
        raise ValueError(
            f"Not enough ground-truth requests ({len(token_pairs)}) "
            f"for generated arrivals ({len(arrivals)})"
        )

    # Write workload .jsonl
    with open(output_path, "w") as f:
        for (input_toks, output_toks), arrival_sec in zip(token_pairs, arrivals):
            # Generate dummy token IDs (LLMServingSim only needs counts)
            input_tok_ids = list(range(1, input_toks + 1))

            record = {
                "input_toks": input_toks,
                "output_toks": output_toks,
                "arrival_time_ns": int(arrival_sec * 1e9),
                "input_tok_ids": input_tok_ids,
            }
            f.write(json.dumps(record) + "\n")
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_llmservingsim_adapter.py::test_generate_workload_file -v`
Expected: PASS

- [ ] **Step 9: Commit workload generator**

```bash
git add experiment/adapters/llmservingsim.py tests/test_llmservingsim_adapter.py
git commit -m "feat: implement workload generator

- Generate constant-rate arrival times from stage config
- Read token counts from ground-truth per_request_lifecycle_metrics.json
- Import resolve_perf_dir from experiment.ground_truth (not .resolve)
- Write LLMServingSim-compatible .jsonl workload

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: CLI Arguments Builder

**Files:**
- Modify: `experiment/adapters/llmservingsim.py:252-290`
- Test: `tests/test_llmservingsim_adapter.py`

- [ ] **Step 1: Write test for CLI args generation**

```python
# tests/test_llmservingsim_adapter.py
def test_build_cli_args_single_instance(adapter):
    """Test CLI args for single-instance experiment"""
    exp = _make_experiment(
        profile_config={
            "load": {"stages": [{"rate": 10, "duration": 5}]}
        },
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        dp=1,
    )

    args = adapter._build_cli_args(
        exp,
        cluster_config="/path/to/cluster.json",
        workload="/path/to/workload.jsonl",
        output="/path/to/output.csv"
    )

    assert args[0:2] == ["python", "main.py"]
    assert "--cluster-config" in args
    assert "/path/to/cluster.json" in args
    assert "--dataset" in args
    assert "/path/to/workload.jsonl" in args
    assert "--output" in args
    assert "/path/to/output.csv" in args
    assert "--fp" in args
    assert "16" in args
    assert "--block-size" in args
    assert "16" in args
    assert "--max-batch" in args
    assert "256" in args
    assert "--max-num-batched-tokens" in args
    assert "8192" in args
    assert "--num-req" in args
    assert "50" in args  # 10 req/s * 5s = 50

    # Should NOT have routing policy for single instance
    assert "--request-routing-policy" not in args


def test_build_cli_args_multi_instance(adapter):
    """Test CLI args for multi-instance experiment"""
    exp = _make_experiment(
        profile_config={
            "load": {"stages": [{"rate": 8, "duration": 3}]}
        },
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        dp=3,
    )

    args = adapter._build_cli_args(
        exp,
        cluster_config="/path/to/cluster.json",
        workload="/path/to/workload.jsonl",
        output="/path/to/output.csv"
    )

    # Should have routing policy for multi-instance
    assert "--request-routing-policy" in args
    assert "RR" in args
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llmservingsim_adapter.py::test_build_cli_args_single_instance -v`
Expected: FAIL with "AttributeError: 'LLMServingSimAdapter' object has no attribute '_build_cli_args'"

- [ ] **Step 3: Implement _build_cli_args() method**

```python
# experiment/adapters/llmservingsim.py
def _build_cli_args(
    self,
    experiment: "Experiment",
    cluster_config: str,
    workload: str,
    output: str,
) -> list[str]:
    """Build LLMServingSim CLI arguments.

    Args:
        experiment: Experiment configuration
        cluster_config: Path to cluster config JSON
        workload: Path to workload .jsonl
        output: Path to output CSV

    Returns:
        List of CLI arguments
    """
    # Calculate total requests
    total_requests = sum(
        round(stage["rate"] * stage["duration"])
        for stage in experiment.profile_config["load"]["stages"]
    )

    args = [
        "python",
        "main.py",
        "--cluster-config",
        cluster_config,
        "--dataset",
        workload,
        "--output",
        output,
        "--fp",
        "16",
        "--block-size",
        "16",
        "--max-batch",
        str(experiment.max_num_seqs),
        "--max-num-batched-tokens",
        str(experiment.max_num_batched_tokens),
        "--num-req",
        str(total_requests),
        "--log-level",
        "WARNING",
    ]

    # Add routing policy for multi-instance
    if experiment.dp and experiment.dp > 1:
        args.extend(["--request-routing-policy", "RR"])

    return args
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llmservingsim_adapter.py -k build_cli_args -v`
Expected: All PASS

- [ ] **Step 5: Commit CLI args builder**

```bash
git add experiment/adapters/llmservingsim.py tests/test_llmservingsim_adapter.py
git commit -m "feat: implement CLI arguments builder

- Generate LLMServingSim command-line args from experiment config
- Add round-robin routing policy for multi-instance (dp>1)
- Calculate total requests from stage config

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Result Parser

**Files:**
- Modify: `experiment/adapters/llmservingsim.py:292-450`
- Test: `tests/test_llmservingsim_adapter.py`

- [ ] **Step 1: Write test for stage splitting logic**

```python
# tests/test_llmservingsim_adapter.py
import csv


def test_split_by_stage_single():
    """Test stage splitting with single stage"""
    from experiment.adapters.llmservingsim import _split_by_stage

    rows = [
        {"arrival": "0", "latency": "1000"},
        {"arrival": "1000000000", "latency": "2000"},  # 1s
        {"arrival": "2000000000", "latency": "3000"},  # 2s
    ]
    stages = [{"rate": 1, "duration": 3}]

    buckets = _split_by_stage(rows, stages)
    assert len(buckets) == 1
    assert len(buckets[0]) == 3


def test_split_by_stage_multi():
    """Test stage splitting with multiple stages"""
    from experiment.adapters.llmservingsim import _split_by_stage

    rows = [
        {"arrival": "0", "latency": "1000"},
        {"arrival": "500000000", "latency": "2000"},   # 0.5s (stage 1)
        {"arrival": "1000000000", "latency": "3000"},  # 1.0s (stage 1)
        {"arrival": "2000000001", "latency": "4000"},  # 2.0s + 1ns (stage 2)
        {"arrival": "3000000000", "latency": "5000"},  # 3.0s (stage 2)
    ]
    stages = [
        {"rate": 2, "duration": 2},  # 0-2s
        {"rate": 1, "duration": 2},  # 2-4s
    ]

    buckets = _split_by_stage(rows, stages)
    assert len(buckets) == 2
    assert len(buckets[0]) == 3  # arrivals at 0, 0.5, 1s
    assert len(buckets[1]) == 2  # arrivals at 2s+1ns, 3s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llmservingsim_adapter.py::test_split_by_stage_single -v`
Expected: FAIL with "ImportError: cannot import name '_split_by_stage'"

- [ ] **Step 3: Implement _split_by_stage() helper**

```python
# experiment/adapters/llmservingsim.py
def _split_by_stage(rows: list[dict], stages: list[dict]) -> list[list[dict]]:
    """Split CSV rows by stage based on arrival time.

    Args:
        rows: CSV rows as dicts
        stages: Stage config with rate/duration

    Returns:
        List of row buckets, one per stage
    """
    # Calculate stage boundaries in nanoseconds
    boundaries = []
    cumulative_time = 0.0
    for stage in stages:
        cumulative_time += stage["duration"]
        boundaries.append(cumulative_time * 1e9)

    # Bucket rows by stage
    buckets = [[] for _ in stages]
    for row in rows:
        arrival_ns = float(row["arrival"])
        for i, boundary_ns in enumerate(boundaries):
            if arrival_ns < boundary_ns:  # Strict < to assign boundary to next stage
                buckets[i].append(row)
                break
        else:
            # Fallback to last stage if beyond all boundaries
            buckets[-1].append(row)

    return buckets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llmservingsim_adapter.py -k split_by_stage -v`
Expected: All PASS

- [ ] **Step 5: Write test for result parsing**

```python
# tests/test_llmservingsim_adapter.py
import numpy as np


def test_parse_results_single_stage(adapter, tmp_path):
    """Test result parsing for single-stage experiment"""
    # Create mock CSV output
    csv_path = tmp_path / "output.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance id", "request id", "model", "input", "output",
            "arrival", "end_time", "latency", "queuing_delay",
            "TTFT", "TPOT", "ITL"
        ])
        writer.writeheader()
        writer.writerow({
            "instance id": "0",
            "request id": "0",
            "model": "meta-llama/Llama-3.1-8B",
            "input": "100",
            "output": "50",
            "arrival": "0",
            "end_time": "2000000000",
            "latency": "2000000000",  # 2s = 2000ms
            "queuing_delay": "0",
            "TTFT": "100000000",      # 100ms
            "TPOT": "30000000",       # 30ms
            "ITL": "[30000000, 30000000]",
        })
        writer.writerow({
            "instance id": "0",
            "request id": "1",
            "model": "meta-llama/Llama-3.1-8B",
            "input": "120",
            "output": "60",
            "arrival": "1000000000",
            "end_time": "3500000000",
            "latency": "2500000000",  # 2.5s = 2500ms
            "queuing_delay": "0",
            "TTFT": "120000000",      # 120ms
            "TPOT": "35000000",       # 35ms
            "ITL": "[35000000, 35000000]",
        })

    exp = _make_experiment(
        folder="test-exp",
        profile_config={"load": {"stages": [{"rate": 1, "duration": 2}]}},
    )

    result = adapter._parse_results(str(csv_path), exp)

    # Check adapter_name and experiment_folder
    assert result.adapter_name == "llmservingsim"
    assert result.experiment_folder == "test-exp"

    # Check summary metrics (average of 2 requests)
    assert result.summary.e2e.mean == pytest.approx(2250.0)  # (2000 + 2500) / 2
    assert result.summary.ttft.mean == pytest.approx(110.0)  # (100 + 120) / 2
    assert result.summary.itl.mean == pytest.approx(32.5)    # (30 + 35) / 2

    # Check stage metrics
    assert len(result.stages) == 1
    assert result.stages[0].e2e.mean == pytest.approx(2250.0)
    assert result.stages[0].num_requests == 2
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_llmservingsim_adapter.py::test_parse_results_single_stage -v`
Expected: FAIL with "AttributeError: 'LLMServingSimAdapter' object has no attribute '_parse_results'"

- [ ] **Step 7: Implement _parse_results() method**

```python
# experiment/adapters/llmservingsim.py
import csv
import numpy as np
from experiment.data_model import SimulatorResult, StageMetrics, LatencyDistribution, ThroughputMetrics


def _parse_results(self, csv_path: str, experiment: "Experiment") -> "SimulatorResult":
    """Parse LLMServingSim CSV output into SimulatorResult.

    Args:
        csv_path: Path to LLMServingSim output CSV
        experiment: Experiment configuration (for folder, stages)

    Returns:
        SimulatorResult with per-stage and summary metrics
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"LLMServingSim output CSV not found: {csv_path}")

    # Read CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"LLMServingSim output CSV is empty: {csv_path}")

    # Get stage config
    stages_config = experiment.profile_config["load"]["stages"]

    # Split by stage
    buckets = _split_by_stage(rows, stages_config)

    # Compute per-stage metrics
    stage_metrics = []
    for i, bucket in enumerate(buckets):
        stage_metrics.append(self._compute_stage(bucket, i, stages_config[i]))

    # Compute summary (all rows together)
    total_duration = sum(s["duration"] for s in stages_config)
    summary = self._compute_stage(rows, -1, {"rate": 0, "duration": total_duration})

    return SimulatorResult(
        adapter_name=self.name,
        experiment_folder=experiment.folder,
        stages=stage_metrics,
        summary=summary,
    )


def _compute_stage(
    self,
    bucket: list[dict],
    stage_index: int,
    stage_cfg: dict,
) -> StageMetrics:
    """Compute metrics for a stage.

    Args:
        bucket: CSV rows for this stage
        stage_index: Stage number (-1 for summary)
        stage_cfg: Stage config with rate/duration

    Returns:
        StageMetrics with nested LatencyDistribution and ThroughputMetrics
    """
    zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)
    dur = max(1.0, stage_cfg.get("duration", 0))

    if not bucket:
        return StageMetrics(
            stage_index=stage_index,
            rate=float(stage_cfg.get("rate", 0)),
            duration=float(stage_cfg.get("duration", 0)),
            num_requests=0,
            e2e=zero_lat,
            ttft=zero_lat,
            itl=zero_lat,
            throughput=ThroughputMetrics(
                input_tokens_per_sec=0.0,
                output_tokens_per_sec=0.0,
                requests_per_sec=0.0,
            ),
        )

    # Convert ns to ms
    e2e_vals = np.array([float(r["latency"]) / 1e6 for r in bucket])
    ttft_vals = np.array([float(r["TTFT"]) / 1e6 for r in bucket])
    tpot_vals = np.array([float(r["TPOT"]) / 1e6 for r in bucket])

    # Calculate throughput
    input_tokens = sum(int(r["input"]) for r in bucket)
    output_tokens = sum(int(r["output"]) for r in bucket)

    return StageMetrics(
        stage_index=stage_index,
        rate=float(stage_cfg.get("rate", 0)),
        duration=float(stage_cfg.get("duration", 0)),
        num_requests=len(bucket),
        e2e=LatencyDistribution(
            mean=float(np.mean(e2e_vals)),
            p90=float(np.percentile(e2e_vals, 90)),
            p99=float(np.percentile(e2e_vals, 99)),
        ),
        ttft=LatencyDistribution(
            mean=float(np.mean(ttft_vals)),
            p90=float(np.percentile(ttft_vals, 90)),
            p99=float(np.percentile(ttft_vals, 99)),
        ),
        itl=LatencyDistribution(
            mean=float(np.mean(tpot_vals)),
            p90=float(np.percentile(tpot_vals, 90)),
            p99=float(np.percentile(tpot_vals, 99)),
        ),
        throughput=ThroughputMetrics(
            input_tokens_per_sec=input_tokens / dur,
            output_tokens_per_sec=output_tokens / dur,
            requests_per_sec=len(bucket) / dur,
        ),
    )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_llmservingsim_adapter.py::test_parse_results_single_stage -v`
Expected: PASS

- [ ] **Step 9: Commit result parser**

```bash
git add experiment/adapters/llmservingsim.py tests/test_llmservingsim_adapter.py
git commit -m "feat: implement result parser

- Split CSV rows by stage using arrival times (strict < boundary)
- Convert nanoseconds to milliseconds
- Compute StageMetrics with nested LatencyDistribution/ThroughputMetrics
- Return SimulatorResult with adapter_name and experiment_folder
- Follow VidurAdapter pattern for data model construction

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Main run() Orchestration

**Files:**
- Modify: `experiment/adapters/llmservingsim.py:50-65` (run method)
- Test: `tests/test_llmservingsim_adapter.py`

- [ ] **Step 1: Write integration test for run() method**

```python
# tests/test_llmservingsim_adapter.py
import subprocess
from unittest.mock import patch, MagicMock


def test_run_orchestration(adapter, tmp_path):
    """Test full run() orchestration with mocked subprocess"""
    # Setup mock experiment
    gt_dir = tmp_path / "test-exp"
    results_dir = gt_dir / "results"
    results_dir.mkdir(parents=True)

    # Mock ground-truth metrics
    metrics = [
        {"info": {"input_tokens": 100, "output_tokens": 50}},
        {"info": {"input_tokens": 120, "output_tokens": 60}},
    ]
    with open(results_dir / "per_request_lifecycle_metrics.json", "w") as f:
        json.dump(metrics, f)

    exp = _make_experiment(
        folder=str(gt_dir),
        profile_config={
            "load": {"stages": [{"rate": 1, "duration": 2}]}
        },
    )

    # Mock subprocess.run to avoid actual execution
    mock_csv = tmp_path / "mock_output.csv"
    with open(mock_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "instance id", "request id", "model", "input", "output",
            "arrival", "end_time", "latency", "queuing_delay",
            "TTFT", "TPOT", "ITL"
        ])
        writer.writeheader()
        writer.writerow({
            "instance id": "0",
            "request id": "0",
            "model": "meta-llama/Llama-3.1-8B",
            "input": "100",
            "output": "50",
            "arrival": "0",
            "end_time": "2000000000",
            "latency": "2000000000",
            "queuing_delay": "0",
            "TTFT": "100000000",
            "TPOT": "30000000",
            "ITL": "[30000000]",
        })

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        # Patch _parse_results to use mock CSV
        original_parse = adapter._parse_results
        def mock_parse(csv_path, exp):
            return original_parse(str(mock_csv), exp)

        with patch.object(adapter, "_parse_results", side_effect=mock_parse):
            result = adapter.run(exp)

        # Verify subprocess was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0:2] == ["python", "main.py"]
        assert "--cluster-config" in args
        assert "--dataset" in args
        assert "--output" in args

        # Verify result structure
        assert result.adapter_name == "llmservingsim"
        assert result.experiment_folder == str(gt_dir)
        assert result.summary.e2e.mean > 0
        assert result.summary.ttft.mean > 0
        assert len(result.stages) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llmservingsim_adapter.py::test_run_orchestration -v`
Expected: FAIL with "NotImplementedError: run() not yet implemented"

- [ ] **Step 3: Implement run() method**

```python
# experiment/adapters/llmservingsim.py
import tempfile
import subprocess


def run(self, experiment: "Experiment") -> "SimulatorResult":
    """Run LLMServingSim simulation for the experiment.

    Args:
        experiment: Experiment configuration

    Returns:
        SimulatorResult with metrics
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate cluster config
        cluster_config_path = os.path.join(tmpdir, "cluster.json")
        self._generate_cluster_config(experiment, cluster_config_path)

        # Generate workload
        workload_path = os.path.join(tmpdir, "workload.jsonl")
        self._generate_workload(experiment, workload_path)

        # Build CLI args
        output_path = os.path.join(tmpdir, "output.csv")
        args = self._build_cli_args(
            experiment,
            cluster_config_path,
            workload_path,
            output_path,
        )

        # Execute LLMServingSim
        try:
            subprocess.run(
                args,
                capture_output=True,
                check=True,
                cwd=self.llmservingsim_dir,
                timeout=3600,  # 1 hour timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"LLMServingSim timed out after 1 hour for {experiment.folder}"
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLMServingSim failed (rc={exc.returncode}) for {experiment.folder}: {stderr}"
            )

        # Parse results
        return self._parse_results(output_path, experiment)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llmservingsim_adapter.py::test_run_orchestration -v`
Expected: PASS

- [ ] **Step 5: Commit run() orchestration**

```bash
git add experiment/adapters/llmservingsim.py tests/test_llmservingsim_adapter.py
git commit -m "feat: implement run() orchestration

- Coordinate config generation, workload creation, execution, parsing
- Use temp directory for isolation and cleanup
- Handle subprocess errors with detailed messages

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Integration with experiment/run.py

**Files:**
- Modify: `experiment/adapters/__init__.py:1-5`
- Modify: `experiment/run.py:30-38, 10-15, 41-62, 64-71, bottom`

- [ ] **Step 1: Export LLMServingSimAdapter from __init__.py (preserve existing)**

```python
# experiment/adapters/__init__.py
from experiment.adapters.aiconfigurator_est import AIConfiguratorEstimateAdapter
from experiment.adapters.base import BaseBLISAdapter, SimulatorAdapter
from experiment.adapters.llmservingsim import LLMServingSimAdapter

__all__ = [
    "SimulatorAdapter",
    "BaseBLISAdapter",
    "AIConfiguratorEstimateAdapter",
    "LLMServingSimAdapter",
]
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from experiment.adapters import LLMServingSimAdapter; print('OK')"`
Expected: "OK"

- [ ] **Step 3: Commit adapter export**

```bash
git add experiment/adapters/__init__.py
git commit -m "feat: export LLMServingSimAdapter from adapters module

- Preserve existing exports (AIConfiguratorEstimateAdapter, BaseBLISAdapter)
- Add LLMServingSimAdapter to __all__

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Add "llmservingsim" to ALL_ADAPTER_NAMES list**

```python
# experiment/run.py (around line 30-38)
ALL_ADAPTER_NAMES = [
    "blis-blackbox",
    "blis-roofline",
    "blis-crossmodel",
    "blis-trained-roofline",
    "vidur",
    "llm-optimizer-estimate",
    "aiconfigurator-estimate",
    "llmservingsim",  # Add this line
]
```

- [ ] **Step 5: Add --llmservingsim-dir CLI argument**

```python
# experiment/run.py (add after other path arguments, around line 10-15)
# Add in parse_args() function after --vidur-dir

parser.add_argument(
    "--llmservingsim-dir",
    default="LLMServingSim",
    help="Path to LLMServingSim directory containing main.py",
)
```

- [ ] **Step 6: Update build_adapter_registry signature and factories**

```python
# experiment/run.py (around line 41-62)
def build_adapter_registry(
    blis_binary: str,
    vidur_dir: str,
    llmservingsim_dir: str,  # Add this parameter
    adapter_names: list[str] | None = None,
) -> dict[str, SimulatorAdapter]:
    """Build a name → adapter instance mapping.

    Only instantiates adapters listed in *adapter_names* (default: all).
    """
    from experiment.adapters.blis_blackbox import BLISBlackboxAdapter
    from experiment.adapters.blis_crossmodel import BLISCrossModelAdapter
    from experiment.adapters.blis_roofline import BLISRooflineAdapter
    from experiment.adapters.blis_trained_roofline import BLISTrainedRooflineAdapter
    from experiment.adapters.llm_optimizer_est import LLMOptimizerEstimateAdapter
    from experiment.adapters.llmservingsim import LLMServingSimAdapter  # Add import

    factories: dict[str, callable] = {
        "blis-blackbox": lambda: BLISBlackboxAdapter(blis_binary),
        "blis-roofline": lambda: BLISRooflineAdapter(blis_binary),
        "blis-crossmodel": lambda: BLISCrossModelAdapter(blis_binary),
        "blis-trained-roofline": lambda: BLISTrainedRooflineAdapter(blis_binary),
        "vidur": lambda: VidurAdapter(vidur_dir),
        "llm-optimizer-estimate": lambda: LLMOptimizerEstimateAdapter(),
        "aiconfigurator-estimate": lambda: AIConfiguratorEstimateAdapter(),
        "llmservingsim": lambda: LLMServingSimAdapter(llmservingsim_dir),  # Add factory
    }
    if adapter_names is None:
        adapter_names = list(factories.keys())
    return {name: factories[name]() for name in adapter_names if name in factories}
```

- [ ] **Step 7: Update run_pipeline signature**

```python
# experiment/run.py (around line 64-71)
def run_pipeline(
    data_dir: str,
    blis_binary: str,
    vidur_dir: str,
    llmservingsim_dir: str,  # Add parameter
    output_dir: str,
    adapter_names: list[str] | None = None,
    no_dp_scaling: bool = False,
) -> tuple[list[ErrorRecord], list[RuntimeRecord]]:
    """Core pipeline: discover → run → compute errors → report.

    Returns (error_records, runtime_records).
    """
    # ... existing code ...

    adapters = build_adapter_registry(
        blis_binary,
        vidur_dir,
        llmservingsim_dir,  # Pass through
        adapter_names,
    )

    # ... rest of function ...
```

- [ ] **Step 8: Update main() to pass llmservingsim_dir**

```python
# experiment/run.py (at bottom, in main() function)
def main():
    # ... existing parse_args() ...

    run_pipeline(
        data_dir=args.data_dir,
        blis_binary=args.blis_binary,
        vidur_dir=args.vidur_dir,
        llmservingsim_dir=args.llmservingsim_dir,  # Add this line
        output_dir=args.output_dir,
        adapter_names=args.adapters,
        no_dp_scaling=args.no_dp_scaling,
    )
```

- [ ] **Step 9: Run basic smoke test**

Run: `python -m experiment.run --help | grep llmservingsim`
Expected: Should show --llmservingsim-dir argument and llmservingsim in choices

- [ ] **Step 10: Commit integration**

```bash
git add experiment/run.py
git commit -m "feat: integrate LLMServingSim adapter into run pipeline

- Add 'llmservingsim' to ALL_ADAPTER_NAMES list
- Add --llmservingsim-dir CLI argument (default: LLMServingSim)
- Update build_adapter_registry with llmservingsim_dir parameter
- Update run_pipeline signature with llmservingsim_dir
- Thread llmservingsim_dir through main() call

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: End-to-End Integration Test

**Files:**
- Test: `tests/test_llmservingsim_integration.py` (create new file)

- [ ] **Step 1: Write E2E test with real LLMServingSim (if available)**

```python
# tests/test_llmservingsim_integration.py
"""Integration tests for LLMServingSim adapter (requires LLMServingSim installed)."""

import pytest
import os
import json
from experiment.adapters.llmservingsim import LLMServingSimAdapter
from experiment.data_model import (
    Experiment,
    LatencyDistribution,
    StageMetrics,
    ThroughputMetrics,
)


def _make_experiment(**overrides):
    """Helper to create Experiment with all required fields."""
    zero_lat = LatencyDistribution(mean=0.0, p90=0.0, p99=0.0)
    zero_stage = StageMetrics(
        stage_index=-1,
        rate=0.0,
        duration=0.0,
        num_requests=0,
        e2e=zero_lat,
        ttft=zero_lat,
        itl=zero_lat,
        throughput=ThroughputMetrics(
            input_tokens_per_sec=0.0,
            output_tokens_per_sec=0.0,
            requests_per_sec=0.0
        ),
    )
    defaults = {
        "folder": "dummy",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "tp": 1,
        "workload": "general",
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "total_kv_blocks": 1000,
        "cpu_kv_blocks": 0,
        "stages": [],
        "summary": zero_stage,
        "profile_config": {"load": {"stages": []}},
        "hardware": "H100",
        "precision": "FP16",
        "dp": 1,
        "cpu_offload": False,
    }
    defaults.update(overrides)
    return Experiment(**defaults)


@pytest.mark.skipif(
    not os.path.exists("LLMServingSim/main.py"),
    reason="LLMServingSim not installed"
)
def test_e2e_llama_h100_tp1(tmp_path):
    """End-to-end test with real LLMServingSim on small workload"""
    # Create mock ground-truth data
    gt_dir = tmp_path / "llama-h100-tp1"
    results_dir = gt_dir / "results"
    results_dir.mkdir(parents=True)

    # Small workload (5 requests)
    metrics = [
        {"info": {"input_tokens": 10, "output_tokens": 5}},
        {"info": {"input_tokens": 12, "output_tokens": 6}},
        {"info": {"input_tokens": 11, "output_tokens": 5}},
        {"info": {"input_tokens": 13, "output_tokens": 7}},
        {"info": {"input_tokens": 10, "output_tokens": 6}},
    ]
    with open(results_dir / "per_request_lifecycle_metrics.json", "w") as f:
        json.dump(metrics, f)

    exp = _make_experiment(
        folder=str(gt_dir),
        profile_config={
            "load": {"stages": [{"rate": 5, "duration": 1}]}  # 5 req/s for 1s
        },
    )

    adapter = LLMServingSimAdapter("LLMServingSim")

    # Verify can_run
    assert adapter.can_run(exp) is True

    # Run simulation (this will take ~30 seconds)
    result = adapter.run(exp)

    # Sanity checks on results
    assert result.adapter_name == "llmservingsim"
    assert result.experiment_folder == str(gt_dir)
    assert result.summary.e2e.mean > 0
    assert result.summary.ttft.mean > 0
    assert result.summary.itl.mean > 0
    assert result.summary.ttft.mean < result.summary.e2e.mean
    assert result.summary.itl.mean < result.summary.ttft.mean
    assert len(result.stages) == 1
    assert result.stages[0].e2e.mean > 0
```

- [ ] **Step 2: Run E2E test (skip if LLMServingSim not available)**

Run: `pytest tests/test_llmservingsim_integration.py -v -s`
Expected: PASS if LLMServingSim installed, SKIPPED otherwise

- [ ] **Step 3: Commit E2E test**

```bash
git add tests/test_llmservingsim_integration.py
git commit -m "test: add end-to-end integration test for LLMServingSim adapter

- Test full run() with real LLMServingSim on small workload
- Skip if LLMServingSim not installed
- Verify result structure and sanity check metrics with nested data model

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Documentation and Manual Validation

**Files:**
- Create: `experiment/adapters/README.md` (update or create)

- [ ] **Step 1: Document LLMServingSim adapter usage**

```markdown
# experiment/adapters/README.md (add section or create file)

## LLMServingSim Adapter

### Overview

The LLMServingSim adapter validates LLMServingSim's prediction accuracy against real vLLM experiments.

### Supported Configurations

- **Hardware**: H100 only
- **Models**:
  - Llama-3.1-8B (tp1, tp2)
  - Mixtral-8x7B-v0.1 (tp1, tp2, tp4)
- **Precision**: FP16 only
- **Features**: CPU offloading, multi-instance (dp>1)

### Usage

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters llmservingsim \
  --llmservingsim-dir LLMServingSim
```

### Requirements

- LLMServingSim installed at `LLMServingSim/` (or custom path via `--llmservingsim-dir`)
- Performance models at `llm_profile/perf_models/H100/{model}/tp{N}/`
- Ground-truth data with `per_request_lifecycle_metrics.json`

### How It Works

1. **Eligibility**: Checks H100 hardware, model support, perf model existence, FP16 precision
2. **Config Generation**: Creates cluster_config JSON with TP, memory, instance settings
3. **Workload Generation**: Reads ground-truth token counts, generates constant-rate arrivals
4. **Execution**: Runs LLMServingSim via subprocess with temp configs
5. **Parsing**: Converts CSV output (ns) to SimulatorResult (ms) with nested LatencyDistribution/ThroughputMetrics
```

- [ ] **Step 2: Commit documentation**

```bash
git add experiment/adapters/README.md
git commit -m "docs: add LLMServingSim adapter documentation

- Document supported configs, usage, requirements
- Explain adapter workflow
- Use actual CLI arguments (--data-dir, --output-dir)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Run manual validation on 1-2 real experiments**

Manual steps:
1. Verify LLMServingSim is installed: `ls LLMServingSim/main.py`
2. Run adapter on a simple experiment:
   ```bash
   python -m experiment.run \
     --data-dir vllm_data/ground_truth \
     --output-dir test_results \
     --adapters llmservingsim \
     --llmservingsim-dir LLMServingSim
   ```
3. Check output directory contains llmservingsim results
4. Verify metrics are reasonable (ITL > 0, TTFT > ITL, E2E > TTFT)
5. Compare order-of-magnitude against ground-truth

- [ ] **Step 4: Document validation results**

Create validation checklist in a temporary note:
```
Manual Validation Checklist:
- [ ] Llama-3.1-8B tp1 (no cpu_offload): PASS/FAIL
- [ ] Llama-3.1-8B tp1 (with cpu_offload): PASS/FAIL
- [ ] Mixtral-8x7B tp2 (with cpu_offload): PASS/FAIL
- [ ] Multi-instance dp=2: PASS/FAIL
- [ ] Output directory format correct: YES/NO
- [ ] Metrics order-of-magnitude reasonable: YES/NO
```

- [ ] **Step 5: Final commit for validation**

```bash
git commit --allow-empty -m "test: manual validation complete

Validated on real experiments:
- Llama-3.1-8B tp1 (with and without cpu_offload)
- Mixtral-8x7B tp2 with cpu_offload
- Multi-instance dp=2 experiment

All metrics within expected range.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Plan Review and Execution Handoff

After completing all tasks, this plan will be reviewed by plan-document-reviewer subagent and then ready for execution via superpowers:subagent-driven-development or superpowers:executing-plans.

**Total estimated completion time**: 6-8 hours for experienced developer following TDD

**Critical success factors**:
- All tests pass before moving to next task
- Follow TDD cycle strictly: test → fail → implement → pass → commit
- Use correct data model with nested LatencyDistribution/ThroughputMetrics/StageMetrics
- Import resolve_perf_dir from experiment.ground_truth (not experiment.resolve)
- Preserve existing exports in __init__.py
- Use actual CLI arguments (--data-dir, --output-dir, not --experiments-yaml, --output-csv)
- Verify integration manually before claiming completion
- Keep commits atomic and descriptive
