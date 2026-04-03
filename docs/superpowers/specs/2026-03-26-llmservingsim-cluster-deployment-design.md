# LLMServingSim Cluster Deployment Design

**Date**: 2026-03-26
**Purpose**: Deploy LLMServingSim evaluation on Kubernetes cluster to run slow experiments while fast simulators run locally
**Target Cluster**: diya namespace

## Overview

This design provides a Kubernetes deployment for running LLMServingSim adapter evaluations against ground-truth vLLM data. Since LLMServingSim is extremely slow (1+ hours per experiment), it will run on the cluster while the 7 fast simulators (BLIS variants, Vidur, estimators) run locally. Results from both runs can be merged afterward for complete analysis.

## Architecture

### Components

1. **PersistentVolumeClaim (PVC)**: 75GB dedicated storage for data, code, and results
2. **Upload Helper Pod**: Temporary pod for uploading data to PVC
3. **Evaluation Job**: CPU-only job that runs experiment.run with llmservingsim adapter
4. **Base Container**: Ubuntu 22.04 with Python and build tools

### Data Flow

```
Local Machine                   Cluster (diya namespace)
─────────────                   ────────────────────────

vllm_data/          ─────►      PVC (llmservingsim-eval-pvc)
LLMServingSim/      ─────►      ├── vllm_data/ground_truth/  (~20GB)
experiment/         ─────►      ├── LLMServingSim/           (~3GB)
                                ├── experiment/               (~50MB)
                                └── results/                  (output)

                                         │
                                         ▼
                                    Job Pod (16 CPU, 32GB RAM)
                                    └── python -m experiment.run
                                              --adapters llmservingsim
                                         │
                                         ▼
                                    results/
                                    ├── error_records.csv
                                    └── runtime.csv
                                         │
                                         ▼
                    ◄───────  kubectl cp (retrieve results)

Local results/      ─────►      Merge CSVs
(7 fast sims)                   └── All 8 simulators combined
```

## Storage Design

### PVC Configuration

- **Name**: `llmservingsim-eval-pvc`
- **Size**: 75Gi
- **Storage Class**: `ibm-spectrum-scale-fileset` (same as existing data-pvc)
- **Access Mode**: `ReadWriteMany`
- **Namespace**: `diya`

**Rationale:**
- Dedicated PVC provides clean isolation and easy cleanup
- 75GB is sufficient for ~20GB data + ~3GB code + ~45GB buffer
- ReadWriteMany allows multiple pods if needed (e.g., monitoring, retrieval)
- IBM Spectrum Scale is proven to work (data-pvc uses same class)

### Data Layout on PVC

```
/data/
├── vllm_data/
│   └── ground_truth/                    # ~20GB
│       ├── experiments.json
│       └── [experiment directories]/
│           ├── exp-config.yaml
│           ├── profile.yaml
│           ├── vllm.log
│           └── results/
│               └── per_request_lifecycle_metrics.json
├── LLMServingSim/                       # ~2-3GB
│   ├── main.py
│   ├── compile.sh
│   ├── astra-sim/
│   │   └── build/                       # Compiled once on PVC
│   │       └── astra_analytical/
│   │           └── build/AnalyticalAstra/bin/AnalyticalAstra
│   ├── llm_profile/
│   │   └── perf_models/
│   │       └── H100/
│   │           ├── meta-llama/Llama-3.1-8B/
│   │           └── mistralai/Mixtral-8x7B-v0.1/
│   ├── cluster_config/
│   └── dataset/
├── experiment/                          # ~50MB
│   ├── __init__.py
│   ├── run.py
│   ├── adapters/
│   │   ├── llmservingsim.py
│   │   ├── base.py
│   │   └── ...
│   ├── ground_truth.py
│   ├── metrics.py
│   └── report.py
└── results/                             # Output directory
    ├── error_records.csv
    └── runtime.csv
```

## Container Design

### Base Image

**Choice**: `ubuntu:22.04`

**Dependencies to install:**
- Build tools: `build-essential`, `cmake`, `git`, `wget`, `curl`
- Python: `python3`, `python3-pip`
- Python packages: `numpy`, `pyyaml`, `pandas`, `matplotlib`
- Astra-sim dependencies: Handled by LLMServingSim's compile.sh

**Why not a custom image?**
- Everything (code, compiled binaries) lives on PVC
- Base image only needs Python + build tools
- More flexible - can modify code without rebuilding image
- Simpler workflow - no Docker image registry needed

### Job Resource Specification

```yaml
resources:
  requests:
    cpu: "16"
    memory: "32Gi"
  limits:
    cpu: "16"
    memory: "32Gi"
```

**Rationale:**
- **16 CPUs**: astra-sim analytical backend is CPU-intensive (network simulation)
- **32GB RAM**: Handles simulation state, loaded data, intermediate results
- **No GPU**: Pure simulation, no actual inference

## Deployment Workflow

### Phase 1: One-Time Setup

#### Step 1: Create PVC

```bash
kubectl apply -f k8s/llmservingsim-eval-pvc.yaml
kubectl get pvc llmservingsim-eval-pvc -n diya -w  # Wait for Bound
```

#### Step 2: Launch Upload Helper Pod

```bash
kubectl apply -f k8s/upload-pod.yaml
kubectl wait --for=condition=Ready pod/upload-helper -n diya --timeout=120s
```

#### Step 3: Upload Data

```bash
# Create directory structure
kubectl exec -n diya upload-helper -- mkdir -p \
  /data/vllm_data \
  /data/LLMServingSim \
  /data/experiment \
  /data/results

# Upload ground truth data (~20GB - will take time)
kubectl cp vllm_data/ground_truth upload-helper:/data/vllm_data/ground_truth -n diya

# Upload LLMServingSim codebase (~3GB, includes astra-sim submodules)
# Note: Ensure LLMServingSim was cloned with --recurse-submodules
kubectl cp LLMServingSim/ upload-helper:/data/LLMServingSim/ -n diya

# Upload experiment framework
kubectl cp experiment/ upload-helper:/data/experiment/ -n diya

# Verify uploads
kubectl exec -n diya upload-helper -- du -sh /data/*
kubectl exec -n diya upload-helper -- ls -R /data/vllm_data/ground_truth | head -50
```

**Expected output:**
```
20G     /data/vllm_data
2.8G    /data/LLMServingSim
50M     /data/experiment
```

#### Step 4: Compile astra-sim

```bash
# Install Python dependencies first
kubectl exec -n diya upload-helper -- bash -c "
  apt-get update && apt-get install -y build-essential cmake git python3-pip && \
  pip3 install numpy pyyaml pandas matplotlib
"

# Run compilation (takes ~15-30 minutes)
kubectl exec -n diya upload-helper -- bash -c "
  cd /data/LLMServingSim && \
  ./compile.sh
"

# Verify binary was created
kubectl exec -n diya upload-helper -- test -f \
  /data/LLMServingSim/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra \
  && echo "✓ Build successful" || echo "✗ Build failed"
```

#### Step 5: Clean Up Helper Pod

```bash
kubectl delete pod upload-helper -n diya
```

### Phase 2: Execution

#### Step 1: Launch Evaluation Job

```bash
kubectl apply -f k8s/llmservingsim-eval-job.yaml
```

#### Step 2: Monitor Progress

```bash
# Stream logs (shows OK/FAIL/SKIP for each experiment)
kubectl logs -f job/llmservingsim-eval -n diya

# Check job status
kubectl get jobs -n diya
kubectl describe job llmservingsim-eval -n diya

# Check pod status
kubectl get pods -n diya -l job-name=llmservingsim-eval

# View recent log entries
kubectl logs job/llmservingsim-eval -n diya --tail=50
```

**Expected log output:**
```
Found 49 experiments
Parsed 49 experiments successfully
  OK: llmservingsim × meta-llama/Llama-3.1-8B-Instruct (shared_prefix) [1847.23s]
  SKIP: llmservingsim × CodeLlama-34b-Instruct (general)
  FAIL: llmservingsim × ... : Model not in MODEL_MAP
```

#### Step 3: Check Intermediate Results

```bash
# While job is running, check results written so far
POD=$(kubectl get pod -n diya -l job-name=llmservingsim-eval -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n diya $POD -- wc -l /data/results/error_records.csv
kubectl exec -n diya $POD -- tail -20 /data/results/error_records.csv
```

### Phase 3: Result Retrieval

#### Step 1: Wait for Completion

```bash
# Wait up to 7 days (168 hours)
kubectl wait --for=condition=complete --timeout=168h job/llmservingsim-eval -n diya

# Check final status
kubectl get job llmservingsim-eval -n diya
```

#### Step 2: Download Results

```bash
# Get pod name (even if completed, pod is preserved)
POD=$(kubectl get pod -n diya -l job-name=llmservingsim-eval -o jsonpath='{.items[0].metadata.name}')

# Download results
mkdir -p cluster_results
kubectl cp diya/$POD:/data/results/error_records.csv cluster_results/
kubectl cp diya/$POD:/data/results/runtime.csv cluster_results/

# Or use a retrieval pod if job pod is deleted
kubectl run retriever -n diya --image=busybox --restart=Never --rm -it \
  --overrides='{"spec":{"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"llmservingsim-eval-pvc"}}],"containers":[{"name":"retriever","image":"busybox","volumeMounts":[{"name":"data","mountPath":"/data"}],"command":["sh","-c","tar czf - /data/results"]}]}}' \
  > cluster_results.tar.gz
```

## Results Merging

### Local + Cluster Results

Since LLMServingSim runs on cluster and other simulators run locally, results need to be merged.

#### Step 1: Run Fast Simulators Locally

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters blis-blackbox blis-roofline blis-crossmodel \
              blis-trained-roofline vidur llm-optimizer-estimate \
              aiconfigurator-estimate
```

Produces:
- `results/error_records.csv` (7 simulators × N experiments)
- `results/runtime.csv` (7 simulators × N experiments)

#### Step 2: Merge with Cluster Results

**Option A: Shell commands**
```bash
# Backup originals
cp results/error_records.csv results/error_records_local.csv
cp results/runtime.csv results/runtime_local.csv

# Merge (skip header from cluster results)
cat results/error_records.csv > results/merged_error_records.csv
tail -n +2 cluster_results/error_records.csv >> results/merged_error_records.csv

cat results/runtime.csv > results/merged_runtime.csv
tail -n +2 cluster_results/runtime.csv >> results/merged_runtime.csv

# Replace originals
mv results/merged_error_records.csv results/error_records.csv
mv results/merged_runtime.csv results/runtime.csv
```

**Option B: Python script** (safer, handles duplicates)
```python
# merge_results.py
import pandas as pd

# Read local and cluster results
local_errors = pd.read_csv('results/error_records.csv')
cluster_errors = pd.read_csv('cluster_results/error_records.csv')

local_runtime = pd.read_csv('results/runtime.csv')
cluster_runtime = pd.read_csv('cluster_results/runtime.csv')

# Concatenate
merged_errors = pd.concat([local_errors, cluster_errors], ignore_index=True)
merged_runtime = pd.concat([local_runtime, cluster_runtime], ignore_index=True)

# Remove duplicates if any (e.g., if you accidentally ran same adapter twice)
merged_errors = merged_errors.drop_duplicates(
    subset=['simulator', 'experiment_folder', 'stage_index', 'metric_name']
)
merged_runtime = merged_runtime.drop_duplicates(
    subset=['simulator', 'experiment_folder']
)

# Save
merged_errors.to_csv('results/error_records.csv', index=False)
merged_runtime.to_csv('results/runtime.csv', index=False)

print(f"Merged {len(merged_errors)} error records from {len(merged_errors['simulator'].unique())} simulators")
print(f"Merged {len(merged_runtime)} runtime records")
```

Run with:
```bash
python merge_results.py
```

#### Step 3: Generate Figures

```bash
python -m experiment.figures \
  --results-dir results \
  --output-dir results/figures
```

Produces all publication figures with complete 8-simulator data.

## Execution Configuration

### Job Specification

**Job name**: `llmservingsim-eval`
**Namespace**: `diya`
**Restart policy**: `Never` (preserve logs, don't auto-retry)
**Backoff limit**: `0` (fail fast, debug manually)
**Completions**: `1`
**Parallelism**: `1`
**TTL after finished**: `604800` (7 days - keeps completed job for log retrieval)

### Environment Variables

```yaml
env:
  - name: PYTHONPATH
    value: "/data"
  - name: OMP_NUM_THREADS
    value: "16"
```

### Command

```bash
cd /data && python3 -m experiment.run \
  --adapters llmservingsim \
  --data-dir /data/vllm_data/ground_truth \
  --llmservingsim-dir /data/LLMServingSim \
  --output-dir /data/results \
  --no-docker \
  --verbose \
  --max-requests-per-experiment 100
```

**Key flags:**
- `--adapters llmservingsim`: Only run LLMServingSim (fast sims run locally)
- `--no-docker`: Use native execution (already inside container)
- `--verbose`: Enable INFO-level logging for progress tracking
- `--max-requests-per-experiment 100`: Limit to 100 requests per experiment (optional, for faster testing)

### Timeout Strategy

**Per-experiment timeout**: 1 hour (hardcoded in LLMServingSimAdapter)
**Job-level timeout**: None (or `activeDeadlineSeconds: 604800` = 7 days)

**Rationale:**
- Some experiments may timeout, adapter handles this gracefully
- Total experiments: ~49, but only subset will run (H100 + supported models)
- Estimated compatible experiments: 10-20
- Conservative total time: 20-40 hours
- 7-day limit provides safety net without premature termination

## Monitoring and Debugging

### Progress Tracking

**Real-time monitoring:**
```bash
# Watch for experiment completions
kubectl logs -f job/llmservingsim-eval -n diya | grep -E "(OK|FAIL|SKIP)"

# Count completed experiments
kubectl logs job/llmservingsim-eval -n diya | grep -c "OK:"

# Check current resource usage
kubectl top pod -n diya -l job-name=llmservingsim-eval
```

**Check results incrementally:**
```bash
POD=$(kubectl get pod -n diya -l job-name=llmservingsim-eval -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n diya $POD -- wc -l /data/results/error_records.csv
kubectl exec -n diya $POD -- tail /data/results/error_records.csv
```

### Expected Performance

**Per-experiment estimates:**
- Fast experiments: 15-30 minutes
- Slow experiments: 45-60 minutes (may timeout)
- Skipped experiments: instant (hardware/model incompatibility)

**Compatibility filtering (can_run()):**
- Hardware must be H100
- Model must be in MODEL_MAP (Llama-3.1-8B, Mixtral-8x7B)
- TP configuration must have perf models
- Precision must be FP16
- Attention predictions must exist

**Expected compatible subset**: ~10-20 of 49 experiments

**Total estimated runtime**: 10-30 hours

### Common Issues and Solutions

#### Job Won't Start

**Check pod status:**
```bash
kubectl get pods -n diya -l job-name=llmservingsim-eval
kubectl describe pod -n diya -l job-name=llmservingsim-eval
```

**Common causes:**
- PVC not bound: `kubectl get pvc -n diya`
- Resource quota exceeded: `kubectl describe resourcequota -n diya`
- Image pull failure: Check imagePullPolicy and image availability

#### Out of Memory (OOMKilled)

**Check memory usage:**
```bash
kubectl top pod -n diya -l job-name=llmservingsim-eval
```

**Solution:**
- Increase memory limit in job spec to 48Gi or 64Gi
- Reduce `--max-requests-per-experiment` to lower memory footprint

#### Compilation Failures

**Check compile.sh output:**
```bash
kubectl logs upload-helper -n diya
```

**Common causes:**
- Missing dependencies: Ensure apt packages installed
- Git submodule issues: Check LLMServingSim was uploaded with submodules
- CMake version: May need newer cmake (install from source)

#### Experiment Failures

**Check specific error:**
```bash
kubectl logs job/llmservingsim-eval -n diya | grep "FAIL:"
```

**Common causes:**
- Model not in MODEL_MAP: Only Llama-3.1-8B and Mixtral-8x7B supported
- Missing perf models: Check `llm_profile/perf_models/H100/{model}/tp{N}/`
- Missing attention predictions: Need `predictions/attn_*_predictions.csv`
- Hardware mismatch: Only H100 supported
- Precision mismatch: Only FP16 supported

#### Slow Progress

**Normal behavior:**
- LLMServingSim is inherently slow (astra-sim network simulation)
- 1 hour per experiment is expected
- Many experiments may be skipped via `can_run()` filter

**Check if stuck:**
```bash
# If no new logs for 2+ hours, may be stuck
kubectl logs job/llmservingsim-eval -n diya --tail=100 --timestamps
```

## Cleanup

### After Successful Completion

**Keep results, clean up job:**
```bash
# Download results first (see Phase 3)
kubectl cp diya/$POD:/data/results ./cluster_results

# Delete job (keeps PVC)
kubectl delete job llmservingsim-eval -n diya
```

### Complete Teardown

**Delete everything:**
```bash
# Delete job
kubectl delete job llmservingsim-eval -n diya

# Delete PVC (WARNING: deletes all data and results)
kubectl delete pvc llmservingsim-eval-pvc -n diya
```

### Incremental Cleanup

**Keep PVC for future runs:**
```bash
# Only delete job
kubectl delete job llmservingsim-eval -n diya

# Clear results directory for clean re-run
kubectl run cleaner -n diya --image=busybox --restart=Never --rm -it \
  --overrides='{"spec":{"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"llmservingsim-eval-pvc"}}],"containers":[{"name":"cleaner","image":"busybox","volumeMounts":[{"name":"data","mountPath":"/data"}],"command":["sh","-c","rm -rf /data/results/*"]}]}}'

# Re-run job with same PVC
kubectl apply -f k8s/llmservingsim-eval-job.yaml
```

## File Manifests Overview

The implementation will provide these YAML files:

1. **`k8s/llmservingsim-eval-pvc.yaml`** - PVC definition (75Gi)
2. **`k8s/upload-pod.yaml`** - Helper pod for data upload
3. **`k8s/llmservingsim-eval-job.yaml`** - Main evaluation job
4. **`merge_results.py`** - Python script to merge local + cluster CSVs
5. **`README-cluster-deployment.md`** - Quick-start guide with all commands

## Design Decisions Summary

### Key Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Storage** | Dedicated 75GB PVC | Clean isolation, easy cleanup |
| **Image** | Base Ubuntu + PVC for code | Flexibility, no custom registry needed |
| **Compilation** | One-time on PVC | Binary persists, don't rebuild per run |
| **Execution** | Single Job, sequential | Simple, sufficient for slow adapter |
| **Resources** | 16 CPU, 32GB RAM | Matches simulation needs, no GPU |
| **Timeout** | 7-day max, 1hr per experiment | Conservative, handles slow runs |
| **Results** | Write to PVC, merge locally | Simple retrieval, flexible merging |

### Alternative Approaches Considered

**Custom Docker image with code baked in:**
- Pro: Faster pod startup
- Con: Must rebuild image for code changes, need image registry
- Decision: Rejected - PVC approach is more flexible

**Run all 8 simulators on cluster:**
- Pro: Everything in one place
- Con: Fast sims don't need cluster, wastes resources
- Decision: Rejected - only run slow LLMServingSim on cluster

**Parallel job per experiment:**
- Pro: Faster completion if cluster has capacity
- Con: Complex orchestration, resource contention
- Decision: Deferred - single sequential job is simplest MVP

**Docker-in-Docker for LLMServingSim:**
- Pro: Matches local execution
- Con: Requires privileged pod, more complex
- Decision: Rejected - use `--no-docker` flag for native execution

## Success Criteria

1. ✓ PVC created and data uploaded successfully
2. ✓ astra-sim compiles without errors
3. ✓ Job starts and runs experiments sequentially
4. ✓ At least one experiment completes successfully (generates metrics)
5. ✓ Results can be retrieved from PVC
6. ✓ Merged results (local + cluster) produce valid figures

## Next Steps

After design approval:
1. Create implementation plan with detailed steps
2. Write YAML manifests for PVC, pods, and job
3. Write merge_results.py script
4. Create step-by-step deployment guide
5. Test upload process with small data subset
6. Run test job with --max-requests-per-experiment 10
7. Full production run with all experiments
