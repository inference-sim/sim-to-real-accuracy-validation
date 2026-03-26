# LLMServingSim Cluster Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create Kubernetes manifests and scripts for deploying LLMServingSim evaluation on cluster

**Architecture:** Three YAML manifests (PVC, upload pod, job), one Python merge script, and deployment guide. All code lives on PVC, base Ubuntu image with runtime dependencies.

**Tech Stack:** Kubernetes YAML, Python 3.10+, pandas, kubectl

**Design Document:** `docs/superpowers/specs/2026-03-26-llmservingsim-cluster-deployment-design.md`

---

## File Structure

```
sim-to-real-accuracy-validation/
├── k8s/                                  # New directory for K8s manifests
│   ├── llmservingsim-eval-pvc.yaml      # PVC definition (75Gi)
│   ├── upload-pod.yaml                   # Helper pod for data upload
│   └── llmservingsim-eval-job.yaml      # Main evaluation job
├── scripts/                              # New directory for helper scripts
│   └── merge_results.py                  # Merge local + cluster CSVs
└── docs/
    └── cluster-deployment/               # New directory for deployment docs
        └── README.md                     # Quick-start deployment guide
```

---

## Task 1: Create PVC Manifest

**Files:**
- Create: `k8s/llmservingsim-eval-pvc.yaml`

- [ ] **Step 1: Create k8s directory**

```bash
mkdir -p k8s
```

- [ ] **Step 2: Write PVC manifest**

Create `k8s/llmservingsim-eval-pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llmservingsim-eval-pvc
  namespace: diya
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 75Gi
  storageClassName: ibm-spectrum-scale-fileset
```

- [ ] **Step 3: Validate YAML syntax**

```bash
# Test YAML is valid (requires kubectl context)
kubectl apply -f k8s/llmservingsim-eval-pvc.yaml --dry-run=client

# Or validate without cluster access
python3 -c "import yaml; yaml.safe_load(open('k8s/llmservingsim-eval-pvc.yaml'))" && echo "✓ Valid YAML"
```

Expected output: `✓ Valid YAML` or dry-run success message

- [ ] **Step 4: Commit**

```bash
git add k8s/llmservingsim-eval-pvc.yaml
git commit -m "feat: add PVC manifest for LLMServingSim evaluation

Create 75GB PVC on IBM Spectrum Scale storage for:
- Ground truth data (~20GB)
- LLMServingSim codebase (~3GB)
- Experiment scripts (~50MB)
- Results output

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create Upload Helper Pod Manifest

**Files:**
- Create: `k8s/upload-pod.yaml`

- [ ] **Step 1: Write upload pod manifest**

Create `k8s/upload-pod.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: upload-helper
  namespace: diya
  labels:
    app: llmservingsim-eval
    component: upload-helper
spec:
  restartPolicy: Never
  containers:
  - name: uploader
    image: ubuntu:22.04
    command: ["sleep", "infinity"]
    resources:
      requests:
        cpu: "2"
        memory: "4Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: llmservingsim-eval-pvc
```

- [ ] **Step 2: Validate YAML syntax**

```bash
kubectl apply -f k8s/upload-pod.yaml --dry-run=client
# Or
python3 -c "import yaml; yaml.safe_load(open('k8s/upload-pod.yaml'))" && echo "✓ Valid YAML"
```

Expected output: `✓ Valid YAML` or dry-run success message

- [ ] **Step 3: Commit**

```bash
git add k8s/upload-pod.yaml
git commit -m "feat: add upload helper pod manifest

Temporary pod for uploading data to PVC via kubectl cp:
- Ubuntu 22.04 base image
- Mounts llmservingsim-eval-pvc at /data
- Sleeps indefinitely until data upload complete
- 2 CPU, 4GB RAM for kubectl cp operations

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Create Evaluation Job Manifest

**Files:**
- Create: `k8s/llmservingsim-eval-job.yaml`

- [ ] **Step 1: Write job manifest**

Create `k8s/llmservingsim-eval-job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: llmservingsim-eval
  namespace: diya
  labels:
    app: llmservingsim-eval
    component: evaluation
spec:
  # Job completion settings
  completions: 1
  parallelism: 1
  backoffLimit: 0
  ttlSecondsAfterFinished: 604800  # Keep for 7 days

  # 7-day timeout (604800 seconds)
  activeDeadlineSeconds: 604800

  template:
    metadata:
      labels:
        app: llmservingsim-eval
        component: evaluation
    spec:
      restartPolicy: Never

      containers:
      - name: llmservingsim
        image: ubuntu:22.04

        # Command to run evaluation
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -e

            # Install dependencies
            echo "Installing dependencies..."
            apt-get update -qq
            apt-get install -y -qq build-essential cmake git python3 python3-pip wget curl
            pip3 install -q numpy pyyaml pandas matplotlib

            # Verify data exists
            echo "Verifying data on PVC..."
            ls -lh /data/
            test -d /data/vllm_data/ground_truth || { echo "ERROR: Ground truth data not found"; exit 1; }
            test -d /data/LLMServingSim || { echo "ERROR: LLMServingSim not found"; exit 1; }
            test -d /data/experiment || { echo "ERROR: Experiment scripts not found"; exit 1; }
            test -f /data/LLMServingSim/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra || { echo "ERROR: astra-sim not compiled"; exit 1; }

            # Run evaluation
            echo "Starting LLMServingSim evaluation..."
            cd /data
            python3 -m experiment.run \
              --adapters llmservingsim \
              --data-dir /data/vllm_data/ground_truth \
              --llmservingsim-dir /data/LLMServingSim \
              --output-dir /data/results \
              --no-docker \
              --verbose \
              --max-requests-per-experiment 100

            echo "Evaluation complete!"
            echo "Results written to /data/results/"
            ls -lh /data/results/

        # Environment variables
        env:
        - name: PYTHONPATH
          value: "/data"
        - name: OMP_NUM_THREADS
          value: "16"
        - name: PYTHONUNBUFFERED
          value: "1"

        # Resource requests and limits
        resources:
          requests:
            cpu: "16"
            memory: "32Gi"
          limits:
            cpu: "16"
            memory: "32Gi"

        # Mount PVC
        volumeMounts:
        - name: data
          mountPath: /data

      # PVC volume
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: llmservingsim-eval-pvc
```

- [ ] **Step 2: Validate YAML syntax**

```bash
kubectl apply -f k8s/llmservingsim-eval-job.yaml --dry-run=client
# Or
python3 -c "import yaml; yaml.safe_load(open('k8s/llmservingsim-eval-job.yaml'))" && echo "✓ Valid YAML"
```

Expected output: `✓ Valid YAML` or dry-run success message

- [ ] **Step 3: Commit**

```bash
git add k8s/llmservingsim-eval-job.yaml
git commit -m "feat: add evaluation job manifest

CPU-only job for running LLMServingSim experiments:
- 16 CPU cores, 32GB RAM
- 7-day timeout for long-running evaluation
- Installs dependencies at runtime
- Validates PVC data before running
- Runs experiment.run with llmservingsim adapter
- Limits to 100 requests per experiment for faster testing

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Create Results Merge Script

**Files:**
- Create: `scripts/merge_results.py`

- [ ] **Step 1: Create scripts directory**

```bash
mkdir -p scripts
```

- [ ] **Step 2: Write merge script**

Create `scripts/merge_results.py`:

```python
#!/usr/bin/env python3
"""Merge local and cluster evaluation results.

Usage:
    python scripts/merge_results.py

Expects:
    - results/error_records.csv (local simulators)
    - results/runtime.csv (local simulators)
    - cluster_results/error_records.csv (LLMServingSim from cluster)
    - cluster_results/runtime.csv (LLMServingSim from cluster)

Produces:
    - results/error_records.csv (merged, 8 simulators)
    - results/runtime.csv (merged, 8 simulators)
    - results/error_records_local.csv (backup of original local results)
    - results/runtime_local.csv (backup of original local results)
"""

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Install with: pip install pandas")
    sys.exit(1)


def merge_results():
    """Merge local and cluster CSV results."""

    # Define paths
    local_errors = Path("results/error_records.csv")
    local_runtime = Path("results/runtime.csv")
    cluster_errors = Path("cluster_results/error_records.csv")
    cluster_runtime = Path("cluster_results/runtime.csv")

    # Check all files exist
    missing = []
    for f in [local_errors, local_runtime, cluster_errors, cluster_runtime]:
        if not f.exists():
            missing.append(str(f))

    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nExpected directory structure:")
        print("  results/error_records.csv (local)")
        print("  results/runtime.csv (local)")
        print("  cluster_results/error_records.csv (cluster)")
        print("  cluster_results/runtime.csv (cluster)")
        sys.exit(1)

    print("Loading CSV files...")

    # Read local results
    try:
        local_err_df = pd.read_csv(local_errors)
        local_rt_df = pd.read_csv(local_runtime)
    except Exception as e:
        print(f"ERROR reading local results: {e}")
        sys.exit(1)

    # Read cluster results
    try:
        cluster_err_df = pd.read_csv(cluster_errors)
        cluster_rt_df = pd.read_csv(cluster_runtime)
    except Exception as e:
        print(f"ERROR reading cluster results: {e}")
        sys.exit(1)

    print(f"Local results: {len(local_err_df)} error records, {len(local_rt_df)} runtime records")
    print(f"Cluster results: {len(cluster_err_df)} error records, {len(cluster_rt_df)} runtime records")

    # Check for simulator overlap (shouldn't happen)
    local_sims = set(local_err_df['simulator'].unique())
    cluster_sims = set(cluster_err_df['simulator'].unique())
    overlap = local_sims & cluster_sims

    if overlap:
        print(f"WARNING: Simulators present in both local and cluster results: {overlap}")
        print("This may indicate duplicate runs. Duplicates will be removed.")

    # Merge error records
    print("\nMerging error records...")
    merged_err = pd.concat([local_err_df, cluster_err_df], ignore_index=True)

    # Remove duplicates (keep first occurrence)
    before_dedup = len(merged_err)
    merged_err = merged_err.drop_duplicates(
        subset=['simulator', 'experiment_folder', 'stage_index', 'metric_name'],
        keep='first'
    )
    after_dedup = len(merged_err)

    if before_dedup != after_dedup:
        print(f"Removed {before_dedup - after_dedup} duplicate error records")

    # Merge runtime records
    print("Merging runtime records...")
    merged_rt = pd.concat([local_rt_df, cluster_rt_df], ignore_index=True)

    # Remove duplicates
    before_dedup = len(merged_rt)
    merged_rt = merged_rt.drop_duplicates(
        subset=['simulator', 'experiment_folder'],
        keep='first'
    )
    after_dedup = len(merged_rt)

    if before_dedup != after_dedup:
        print(f"Removed {before_dedup - after_dedup} duplicate runtime records")

    # Backup originals
    print("\nBacking up original local results...")
    local_errors.rename(local_errors.parent / "error_records_local.csv")
    local_runtime.rename(local_runtime.parent / "runtime_local.csv")
    print("  - results/error_records_local.csv")
    print("  - results/runtime_local.csv")

    # Save merged results
    print("\nSaving merged results...")
    merged_err.to_csv(local_errors, index=False)
    merged_rt.to_csv(local_runtime, index=False)

    # Summary
    print("\n✓ Merge complete!")
    print(f"  - {len(merged_err)} total error records")
    print(f"  - {len(merged_rt)} total runtime records")
    print(f"  - {len(merged_err['simulator'].unique())} unique simulators:")
    for sim in sorted(merged_err['simulator'].unique()):
        count = len(merged_rt[merged_rt['simulator'] == sim])
        print(f"    • {sim}: {count} experiments")

    print("\nNext steps:")
    print("  python -m experiment.figures --results-dir results --output-dir results/figures")


if __name__ == "__main__":
    merge_results()
```

- [ ] **Step 3: Make script executable**

```bash
chmod +x scripts/merge_results.py
```

- [ ] **Step 4: Test script with mock data (validation only)**

Create minimal test CSVs to validate script logic:

```bash
# Create mock local results
mkdir -p results
echo "simulator,experiment_folder,model,workload,stage_index,metric_name,predicted,actual,mape,mpe,absolute_error" > results/error_records.csv
echo "blis-roofline,exp1,llama,general,0,e2e_mean,100,110,9.09,10.0,10" >> results/error_records.csv

echo "simulator,experiment_folder,model,workload,wall_clock_seconds" > results/runtime.csv
echo "blis-roofline,exp1,llama,general,45.2" >> results/runtime.csv

# Create mock cluster results
mkdir -p cluster_results
echo "simulator,experiment_folder,model,workload,stage_index,metric_name,predicted,actual,mape,mpe,absolute_error" > cluster_results/error_records.csv
echo "llmservingsim,exp1,llama,general,0,e2e_mean,105,110,4.54,5.0,5" >> cluster_results/error_records.csv

echo "simulator,experiment_folder,model,workload,wall_clock_seconds" > cluster_results/runtime.csv
echo "llmservingsim,exp1,llama,general,1847.3" >> cluster_results/runtime.csv

# Run merge script
python3 scripts/merge_results.py

# Verify output
echo "Checking merged results..."
grep -c "blis-roofline" results/error_records.csv
grep -c "llmservingsim" results/error_records.csv

# Clean up test data
rm -rf results/ cluster_results/
```

Expected output:
```
Loading CSV files...
Local results: 1 error records, 1 runtime records
Cluster results: 1 error records, 1 runtime records

Merging error records...
Merging runtime records...

Backing up original local results...
  - results/error_records_local.csv
  - results/runtime_local.csv

Saving merged results...

✓ Merge complete!
  - 2 total error records
  - 2 total runtime records
  - 2 unique simulators:
    • blis-roofline: 1 experiments
    • llmservingsim: 1 experiments
```

- [ ] **Step 5: Commit**

```bash
git add scripts/merge_results.py
git commit -m "feat: add results merge script

Python script to merge local and cluster CSV results:
- Combines error_records.csv and runtime.csv
- Removes duplicates if same simulator run twice
- Backs up original local results
- Validates all expected files exist
- Provides summary of merged data

Usage: python scripts/merge_results.py

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Create Deployment Guide

**Files:**
- Create: `docs/cluster-deployment/README.md`

- [ ] **Step 1: Create directory**

```bash
mkdir -p docs/cluster-deployment
```

- [ ] **Step 2: Write deployment guide**

Create `docs/cluster-deployment/README.md`:

```markdown
# LLMServingSim Cluster Deployment Guide

Quick-start guide for deploying LLMServingSim evaluation on Kubernetes cluster.

**Full Design**: See `docs/superpowers/specs/2026-03-26-llmservingsim-cluster-deployment-design.md`

## Prerequisites

- kubectl configured for diya namespace
- Local copy of this repository
- Ground truth data in `vllm_data/ground_truth/` (~20GB)
- LLMServingSim cloned with submodules in `LLMServingSim/` (~3GB)

## Quick Start

### 1. Create PVC

```bash
kubectl apply -f k8s/llmservingsim-eval-pvc.yaml
kubectl get pvc llmservingsim-eval-pvc -n diya -w  # Wait for Bound
```

### 2. Launch Upload Helper

```bash
kubectl apply -f k8s/upload-pod.yaml
kubectl wait --for=condition=Ready pod/upload-helper -n diya --timeout=120s
```

### 3. Upload Data

```bash
# Create directories
kubectl exec -n diya upload-helper -- mkdir -p /data/vllm_data /data/LLMServingSim /data/experiment /data/results

# Upload ground truth (~20GB - takes time)
kubectl cp vllm_data/ground_truth upload-helper:/data/vllm_data/ground_truth -n diya

# Upload LLMServingSim
kubectl cp LLMServingSim/ upload-helper:/data/LLMServingSim/ -n diya

# Upload experiment scripts
kubectl cp experiment/ upload-helper:/data/experiment/ -n diya

# Verify
kubectl exec -n diya upload-helper -- du -sh /data/*
```

Expected output:
```
20G     /data/vllm_data
2.8G    /data/LLMServingSim
50M     /data/experiment
```

### 4. Compile astra-sim

```bash
# Install dependencies
kubectl exec -n diya upload-helper -- bash -c "
  apt-get update && apt-get install -y build-essential cmake git python3-pip && \
  pip3 install numpy pyyaml pandas matplotlib
"

# Compile (takes 15-30 minutes)
kubectl exec -n diya upload-helper -- bash -c "cd /data/LLMServingSim && ./compile.sh"

# Verify
kubectl exec -n diya upload-helper -- test -f \
  /data/LLMServingSim/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra \
  && echo "✓ Build successful" || echo "✗ Build failed"
```

### 5. Clean Up Upload Pod

```bash
kubectl delete pod upload-helper -n diya
```

### 6. Launch Evaluation Job

```bash
kubectl apply -f k8s/llmservingsim-eval-job.yaml
```

### 7. Monitor Progress

```bash
# Stream logs
kubectl logs -f job/llmservingsim-eval -n diya

# Check status
kubectl get jobs -n diya
kubectl get pods -n diya -l job-name=llmservingsim-eval

# Check intermediate results
POD=$(kubectl get pod -n diya -l job-name=llmservingsim-eval -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n diya $POD -- wc -l /data/results/error_records.csv
```

### 8. Wait for Completion

```bash
kubectl wait --for=condition=complete --timeout=168h job/llmservingsim-eval -n diya
```

### 9. Download Results

```bash
POD=$(kubectl get pod -n diya -l job-name=llmservingsim-eval -o jsonpath='{.items[0].metadata.name}')
mkdir -p cluster_results
kubectl cp diya/$POD:/data/results/error_records.csv cluster_results/
kubectl cp diya/$POD:/data/results/runtime.csv cluster_results/
```

## Merging Results

After running fast simulators locally and LLMServingSim on cluster:

### 1. Run Local Simulators

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters blis-blackbox blis-roofline blis-crossmodel \
              blis-trained-roofline vidur llm-optimizer-estimate \
              aiconfigurator-estimate
```

### 2. Merge with Cluster Results

```bash
python scripts/merge_results.py
```

### 3. Generate Figures

```bash
python -m experiment.figures \
  --results-dir results \
  --output-dir results/figures
```

## Troubleshooting

### Job Won't Start

```bash
kubectl describe pod -n diya -l job-name=llmservingsim-eval
kubectl get pvc -n diya
```

### Check Logs

```bash
kubectl logs job/llmservingsim-eval -n diya --tail=100
```

### Out of Memory

Increase memory in `k8s/llmservingsim-eval-job.yaml`:

```yaml
resources:
  requests:
    memory: "48Gi"
  limits:
    memory: "48Gi"
```

### Compilation Failed

```bash
kubectl logs upload-helper -n diya
```

## Cleanup

### Keep PVC, Delete Job

```bash
kubectl delete job llmservingsim-eval -n diya
```

### Delete Everything

```bash
kubectl delete job llmservingsim-eval -n diya
kubectl delete pvc llmservingsim-eval-pvc -n diya
```

## Expected Runtime

- Compatible experiments: ~10-20 of 49 (H100 + supported models only)
- Per-experiment time: 15-60 minutes
- Total estimated: 10-30 hours

## Files

- `k8s/llmservingsim-eval-pvc.yaml` - 75GB PVC definition
- `k8s/upload-pod.yaml` - Helper pod for data upload
- `k8s/llmservingsim-eval-job.yaml` - Main evaluation job
- `scripts/merge_results.py` - Merge local + cluster CSVs
- `docs/superpowers/specs/2026-03-26-llmservingsim-cluster-deployment-design.md` - Full design
```

- [ ] **Step 3: Commit**

```bash
git add docs/cluster-deployment/README.md
git commit -m "docs: add cluster deployment quick-start guide

Step-by-step instructions for deploying LLMServingSim evaluation:
- PVC creation and data upload
- astra-sim compilation
- Job launch and monitoring
- Results retrieval and merging
- Troubleshooting common issues

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Validation and Documentation

**Files:**
- Modify: `README.md` (add link to cluster deployment guide)

- [ ] **Step 1: Add cluster deployment section to main README**

Add to `README.md` after the "Usage" section:

```markdown
## Cluster Deployment

For running LLMServingSim (extremely slow adapter) on a Kubernetes cluster:

See **[Cluster Deployment Guide](docs/cluster-deployment/README.md)** for complete instructions.

Quick summary:
1. Create 75GB PVC on cluster
2. Upload data and code to PVC
3. Compile astra-sim once on PVC
4. Launch evaluation job (10-30 hours runtime)
5. Download results and merge with local simulator outputs

This allows fast simulators to run locally while slow LLMServingSim runs on cluster resources.
```

- [ ] **Step 2: Verify all manifests are valid**

```bash
# Validate all YAML files
for f in k8s/*.yaml; do
  echo "Validating $f..."
  python3 -c "import yaml; yaml.safe_load(open('$f'))" && echo "  ✓ Valid" || echo "  ✗ Invalid"
done
```

Expected output:
```
Validating k8s/llmservingsim-eval-pvc.yaml...
  ✓ Valid
Validating k8s/upload-pod.yaml...
  ✓ Valid
Validating k8s/llmservingsim-eval-job.yaml...
  ✓ Valid
```

- [ ] **Step 3: Verify script is executable**

```bash
test -x scripts/merge_results.py && echo "✓ merge_results.py is executable" || echo "✗ Not executable"
python3 scripts/merge_results.py --help 2>&1 | head -5
```

- [ ] **Step 4: Verify documentation links**

```bash
# Check that design doc exists
test -f docs/superpowers/specs/2026-03-26-llmservingsim-cluster-deployment-design.md && \
  echo "✓ Design spec exists" || echo "✗ Design spec missing"

# Check deployment guide exists
test -f docs/cluster-deployment/README.md && \
  echo "✓ Deployment guide exists" || echo "✗ Deployment guide missing"
```

- [ ] **Step 5: Commit README update**

```bash
git add README.md
git commit -m "docs: add cluster deployment section to README

Link to cluster deployment guide for running LLMServingSim on K8s.
Explains rationale (slow adapter on cluster, fast adapters local).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

### Spec Coverage

From `docs/superpowers/specs/2026-03-26-llmservingsim-cluster-deployment-design.md`:

- [x] **PVC Configuration** - Task 1: llmservingsim-eval-pvc.yaml (75Gi, IBM Spectrum Scale)
- [x] **Upload Helper Pod** - Task 2: upload-pod.yaml (Ubuntu 22.04, mounts PVC)
- [x] **Evaluation Job** - Task 3: llmservingsim-eval-job.yaml (16 CPU, 32GB RAM, runs experiment.run)
- [x] **Results Merging** - Task 4: merge_results.py (merges local + cluster CSVs)
- [x] **Deployment Guide** - Task 5: README.md with all kubectl commands
- [x] **Integration with main docs** - Task 6: Update main README

### Placeholder Scan

- [x] No "TBD" or "TODO" markers
- [x] All code blocks are complete (no "..." or "add implementation here")
- [x] All file paths are exact
- [x] All commands have expected output
- [x] No "similar to Task N" references without actual code

### Type Consistency

- [x] PVC name consistent: `llmservingsim-eval-pvc` in all manifests
- [x] Namespace consistent: `diya` in all manifests
- [x] Label keys consistent: `app: llmservingsim-eval`, `component: upload-helper|evaluation`
- [x] Mount path consistent: `/data` in all manifests
- [x] File paths consistent across all tasks

### Implementation Notes

**Conventions:**
- All Kubernetes manifests in `k8s/` directory
- Helper scripts in `scripts/` directory
- Documentation in `docs/cluster-deployment/`
- Namespace: `diya`
- Storage class: `ibm-spectrum-scale-fileset`

**Critical Paths:**
- PVC mount point: `/data`
- Experiment scripts: `/data/experiment/`
- LLMServingSim: `/data/LLMServingSim/`
- Ground truth: `/data/vllm_data/ground_truth/`
- Results output: `/data/results/`

**Testing Strategy:**
- YAML validation with python yaml.safe_load()
- Script validation with mock CSV data
- Documentation validation by checking file existence

**Commit Message Format:**
All commits follow conventional commits style with Co-Authored-By trailer.

---

## Success Criteria

After completing all tasks:

1. ✓ `k8s/` directory contains 3 valid YAML manifests
2. ✓ `scripts/merge_results.py` is executable and validates CSV structure
3. ✓ `docs/cluster-deployment/README.md` provides step-by-step guide
4. ✓ Main `README.md` links to cluster deployment guide
5. ✓ All YAML validates successfully with kubectl dry-run or python yaml parser
6. ✓ All commits have descriptive messages with Co-Authored-By trailer

## Ready for Deployment

After implementation, user can:

1. Apply PVC manifest: `kubectl apply -f k8s/llmservingsim-eval-pvc.yaml`
2. Launch upload pod: `kubectl apply -f k8s/upload-pod.yaml`
3. Upload data via kubectl cp
4. Compile astra-sim on PVC
5. Launch job: `kubectl apply -f k8s/llmservingsim-eval-job.yaml`
6. Download results and merge: `python scripts/merge_results.py`

No cluster access required for implementation - all tasks can be completed locally with validation only.
