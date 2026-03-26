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
  && echo "Build successful" || echo "Build failed"
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
