# Extended Table Rows for Discussion #598

These rows should be added to the existing table at https://github.com/inference-sim/inference-sim/discussions/598

## DP Scaled Experiments from `vllm_data/other_gt`

| # | Model | Precision | HW | Workload | mbt | cpu_offload | gpu_mem | TP | DP | Notes |
|---|-------|-----------|-----|----------|-----|-------------|---------|----|----|-------|
| 33 | Llama-4-Scout-17B-16E | FP8 | H100 | general | 2048 | false | 0.9 | 2 | 2 | DP sweep: 2 GPUs × DP=2 (4 GPUs total); TTFT p99: 0.150s |
| 34 | Llama-4-Scout-17B-16E | FP8 | H100 | general | 2048 | false | 0.9 | 2 | 4 | DP sweep: 2 GPUs × DP=4 (8 GPUs total); TTFT p99: 0.191s |
| 35 | Llama-2-7b-hf | FP16 | H100 | general | 2048 | false | 0.9 | 1 | 2 | DP sweep: 1 GPU × DP=2 (2 GPUs total); TTFT p99: 0.036s |

## Summary

All three experiments demonstrate **safe** operation (TTFT p99 < 5 seconds) with DP scaling:

- **Experiment 33**: Llama-4-Scout-17B-16E with TP=2, DP=2 (4 GPUs total)
- **Experiment 34**: Llama-4-Scout-17B-16E with TP=2, DP=4 (8 GPUs total)
- **Experiment 35**: Llama-2-7b-hf with TP=1, DP=2 (2 GPUs total)

These experiments provide valuable data points for validating simulator accuracy when Data Parallelism is enabled, complementing the existing 56 experiments from the primary ground truth collection.

## Key Observations

1. **Experiment 35** (Llama-2-7b) was already documented in `experiments.json` with `"done": false` and notes indicating "data not collected", but the data actually exists in `other_gt/35-llama-2-7b-hf-tp1-general-1-2` with excellent metrics.

2. **Experiments 33 & 34** (Llama-4-Scout-17B-16E) test different DP scales (2 vs 4) with the same base configuration, showing minimal TTFT degradation as DP increases (150ms → 191ms).

3. All three experiments use the "general" workload type, which is a multi-stage load profile suitable for comprehensive testing.
