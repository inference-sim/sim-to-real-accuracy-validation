# Simulator Limitations (Non-BLIS)

Known limitations of Vidur, LLM-Optimizer, and AIConfigurator when simulating
the 49 safe ground-truth vLLM experiments(https://ibm.box.com/s/0e2pthvxyu8gmkc47nqamk6ozdvjofme).
All ground truth was collected on vLLM v0.15.1 with specific serving configurations.
This doc covers what each simulator cannot model at the vLLM serving level.

---

## vLLM Parameter Variations

The 49 safe+done experiments sweep these vLLM serving configuration parameters.
These are the knobs set at vLLM launch time that affect serving behavior — simulators
that don't model them will produce identical predictions regardless of the setting:

| Parameter | Values in experiments | Experiment IDs |
|-----------|-----------------------|----------------|
| `mbt` | **1024**, 2048, **8192** | 22, 27 (1024); 23, 28 (8192) |
| `cpu_offload` | false, **true** | 2, 3, 6, 7, 9–12, 24, 29 (true) |
| `gpu_mem` | 0.9, **0.95** | 25, 30 (0.95) |
| `tp` | 1, 2, 4, **8** | varies; 49–51, 53 (tp=8) |
| `dp` | null, 1, **2**, **4** | 32, 33, 35 (dp=2); 34 (dp=4) |
| `precision` | FP16, **FP8** | 17, 20, 21, 33, 34, 44, 48 (FP8) |

### Parameter pass-through by adapter

Which of these parameters each adapter actually sends to the simulator:

| Parameter | Vidur | LLM-Optimizer | AIConfigurator |
|-----------|:-----:|:-------------:|:--------------:|
| `mbt` | **Yes** (`max_tokens_in_batch`) | No | No |
| `cpu_offload` | No (no CPU memory tier) | No (not a roofline input) | No (not a TaskConfig input) |
| `gpu_mem` | No (adapter bypasses memory planner with explicit `num_blocks`) | No (not a roofline input) | No (not a TaskConfig input) |
| `tp` | **Yes** | **Yes** (`num_gpus`) | **Yes** (`total_gpus`) |
| `dp` | **Yes** (`num_replicas`, see methodology for caveats) | No (not in roofline) | No |
| `precision` | No (hardcoded FP16) | **Yes** (`experiment.precision`) | **Yes** (`profiles=["float16_default"]`) |

**Bottom line:** `tp` and `precision` are now passed by LLM-Optimizer and AIConfigurator. `mbt` is passed only by Vidur. `dp` is passed only by Vidur. `cpu_offload` and `gpu_mem` are true limitations for all three — no simulator models them.

---

## Vidur

**Coverage: ~9 of 49 experiments** (3 pre-profiled models × H100/A100)

### Cannot run at all

| Reason | Affected experiments |
|--------|----------------------|
| No model profile | Qwen3-14B, Llama-3.1-8b, Llama-4-Scout, Mixtral-8x7B, Mixtral-8x22B |
| No L40S device SKU | IDs 54, 55 |
| No FP8 | IDs 17, 20, 21, 33, 34, 44, 48 |

### Runs but blind to variation

| Parameter | Effect | Affected experiments |
|-----------|--------|----------------------|
| `cpu_offload` ignored | No CPU memory tier; simulates as GPU-only even when real vLLM offloads KV cache to CPU | IDs 2, 3, 6, 7 |
| `gpu_mem` ignored | Adapter passes explicit `num_blocks`, bypassing Vidur's memory planner; `memory_margin_fraction` is unused | IDs 25, 30 |

### Structural limitations

| Limitation | Detail |
|------------|--------|
| No MoE | Zero MoE references in codebase |
| No chunked prefill | vLLM scheduler processes entire prefill in one shot; real vLLM chunks it. TTFT fidelity gap and potential runtime failures |

---

## LLM-Optimizer (BentoML)

**Coverage: ~46 of 49 experiments** (H100 + A100, excluding L40S and A100 FP8)

### Cannot run at all

| Reason | Affected experiments |
|--------|----------------------|
| No L40S GPU spec | IDs 54, 55 |
| A100 has `FP8_TFLOPS=None` | ID 44 (Scout FP8 on A100) |

### Runs but blind to variation

| Parameter | Effect | Affected experiments |
|-----------|--------|----------------------|
| `mbt` not an input | Roofline doesn't model scheduling; mbt=1024/2048/8192 give identical results | IDs 22, 23, 27, 28 |
| `cpu_offload` not an input | No memory modeling; cpu_offload has no effect | IDs 2, 3, 6, 7, 9–12, 24, 29 |
| `gpu_mem` not an input | No memory modeling; gpu_mem=0.95 same as 0.9 | IDs 25, 30 |
| `dp` not in roofline | Multi-instance experiments simulated as single-instance | IDs 32, 33, 34, 35 |

### Structural limitations

| Limitation | Detail |
|------------|--------|
| Mean metrics only | No P50/P90/P99; adapter sets `p90=None, p99=None`. All tail-latency comparisons impossible. |
| No MoE awareness | Hardcodes `d_ff = 4 * d_model`; MoE experiments (IDs 9–12, 27–32, 42, 49–51, 53, 56) treated as dense |

---

## AIConfigurator

**Coverage: ~20 of 49 experiments** (H100 dense models only)

### Cannot run at all

| Reason | Affected experiments |
|--------|----------------------|
| No A100/L40S vLLM perf data | IDs 36, 38, 40, 41, 42, 44, 54, 55 |
| MoE excluded | IDs 9–12, 17, 20, 21, 27–34, 42, 44, 48, 49–51, 53, 56 |

### Runs but blind to variation

| Parameter | Effect | Affected experiments |
|-----------|--------|----------------------|
| `mbt` not an input | Identical estimates regardless of mbt | IDs 22, 23 |
| `cpu_offload` not an input | Identical estimates with or without offload | IDs 2, 3, 6, 7, 24 |
| `gpu_mem` not an input | 0.95 same as 0.9 | ID 25 |
| `dp` not an input | `total_gpus=tp`, ignores dp entirely | ID 35 |

### Structural limitations

| Limitation | Detail |
|------------|--------|
| Mean metrics only | Returns mean TTFT and TPOT — no percentiles. All tail-latency comparisons impossible. |
