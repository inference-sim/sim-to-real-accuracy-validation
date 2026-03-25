# LLMServingSim Adapter Validation Checklist

## Prerequisites
- [ ] LLMServingSim installed at `LLMServingSim/` with working dependencies
- [ ] Ground-truth vLLM data at `vllm_data/ground_truth/`
- [ ] H100 performance models present in LLMServingSim

## Manual Validation Tests

Run these commands and verify the output:

### Test 1: Llama-3.1-8B tp1 (no cpu_offload)
```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir test_results \
  --adapters llmservingsim \
  --llmservingsim-dir LLMServingSim
```
- [ ] Command completes successfully
- [ ] Output directory `test_results/` contains results
- [ ] Metrics are reasonable (ITL > 0, TTFT > ITL, E2E > TTFT)

### Test 2: Check eligible experiments
```bash
python -c "
from experiment.adapters.llmservingsim import LLMServingSimAdapter
from experiment.ground_truth import discover_experiments, parse_experiment

adapter = LLMServingSimAdapter('LLMServingSim')
discovered = discover_experiments('vllm_data/ground_truth')
eligible = []
for manifest_entry, dir_path in discovered:
    try:
        exp = parse_experiment(dir_path, manifest_entry=manifest_entry)
        if adapter.can_run(exp):
            eligible.append((exp.model, exp.hardware, exp.tp, exp.precision))
    except Exception:
        pass
print(f'Eligible experiments: {len(eligible)} out of {len(discovered)}')
for model, hw, tp, prec in eligible[:5]:
    print(f'  - {model} {hw} tp{tp} {prec}')
"
```
- [ ] Shows ~20 eligible experiments out of 49 total
- [ ] Lists Llama-3.1-8B and Mixtral-8x7B experiments

### Test 3: Multi-instance (dp>1) experiment
Look for an experiment with `dp > 1` in the eligible list and verify it runs with round-robin routing.

- [ ] Multi-instance experiment runs successfully
- [ ] Output contains metrics from all instances

## Expected Results

- **Coverage**: ~20/49 experiments eligible (H100 + Llama/Mixtral + FP16)
- **Metrics order**: ITL < TTFT < E2E (in milliseconds)
- **Throughput**: Non-zero tokens/sec and requests/sec
- **Multi-stage**: Stage boundaries respected in split

## Notes

Record any issues or observations here:
