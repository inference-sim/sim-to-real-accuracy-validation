# Model Comparison

## MoE Architecture Overview

**All models use standard MoE architecture (MoE in every layer)**, not interleaved MoE:
- **Mixtral-8x7B & Mixtral-8x22B**: No `decoder_sparse_step` field → MoE applied to all layers
- **Llama-4-Scout-17B**: `interleave_moe_layer_step: 1` → MoE in every layer (all 48 layers)
- **GPT-OSS-20B & GPT-OSS-120B**: MoE applied to all layers (24 and 36 layers respectively)
- **Qwen3-30B-A3B**: `decoder_sparse_step: 1` → MoE in every layer (all 48 layers)

---

## Mixtral-8x7B

### Similarities with Others
- Same model_type as Mixtral-8x22B
- Same MoE config (8 experts, 2 active) as Mixtral-8x22B
- Standard architecture (silu, RMSNorm, GQA, no tied embeddings)
- BF16 precision like most others
- No quantization like Mixtral-8x22B & Qwen3
- **Standard MoE (every layer)** like all other models

### Unique Differences
- Smallest expert count (8)
- Moderate router loss (0.02) vs others (0.001-0.9)
- Smallest vocab (32K)
- Standard 32K context
- 128-dim heads
- Standard full attention

---

## Mixtral-8x22B

### Similarities with Others
- Same model_type as Mixtral-8x7B
- Same MoE config (8 experts, 2 active) as Mixtral-8x7B
- Standard architecture (silu, RMSNorm, GQA, no tied embeddings)
- BF16 precision like most others
- No quantization like Mixtral-8x7B & Qwen3
- **Standard MoE (every layer)** like all other models

### Unique Differences
- Largest model (56 layers, 6144 hidden)
- 65K context (between Mixtral-8x7B and others)
- Very low router loss (0.001)
- 128-dim heads
- Standard full attention

---

## Llama-4-Scout-17B

### Similarities with Others
- Standard architecture (silu, RMSNorm, GQA, no tied embeddings)
- Conservative routing (1 expert/tok) like Mixtral (2/tok)
- BF16 base dtype
- Low router loss (0.001) like Mixtral-8x22B & Qwen3
- **Standard MoE (every layer)** like all other models (interleave_moe_layer_step: 1)

### Unique Differences
- Only multimodal model (vision+text)
- Only FP8 quantized (8-bit weights & dynamic activations)
- Extreme 10M token context (RoPE scaling 16x)
- Largest vocab (202K)
- 128-dim heads
- Uses QK normalization (unique)

---

## GPT-OSS-20B

### Similarities with Others
- Same model_type as GPT-OSS-120B
- Same MoE routing (32 experts, 4 active) as GPT-OSS-120B
- Standard architecture (silu, RMSNorm, GQA, no tied embeddings)
- MXFP4 quantization like GPT-OSS-120B
- YaRN RoPE scaling like GPT-OSS-120B
- Hybrid attention pattern like GPT-OSS-120B
- **Standard MoE (every layer)** like all other models

### Unique Differences
- Smallest model (24 layers, 2880 hidden)
- 64-dim heads (vs 128 in most others)
- Hybrid sliding+full attention (128 token window)
- Highest router loss (0.9)
- 131K context (YaRN 32x from 4K)
- Attention bias enabled (vs false in most)
- 32 total experts

---

## GPT-OSS-120B

### Similarities with Others
- Same model_type as GPT-OSS-20B
- Same MoE routing (128 experts, 4 active) with Qwen3
- Standard architecture (silu, RMSNorm, GQA, no tied embeddings)
- MXFP4 quantization like GPT-OSS-20B
- YaRN RoPE scaling like GPT-OSS-20B
- Hybrid attention pattern like GPT-OSS-20B
- **Standard MoE (every layer)** like all other models

### Unique Differences
- Most experts (128) tied with Qwen3
- 64-dim heads (vs 128 in most others)
- Hybrid sliding+full attention (128 token window)
- Highest router loss (0.9)
- 131K context (YaRN 32x from 4K)
- Attention bias enabled (vs false in most)
- 36 layers (mid-size)

---

## Qwen3-30B-A3B

### Similarities with Others
- Same expert count (128) as GPT-OSS-120B
- Standard architecture (silu, RMSNorm, GQA, no tied embeddings)
- BF16 precision like Mixtral models
- No quantization like Mixtral models
- Low router loss (0.001) like Mixtral-8x22B & Llama-4-Scout
- Standard full attention like Mixtral & Llama-4-Scout
- **Standard MoE (every layer)** like all other models (decoder_sparse_step: 1)

### Unique Differences
- Most aggressive routing (8 experts/token)
- Smallest hidden size (2048)
- Fewest KV heads (4 vs 8 in others)
- 128-dim heads
- Unique features: norm_topk_prob, decoder_sparse_step
- Slightly different RMS eps (1e-06 vs 1e-05)
- 40K native context

---
