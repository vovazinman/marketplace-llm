# Marketplace LLM — Fine-tuned TinyLlama

Fine-tuned TinyLlama 1.1B on real estate and vehicle marketplace data.

## Benchmark Results

| Method | Tokens/sec | Latency (5 questions) |
|--------|-----------|----------------------|
| HuggingFace Transformers | 13.0 | 34 sec |
| vLLM | 163.0 | 2.4 sec |
| **Improvement** | **12.5x faster** | **14x faster** |

## Setup
- **GPU**: Tesla T4 16GB  
- **Model**: TinyLlama 1.1B + LoRA fine-tune
- **Quantization**: 4-bit (bitsandbytes)
- **Serving**: vLLM with PagedAttention

## Training
- Trainable params: 1.1M / 1.1B (0.1% via LoRA)
- Training time: 41 seconds
- Loss: 2.85 → 1.02
- Domain: Cars & Real Estate Q&A (Israel marketplace)

## Tech Stack
- PyTorch + HuggingFace Transformers
- PEFT (LoRA) — r=8, alpha=32
- vLLM 0.20.2
- 4-bit Quantization (bitsandbytes)

## Files
- `vllm_benchmark.py`         — vLLM serving benchmark  
- `adapter_model.safetensors` — LoRA fine-tuned weights
- `adapter_config.json`       — LoRA configuration
