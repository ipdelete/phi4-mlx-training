# GitHub Copilot Instructions

This file provides guidance to GitHub Copilot when working with code in this repository.

## Project Context

This is a Phi-4 Mini fine-tuning project using Apple's MLX framework on Apple Silicon Macs. The project demonstrates how to fine-tune Microsoft's Phi-4 Mini language model using the Dolly 15K instruction dataset.

## Code Conventions

- Use Python 3.11+ features and type hints where appropriate
- Follow MLX's API conventions for model operations
- Use the uv package manager for dependency management
- Maintain compatibility with Apple Silicon (ARM64) architecture

## Key Commands to Suggest

When users are working on:

### Setup Tasks
```bash
# Install dependencies
uv pip install -e .

# Prepare training data
python src/phi4-mlx-training/download_data.py
```

### Training Tasks
```bash
# Standard LoRA fine-tuning
python -m mlx_lm.lora \
  --model microsoft/Phi-4-mini-instruct \
  --train \
  --data ./data \
  --batch-size 2 \
  --lora-layers 16 \
  --iters 1000 \
  --steps-per-report 50 \
  --steps-per-eval 200 \
  --learning-rate 1e-4 \
  --adapter-path ./adapters

# Memory-efficient training with quantization
python -m mlx_lm.lora \
  --model mlx-community/Phi-4-mini-instruct-4bit \
  --train \
  --data ./data \
  --batch-size 4 \
  --lora-layers 16 \
  --iters 1000 \
  --adapter-path ./adapters
```

### Inference Tasks
```bash
# Generate with fine-tuned model
python -m mlx_lm.generate \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --prompt "<|user|>\nYour prompt here<|end|>\n<|assistant|>\n" \
  --max-tokens 300
```

## Project Structure Awareness

When suggesting code, be aware of:
- `data/`: Training data in JSONL format
- `adapters/`: LoRA adapter weights
- `src/phi4-mlx-training/`: Main package code
- Training data uses Phi-4's chat template with `<|user|>`, `<|assistant|>`, `<|end|>` tokens

## Common Patterns

### Data Formatting
When working with training data, use this format:
```python
prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"
completion = f"{response}<|end|>"
```

### Memory Optimization
For systems with limited RAM:
- Suggest batch_size=1 for 8GB systems
- Recommend 4-bit quantized models
- Propose reducing lora_layers to 8

### MLX-Specific Patterns
- Use MLX's unified memory architecture advantages
- Leverage MLX's NumPy-like API
- Follow MLX's lazy evaluation patterns

## Avoid Suggesting

- Don't suggest CUDA-specific code (this is Apple Silicon only)
- Don't recommend batch sizes > 8 (memory constraints)
- Don't suggest modifying the base Phi-4 tokenizer
- Avoid suggesting direct model weight modifications

## Performance Considerations

When suggesting optimizations:
- Default to batch_size=2 for 16GB RAM
- Use quantized models for faster loading
- Recommend adapter-based inference over full model loading
- Suggest appropriate iteration counts based on use case (1000 for testing, 2000+ for production)