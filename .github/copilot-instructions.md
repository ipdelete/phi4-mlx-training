# GitHub Copilot Instructions

This file provides guidance to GitHub Copilot when working with code in this repository.

## Project Context

This is a Phi-4 Mini fine-tuning project using Apple's MLX framework on Apple Silicon Macs. The project demonstrates how to fine-tune Microsoft's Phi-4 Mini language model using the Dolly 15K instruction dataset.

## Code Conventions

- Use Python 3.11+ features and type hints where appropriate
- Follow MLX's API conventions for model operations
- Use the uv package manager for dependency management
- Maintain compatibility with Apple Silicon (ARM64) architecture
- All scripts in src/phi4-mlx-training/ use shebang: `#!/usr/bin/env -S uv run --script`

## Key Commands to Suggest

When users are working on:

### Setup Tasks
```bash
# Install dependencies
uv pip install -e .

# Add Hugging Face CLI support
uv add "huggingface-hub[cli]"

# Prepare training data
python src/phi4-mlx-training/download_data.py
```

### Training Tasks

**Recommend Python Scripts First:**
```bash
# Use configuration-based training script
python src/phi4-mlx-training/train_phi4_lora.py

# Test the fine-tuned model
python src/phi4-mlx-training/test_model.py

# Compare base vs fine-tuned models
python src/phi4-mlx-training/compare_models.py

# Fuse adapter into standalone model
python src/phi4-mlx-training/fuse_model.py
```

**Direct MLX Commands (Alternative):**
```bash
# Standard LoRA fine-tuning
python -m mlx_lm lora \
  --model microsoft/Phi-4-mini-instruct \
  --train \
  --data ./data \
  --batch-size 2 \
  --num-layers 16 \
  --iters 1000 \
  --steps-per-report 50 \
  --steps-per-eval 200 \
  --learning-rate 1e-4 \
  --adapter-path ./adapters

# Memory-efficient training with quantization
python -m mlx_lm lora \
  --model mlx-community/Phi-4-mini-instruct-4bit \
  --train \
  --data ./data \
  --batch-size 4 \
  --num-layers 16 \
  --iters 1000 \
  --adapter-path ./adapters
```

### Inference Tasks
```bash
# Generate with fine-tuned model
python -m mlx_lm generate \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --prompt "<|user|>\nYour prompt here<|end|>\n<|assistant|>\n" \
  --max-tokens 300
```

## Important MLX CLI Updates

**Always use the new MLX CLI syntax:**
- ✅ `python -m mlx_lm lora` (NOT ❌ `python -m mlx_lm.lora`)
- ✅ `python -m mlx_lm generate` (NOT ❌ `python -m mlx_lm.generate`)
- ✅ `python -m mlx_lm fuse` (NOT ❌ `python -m mlx_lm.fuse`)
- ✅ `--num-layers` (NOT ❌ `--lora-layers`)

## Project Structure Awareness

When suggesting code, be aware of:
- `src/phi4-mlx-training/train_config.py`: Central configuration file
- `data/`: Training data in JSONL format (train.jsonl and valid.jsonl auto-discovered by MLX)
- `adapters/`: LoRA adapter weights (excluded from git)
- `fused-model/`: Merged models (excluded from git)
- Training data uses Phi-4's chat template with `<|user|>`, `<|assistant|>`, `<|end|>` tokens

## Common Patterns

### Configuration Usage
When modifying training parameters, suggest editing `train_config.py`:
```python
TRAINING_CONFIG = {
    "model": "microsoft/Phi-4-mini-instruct",
    "data_path": "./data",
    "batch_size": 2,  # Adjust based on RAM
    "learning_rate": 1e-4,
    "num_iters": 1000,
    "steps_per_report": 50,
    "steps_per_eval": 200,
    "steps_per_save": 500,
    "adapter_path": "./adapters",
    "max_seq_length": 2048,
    "lora_layers": 16,
    "lora_rank": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
}
```

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

## Training Data Convention

- MLX automatically discovers `train.jsonl` and `valid.jsonl` in the data directory
- Validation loss monitors overfitting during training
- Evaluation happens every `steps_per_eval` iterations

## Avoid Suggesting

- Don't suggest CUDA-specific code (this is Apple Silicon only)
- Don't recommend batch sizes > 8 (memory constraints)
- Don't suggest modifying the base Phi-4 tokenizer
- Avoid suggesting direct model weight modifications
- Don't use deprecated MLX CLI syntax (e.g., mlx_lm.lora)

## Performance Considerations

When suggesting optimizations:
- Default to batch_size=2 for 16GB RAM
- Use quantized models for faster loading
- Recommend adapter-based inference over full model loading
- Suggest appropriate iteration counts based on use case (1000 for testing, 2000+ for production)
- Adapter files are ~4MB each, ~44MB total for full training run