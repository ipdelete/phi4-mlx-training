# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Phi-4 Mini fine-tuning project using MLX on Apple Silicon Macs. The project uses the Dolly 15K dataset to fine-tune Microsoft's Phi-4 Mini language model for better instruction following.

## Key Commands

### Setup and Installation
```bash
# Install dependencies (using uv package manager)
uv pip install -e .

# Add Hugging Face CLI support
uv add "huggingface-hub[cli]"

# Download and prepare training data
python src/phi4-mlx-training/download_data.py
```

### Training Commands

**Using Python Scripts (Recommended):**
```bash
# Run training with configuration
python src/phi4-mlx-training/train_phi4_lora.py

# Test the fine-tuned model
python src/phi4-mlx-training/test_model.py

# Compare base vs fine-tuned model
python src/phi4-mlx-training/compare_models.py

# Fuse adapter into standalone model
python src/phi4-mlx-training/fuse_model.py
```

**Direct MLX Commands:**
```bash
# Basic LoRA fine-tuning
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

# Memory-optimized with 4-bit quantization
python -m mlx_lm lora \
  --model mlx-community/Phi-4-mini-instruct-4bit \
  --train \
  --data ./data \
  --batch-size 4 \
  --num-layers 16 \
  --iters 1000 \
  --adapter-path ./adapters
```

### Inference Commands
```bash
# Test fine-tuned model with adapter
python -m mlx_lm generate \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --prompt "<|user|>\nYour prompt here<|end|>\n<|assistant|>\n" \
  --max-tokens 300

# Fuse adapter into standalone model
python -m mlx_lm fuse \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --save-path ./fused-model
```

## Project Architecture

The project follows MLX's fine-tuning workflow:

1. **Data Preparation**: `download_data.py` fetches Dolly 15K dataset and formats it for Phi-4's chat template
2. **Training Data Format**: JSONL files with full conversations including special tokens (`<|user|>`, `<|assistant|>`, `<|end|>`)
3. **LoRA Fine-tuning**: Uses MLX's efficient LoRA implementation to train adapters instead of full model
4. **Configuration**: `train_config.py` centralizes all training parameters
5. **Output**: Produces adapter weights that can be applied to base model or fused into a standalone model

Key files:
- `src/phi4-mlx-training/train_config.py`: Centralized training configuration
- `src/phi4-mlx-training/train_phi4_lora.py`: Training script using config
- `src/phi4-mlx-training/test_model.py`: Test fine-tuned model
- `src/phi4-mlx-training/compare_models.py`: Compare base vs fine-tuned
- `src/phi4-mlx-training/fuse_model.py`: Merge adapter with base model

Key directories:
- `data/`: Contains train.jsonl and valid.jsonl formatted for MLX (auto-discovered by MLX)
- `adapters/`: Stores LoRA adapter weights after training (excluded from git)
- `fused-model/`: (Optional) Contains merged model after fusion (excluded from git)

## Important Considerations

### MLX CLI Syntax Updates
- Use `mlx_lm lora` instead of deprecated `mlx_lm.lora`
- Use `--num-layers` instead of deprecated `--lora-layers`
- Apply same pattern for other commands: `mlx_lm generate`, `mlx_lm fuse`

### Memory Management
- Memory usage scales with batch size and LoRA layers; adjust based on available RAM
- Default configuration in `train_config.py` targets 16GB+ RAM systems
- For 8GB systems: set batch_size=1 and lora_layers=8
- Consider using 4-bit quantized models for better memory efficiency

### Training Data Convention
- When specifying `--data ./data`, MLX automatically uses both files:
  - `train.jsonl` for training
  - `valid.jsonl` for validation (monitors overfitting)
- Validation loss is evaluated every `--steps-per-eval` iterations

### Performance Expectations
- Training on M1/M2/M3/M4 Macs typically takes 25-60 minutes for 1000 iterations
- The project uses Phi-4's specific chat template format for proper instruction following
- Adapter files are ~4MB each, totaling ~44MB for a full training run