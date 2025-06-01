# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Phi-4 Mini fine-tuning project using MLX on Apple Silicon Macs. The project uses the Dolly 15K dataset to fine-tune Microsoft's Phi-4 Mini language model for better instruction following.

## Key Commands

### Setup and Installation
```bash
# Install dependencies (using uv package manager)
uv pip install -e .

# Download and prepare training data
python src/phi4-mlx-training/download_data.py
```

### Training Commands
```bash
# Basic LoRA fine-tuning
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

# Memory-optimized with 4-bit quantization
python -m mlx_lm.lora \
  --model mlx-community/Phi-4-mini-instruct-4bit \
  --train \
  --data ./data \
  --batch-size 4 \
  --lora-layers 16 \
  --iters 1000 \
  --adapter-path ./adapters
```

### Inference Commands
```bash
# Test fine-tuned model with adapter
python -m mlx_lm.generate \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --prompt "<|user|>\nYour prompt here<|end|>\n<|assistant|>\n" \
  --max-tokens 300

# Fuse adapter into standalone model
python -m mlx_lm.fuse \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --save-path ./fused-model
```

## Project Architecture

The project follows MLX's fine-tuning workflow:

1. **Data Preparation**: `download_data.py` fetches Dolly 15K dataset and formats it for Phi-4's chat template
2. **Training Data Format**: JSONL files with full conversations including special tokens (`<|user|>`, `<|assistant|>`, `<|end|>`)
3. **LoRA Fine-tuning**: Uses MLX's efficient LoRA implementation to train adapters instead of full model
4. **Output**: Produces adapter weights that can be applied to base model or fused into a standalone model

Key directories:
- `data/`: Contains train.jsonl and valid.jsonl formatted for MLX
- `adapters/`: Stores LoRA adapter weights after training
- `fused-model/`: (Optional) Contains merged model after fusion

## Important Considerations

- Memory usage scales with batch size and LoRA layers; adjust based on available RAM
- The project uses Phi-4's specific chat template format for proper instruction following
- Training on M1/M2/M3/M4 Macs typically takes 25-60 minutes for 1000 iterations
- Default configuration targets 16GB+ RAM systems; reduce batch_size for 8GB systems