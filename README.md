# Phi-4 Mini Fine-Tuning with MLX

A complete tutorial for fine-tuning Microsoft's Phi-4 Mini language model using Apple's MLX framework on Mac computers with Apple Silicon.

## What This Tutorial Covers

This tutorial demonstrates how to:
- Fine-tune Phi-4 Mini using the Dolly 15K instruction dataset
- Use LoRA (Low-Rank Adaptation) for efficient training on consumer hardware
- Run the entire training process locally on Apple Silicon Macs
- Test and deploy your fine-tuned model

## Prerequisites

- **Hardware**: Mac with Apple Silicon (M1, M2, M3, or M4)
- **OS**: macOS 12.0 or later
- **RAM**: Minimum 16GB (32GB+ recommended)
- **Storage**: 50GB+ free disk space
- **Python**: 3.11 or later

## Documentation

- 🚀 [Full Tutorial (START HERE)](docs/tutorial.md) – Detailed step-by-step guide
- [MLX Documentation](https://ml-explore.github.io/mlx/) - Official MLX docs
- [Phi-4 Model Card](https://huggingface.co/microsoft/Phi-4-mini-instruct) - Model details
- [Fine tuned model by Ian Philpot](https://huggingface.co/ianphil/phi4-mini-dolly-15k-mlx)

## Quick Start

### 1. Clone and Set Up Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/phi4-mlx-training.git
cd phi4-mlx-training

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### 2. Prepare Training Data

```bash
# Download and format Dolly 15K dataset
python src/phi4-mlx-training/download_data.py
```

This creates:
- `data/train.jsonl`: ~12,000 training examples
- `data/valid.jsonl`: ~3,000 validation examples

### 3. Start Fine-Tuning

Using the provided training script (recommended):
```bash
# Run training with configuration
python src/phi4-mlx-training/train_phi4_lora.py
```

Or run MLX directly:
```bash
# Run LoRA fine-tuning
python -m mlx_lm lora \
  --model microsoft/Phi-4-mini-instruct \
  --train \
  --data ./data \
  --batch-size 2 \
  --num-layers 16 \
  --iters 1000 \
  --adapter-path ./adapters
```

**Note:** MLX CLI syntax has been updated. Use `mlx_lm lora` instead of `mlx_lm.lora`, and `--num-layers` instead of `--lora-layers`.

Training takes approximately:
- M1 (8GB): 45-60 minutes
- M2/M3 (16GB+): 30-40 minutes
- M4 (16GB+): 25-35 minutes

### 4. Test Your Model

Using the provided test script:
```bash
# Run test script
python src/phi4-mlx-training/test_model.py
```

Or test manually:
```bash
# Generate text with fine-tuned model
python -m mlx_lm generate \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --prompt "<|user|>\nExplain quantum computing<|end|>\n<|assistant|>\n" \
  --max-tokens 300
```

## Project Structure

```
phi4-mlx-training/
├── README.md                     # This file
├── CLAUDE.md                     # Guide for Claude Code AI assistant
├── pyproject.toml               # Project dependencies
├── docs/
│   └── tutorial.md              # Detailed step-by-step tutorial
├── src/
│   └── phi4-mlx-training/
│       ├── __init__.py
│       ├── download_data.py     # Dataset preparation script
│       ├── train_config.py      # Training configuration
│       ├── train_phi4_lora.py   # Training script
│       ├── test_model.py        # Model testing script
│       ├── compare_models.py    # Compare base vs fine-tuned
│       └── fuse_model.py        # Merge adapter with base model
├── data/                        # Training data (generated)
│   ├── train.jsonl
│   └── valid.jsonl
└── adapters/                    # LoRA weights (generated)
    └── adapter files...
```

## What is Phi-4 Mini?

Phi-4 Mini is Microsoft's latest small language model (3.8B parameters) that achieves strong performance on reasoning and instruction-following tasks. This tutorial fine-tunes it on the Dolly 15K dataset, which contains 15,000 human-generated instruction-following examples.

## Why MLX?

MLX is Apple's machine learning framework optimized for Apple Silicon. It provides:
- Efficient memory usage through unified memory architecture
- Fast training on M-series chips
- Simple Python API similar to NumPy/PyTorch

## Memory Requirements

| Configuration | Minimum RAM | Recommended RAM | Batch Size |
|--------------|-------------|-----------------|------------|
| Base model   | 8GB         | 16GB           | 1-2        |
| 4-bit quant  | 8GB         | 16GB           | 2-4        |
| Full training| 16GB        | 32GB           | 4-8        |

## Customization Options

### Adjust for Limited Memory
```bash
# Reduce memory usage
--batch-size 1 --num-layers 8
```

### Use Quantized Model
```bash
# 4-bit quantization for faster loading
--model mlx-community/Phi-4-mini-instruct-4bit
```

### Longer Training
```bash
# More iterations for better results
--iters 2000 --learning-rate 5e-5
```

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` to 1
- Use `--num-layers 8` instead of 16
- Try 4-bit quantized model

### Slow Training
- Ensure you're using Apple Silicon (not Intel)
- Close other applications
- Use quantized model version

### Poor Results
- Increase `--iters` to 2000+
- Adjust `--learning-rate` (try 5e-5 or 2e-4)
- Ensure data format is correct

## License

This project is for educational purposes. Please refer to:
- [Phi-4 License](https://huggingface.co/microsoft/Phi-4-mini-instruct) for model usage
- [Dolly Dataset License](https://huggingface.co/datasets/databricks/databricks-dolly-15k) for data usage

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Microsoft for the Phi-4 model
- Databricks for the Dolly 15K dataset
- Apple MLX team for the framework