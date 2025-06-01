# Complete Tutorial: Fine-tuning Phi-4 Mini with Dolly 15K using MLX on Mac

## Prerequisites
- Mac with Apple Silicon (M1, M2, M3, or M4)
- macOS 12.0 or later
- At least 16GB RAM (32GB+ recommended)
- 50GB+ free disk space

## Step 1: Clone the Repository or Create Project

If you're using the existing repository:

```bash
# Clone the repository
git clone https://github.com/yourusername/phi4-mlx-training.git
cd phi4-mlx-training
```

Or create a new project from scratch:

```bash
# Create project directory
mkdir phi4-mlx-training
cd phi4-mlx-training
```

## Step 2: Install uv Package Manager

Install uv, which is a fast Python package manager:

```bash
# Install uv using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart your terminal or source your shell profile
source ~/.zshrc  # or ~/.bash_profile if using bash

# Verify installation
uv --version
```

## Step 3: Set Up Virtual Environment

```bash
# Create and activate virtual environment with Python 3.11
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
```

## Step 4: Install Required Dependencies

The project includes a `pyproject.toml` file with all dependencies:

```toml
[project]
name = "phi4-mlx-training"
version = "0.1.0"
description = "Fine-tuning Phi-4 Mini with MLX"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "datasets>=3.6.0",
    "huggingface-hub[cli]>=0.32.3",
    "mlx-lm>=0.24.1",
    "numpy>=2.2.6",
    "torch>=2.7.0",
    "tqdm>=4.67.1",
    "transformers>=4.52.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/phi4_mlx_training"]
```

Install dependencies:

```bash
# Install all dependencies
uv pip install -e .

# Verify MLX installation
python -c "import mlx; print('MLX installed successfully')"
```

## Step 5: Set Up Hugging Face CLI and Authentication

First, install the Hugging Face CLI if it's not already installed:

```bash
# Add huggingface-hub with CLI support to the project
uv add "huggingface-hub[cli]"

# Test that the CLI is installed correctly
huggingface-cli whoami
# This will show "Not logged in" if you haven't authenticated yet
```

Optionally, login to Hugging Face (recommended for private models or faster downloads):

```bash
# Login to Hugging Face
huggingface-cli login
# Enter your Hugging Face token when prompted

# Verify you're logged in
huggingface-cli whoami
# This should now show your Hugging Face username
```

## Step 6: Download the Dolly 15K Dataset

The repository includes a data download script at `src/phi4-mlx-training/download_data.py`. Run it to download and prepare the training data:

```bash
python src/phi4-mlx-training/download_data.py
```

This script will:
- Download the Dolly 15K dataset from Hugging Face
- Format it for Phi-4's chat template with special tokens
- Split into 80% training and 20% validation
- Save as JSONL files in the `data/` directory

## Step 7: Configure Training Parameters

The repository includes a training configuration file at `src/phi4-mlx-training/train_config.py`:

```python
# train_config.py
TRAINING_CONFIG = {
    "model": "microsoft/Phi-4-mini-instruct",
    "data_path": "./data",
    "batch_size": 2,  # Adjust based on your RAM
    "learning_rate": 1e-4,
    "num_iters": 1000,  # Start with 1000 for testing
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

You can modify these values based on your hardware and requirements.

## Step 8: Run LoRA Fine-Tuning

### Option 1: Use the Provided Training Script

The easiest way is to use the provided training script that reads from the configuration:

```bash
# Run training with configuration
python src/phi4-mlx-training/train_phi4_lora.py

# Or make it executable and run directly
chmod +x src/phi4-mlx-training/train_phi4_lora.py
./src/phi4-mlx-training/train_phi4_lora.py
```

### Option 2: Run MLX Commands Directly

```bash
# Basic LoRA fine-tuning command
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

# Alternative: Use pre-quantized model for faster loading
python -m mlx_lm lora \
  --model mlx-community/Phi-4-mini-instruct-4bit \
  --train \
  --data ./data \
  --batch-size 4 \
  --num-layers 16 \
  --iters 1000 \
  --steps-per-report 50 \
  --steps-per-eval 200 \
  --learning-rate 1e-4 \
  --adapter-path ./adapters
```

**Note:** The MLX CLI has been updated. Use `mlx_lm lora` instead of `mlx_lm.lora`, and `--num-layers` instead of `--lora-layers`.

**Note about data files:** When you specify `--data ./data`, MLX automatically uses both `train.jsonl` and `valid.jsonl` files by convention. The training data is used to update the model weights, while the validation data is used to evaluate the model periodically (based on `--steps-per-eval`) to monitor for overfitting. This is why you'll see both "Train loss" and "Val loss" in the output - the validation loss helps ensure your model generalizes well beyond just the training examples.

## Step 9: Monitor Training Progress

The training will show output like this:

```
Loading pretrained model
Fetching 13 files: 100%|██████████| 13/13 [00:05<00:00,  2.34it/s]
Loading datasets
Training
Trainable parameters: 0.082% (3.146M/3821.080M)
Starting training..., iters: 1000
Iter 1: Val loss 3.124, Val took 8.123s
Iter 50: Train loss 2.847, Learning Rate 1.000e-04, It/sec 0.892, Tokens/sec 45.231, Trained Tokens 2547, Peak mem 8.234 GB
Iter 100: Train loss 2.456, Learning Rate 1.000e-04, It/sec 0.934, Tokens/sec 52.134, Trained Tokens 5234, Peak mem 8.456 GB
...
```

Expected training time on different Apple Silicon:
- **M1 (8GB)**: ~45-60 minutes for 1000 iterations
- **M2/M3 (16GB+)**: ~30-40 minutes for 1000 iterations  
- **M4 (16GB+)**: ~25-35 minutes for 1000 iterations

## Step 10: Test Your Fine-Tuned Model

Use the provided test script to evaluate your fine-tuned model:

```bash
# Run the test script
python src/phi4-mlx-training/test_model.py

# Or make it executable and run directly
chmod +x src/phi4-mlx-training/test_model.py
./src/phi4-mlx-training/test_model.py
```

This script will test the model with various prompts and display the responses.

## Step 11: Compare with Base Model

Use the comparison script to see the difference between the original and fine-tuned models:

```bash
# Run the comparison script
python src/phi4-mlx-training/compare_models.py

# Or make it executable and run directly
chmod +x src/phi4-mlx-training/compare_models.py
./src/phi4-mlx-training/compare_models.py
```

You can also run manual comparisons:

```bash
# Test original Phi-4 Mini
python -m mlx_lm generate \
  --model microsoft/Phi-4-mini-instruct \
  --prompt "<|user|>\nExplain machine learning in simple terms<|end|>\n<|assistant|>\n" \
  --max-tokens 300

# Test your fine-tuned version
python -m mlx_lm generate \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --prompt "<|user|>\nExplain machine learning in simple terms<|end|>\n<|assistant|>\n" \
  --max-tokens 300
```

## Step 12: Save and Export Your Model (Optional)

### Option 1: Use the Fuse Script

```bash
# Run the fuse script
python src/phi4-mlx-training/fuse_model.py

# Or make it executable and run directly
chmod +x src/phi4-mlx-training/fuse_model.py
./src/phi4-mlx-training/fuse_model.py
```

This script will merge the adapter weights with the base model and automatically test the fused model.

### Option 2: Run Commands Manually

```bash
# Fuse adapter with base model
python -m mlx_lm fuse \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --save-path ./fused-model

# Test the fused model
python -m mlx_lm generate \
  --model ./fused-model \
  --prompt "<|user|>\nWhat is artificial intelligence?<|end|>\n<|assistant|>\n" \
  --max-tokens 300
```

## Step 13: Optimize Memory Usage (If Needed)

If you encounter memory issues, try these optimizations:

```bash
# Use smaller batch size
python -m mlx_lm lora \
  --model microsoft/Phi-4-mini-instruct \
  --train \
  --data ./data \
  --batch-size 1 \
  --num-layers 8 \
  --iters 1000

# Use 4-bit quantized model
python -m mlx_lm lora \
  --model mlx-community/Phi-4-mini-instruct-4bit \
  --train \
  --data ./data \
  --batch-size 4 \
  --iters 1000
```

Or modify the `train_config.py` file to use smaller values for `batch_size` and `lora_layers`.

## Troubleshooting

### Common Issues and Solutions:

**1. Out of Memory Error:**
```bash
# Reduce batch size and LoRA layers
--batch-size 1 --num-layers 8
```

**2. Slow Training:**
```bash
# Use quantized model
--model mlx-community/Phi-4-mini-instruct-4bit
```

**3. Poor Results:**
```bash
# Increase training iterations
--iters 2000
# Adjust learning rate
--learning-rate 5e-5
```

**4. Model Not Found:**
```bash
# Update MLX
uv pip install --upgrade mlx-lm
```

**5. MLX CLI Deprecation Warnings:**
- Use `mlx_lm lora` instead of `mlx_lm.lora`
- Use `mlx_lm generate` instead of `mlx_lm.generate`
- Use `mlx_lm fuse` instead of `mlx_lm.fuse`
- Use `--num-layers` instead of `--lora-layers`

## Performance Expectations

**Training Metrics:**
- **Memory Usage**: 6-12GB depending on configuration
- **Training Speed**: ~0.8-1.2 iterations/second
- **Validation Loss**: Should decrease from ~3.0 to ~1.5-2.0

**Hardware Performance:**
- **M1 8GB**: Can handle batch_size=1-2
- **M2/M3 16GB**: Can handle batch_size=2-4  
- **M4 16GB+**: Can handle batch_size=4-8

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes, and LoRA configurations
2. **Expand the dataset**: Combine Dolly 15K with other datasets for better performance
3. **Domain-specific fine-tuning**: Use enterprise-specific data for your use case
4. **Evaluation**: Create systematic evaluation scripts to measure model performance

## Project Structure

Your final project should look like this:

```
phi4-mlx-training/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .github/
│   └── copilot-instructions.md
├── docs/
│   └── tutorial.md
├── src/
│   └── phi4-mlx-training/
│       ├── __init__.py
│       ├── download_data.py
│       ├── train_config.py
│       ├── train_phi4_lora.py
│       ├── test_model.py
│       ├── compare_models.py
│       └── fuse_model.py
├── data/
│   ├── train.jsonl
│   └── valid.jsonl
├── adapters/
│   ├── 0000100_adapters.safetensors
│   ├── ...
│   ├── adapter_config.json
│   └── adapters.safetensors
└── fused-model/ (optional)
    ├── config.json
    ├── model.safetensors
    └── tokenizer files...
```

## Important Notes

1. **Git Management**: The `adapters/` and `fused-model/` directories are automatically excluded from git via `.gitignore`. These contain large binary files that should not be version controlled.

2. **Scripts**: All Python scripts in `src/phi4-mlx-training/` are executable with the uv shebang (`#!/usr/bin/env -S uv run --script`), allowing direct execution.

3. **Configuration**: The `train_config.py` file centralizes all training parameters, making it easy to adjust settings without modifying scripts.

Congratulations! You've successfully fine-tuned Phi-4 Mini using MLX on your Mac. The model should now be better at following instructions and producing responses in the style of the Dolly 15K dataset.