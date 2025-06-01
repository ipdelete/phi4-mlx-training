# Complete Tutorial: Fine-tuning Phi-4 Mini with Dolly 15K using MLX on Mac

## Prerequisites
- Mac with Apple Silicon (M1, M2, M3, or M4)
- macOS 12.0 or later
- At least 16GB RAM (32GB+ recommended)
- 50GB+ free disk space

## Step 1: Install uv Package Manager

First, install uv, which is a fast Python package manager:

```bash
# Install uv using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart your terminal or source your shell profile
source ~/.zshrc  # or ~/.bash_profile if using bash

# Verify installation
uv --version
```

## Step 2: Create Project Directory and Virtual Environment

```bash
# Create project directory
mkdir phi4-mlx-training
cd phi4-mlx-training

# Initialize a new uv project
uv init .

# Create and activate virtual environment with Python 3.11
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
```

## Step 3: Install Required Dependencies

Create a `pyproject.toml` file for dependency management:

```toml
[project]
name = "phi4-mlx-training"
version = "0.1.0"
description = "Fine-tuning Phi-4 Mini with MLX"
dependencies = [
    "mlx-lm>=0.21.0",
    "huggingface-hub>=0.20.0",
    "transformers>=4.36.0",
    "datasets>=2.14.0",
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "tqdm>=4.64.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Install dependencies:

```bash
# Install all dependencies
uv pip install -e .

# Verify MLX installation
python -c "import mlx; print('MLX installed successfully')"
```

## Step 4: Set Up Hugging Face Authentication

```bash
# Install Hugging Face CLI
uv pip install huggingface-hub[cli]

# Login to Hugging Face (optional but recommended)
huggingface-cli login
# Enter your Hugging Face token when prompted
```

## Step 5: Download the Dolly 15K Dataset

Create a data download script:

```python
# save as download_data.py
import json
from datasets import load_dataset
import os

def download_and_prepare_dolly():
    print("Downloading Dolly 15K dataset...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("databricks/databricks-dolly-15k")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Convert to MLX format
    mlx_data = []
    
    for item in dataset['train']:
        # Format instruction for Phi-4 Mini
        if item['context']:
            prompt = f"<|user|>\n{item['instruction']}\n\nContext: {item['context']}<|end|>\n<|assistant|>\n"
        else:
            prompt = f"<|user|>\n{item['instruction']}<|end|>\n<|assistant|>\n"
        
        completion = f"{item['response']}<|end|>"
        
        mlx_data.append({
            "text": prompt + completion
        })
    
    # Split into train and validation (80/20)
    train_size = int(len(mlx_data) * 0.8)
    train_data = mlx_data[:train_size]
    valid_data = mlx_data[train_size:]
    
    # Save as JSONL files
    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open("data/valid.jsonl", "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Dataset prepared: {len(train_data)} training samples, {len(valid_data)} validation samples")
    print("Files saved: data/train.jsonl, data/valid.jsonl")

if __name__ == "__main__":
    download_and_prepare_dolly()
```

Run the download script:

```bash
python download_data.py
```

## Step 6: Verify Data Format

Create a verification script to check your data:

```python
# save as verify_data.py
import json

def verify_data():
    # Check train.jsonl
    with open("data/train.jsonl", "r") as f:
        train_lines = f.readlines()
    
    # Check valid.jsonl
    with open("data/valid.jsonl", "r") as f:
        valid_lines = f.readlines()
    
    print(f"Training samples: {len(train_lines)}")
    print(f"Validation samples: {len(valid_lines)}")
    
    # Show first example
    first_example = json.loads(train_lines[0])
    print("\nFirst training example:")
    print(first_example["text"][:500] + "...")
    
    return True

if __name__ == "__main__":
    verify_data()
```

```bash
python verify_data.py
```

## Step 7: Create Training Configuration

Create a training configuration file:

```python
# save as train_config.py
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

## Step 8: Run LoRA Fine-Tuning

Now start the fine-tuning process:

```bash
# Basic LoRA fine-tuning command
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

# Alternative: Use pre-quantized model for faster loading
python -m mlx_lm.lora \
  --model mlx-community/Phi-4-mini-instruct-4bit \
  --train \
  --data ./data \
  --batch-size 4 \
  --lora-layers 16 \
  --iters 1000 \
  --steps-per-report 50 \
  --steps-per-eval 200 \
  --learning-rate 1e-4 \
  --adapter-path ./adapters
```

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

Create a test script:

```python
# save as test_model.py
import subprocess
import sys

def test_fine_tuned_model():
    test_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the benefits of renewable energy?",
        "How do you make a perfect cup of coffee?",
    ]
    
    print("Testing fine-tuned Phi-4 Mini model...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}: {prompt}")
        print("-" * 50)
        
        # Run inference with adapter
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", "microsoft/Phi-4-mini-instruct",
            "--adapter-path", "./adapters",
            "--prompt", f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
            "--max-tokens", "500",
            "--temp", "0.7"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("Request timed out")
        
        print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    test_fine_tuned_model()
```

```bash
python test_model.py
```

## Step 11: Compare with Base Model

Test the original model for comparison:

```bash
# Test original Phi-4 Mini
python -m mlx_lm.generate \
  --model microsoft/Phi-4-mini-instruct \
  --prompt "<|user|>\nExplain machine learning in simple terms<|end|>\n<|assistant|>\n" \
  --max-tokens 300

# Test your fine-tuned version
python -m mlx_lm.generate \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --prompt "<|user|>\nExplain machine learning in simple terms<|end|>\n<|assistant|>\n" \
  --max-tokens 300
```

## Step 12: Save and Export Your Model (Optional)

Merge the adapter into the base model for standalone use:

```bash
# Fuse adapter with base model
python -m mlx_lm.fuse \
  --model microsoft/Phi-4-mini-instruct \
  --adapter-path ./adapters \
  --save-path ./fused-model

# Test the fused model
python -m mlx_lm.generate \
  --model ./fused-model \
  --prompt "<|user|>\nWhat is artificial intelligence?<|end|>\n<|assistant|>\n" \
  --max-tokens 300
```

## Step 13: Optimize Memory Usage (If Needed)

If you encounter memory issues, try these optimizations:

```bash
# Use smaller batch size
python -m mlx_lm.lora \
  --model microsoft/Phi-4-mini-instruct \
  --train \
  --data ./data \
  --batch-size 1 \
  --lora-layers 8 \
  --iters 1000

# Use 4-bit quantized model
python -m mlx_lm.lora \
  --model mlx-community/Phi-4-mini-instruct-4bit \
  --train \
  --data ./data \
  --batch-size 4 \
  --iters 1000
```

## Troubleshooting

### Common Issues and Solutions:

**1. Out of Memory Error:**
```bash
# Reduce batch size and LoRA layers
--batch-size 1 --lora-layers 8
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
├── pyproject.toml
├── download_data.py
├── verify_data.py
├── train_config.py
├── test_model.py
├── data/
│   ├── train.jsonl
│   └── valid.jsonl
├── adapters/
│   ├── adapters.npz
│   ├── adapter_config.json
│   └── adapter_weights.00.npz
└── fused-model/ (optional)
    ├── config.json
    ├── model.safetensors
    └── tokenizer files...
```

Congratulations! You've successfully fine-tuned Phi-4 Mini using MLX on your Mac. The model should now be better at following instructions and producing responses in the style of the Dolly 15K dataset.