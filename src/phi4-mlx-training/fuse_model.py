#!/usr/bin/env -S uv run --script

import subprocess
from train_config import TRAINING_CONFIG

def fuse_model():
    fused_model_path = "./fused-model"
    
    print("=" * 70)
    print("FUSING ADAPTER WITH BASE MODEL")
    print("=" * 70)
    print(f"Base model: {TRAINING_CONFIG['model']}")
    print(f"Adapter path: {TRAINING_CONFIG['adapter_path']}")
    print(f"Output path: {fused_model_path}")
    print()
    
    # Fuse adapter with base model
    result = subprocess.run([
        "python", "-m", "mlx_lm", "fuse",
        "--model", TRAINING_CONFIG["model"],
        "--adapter-path", TRAINING_CONFIG["adapter_path"],
        "--save-path", fused_model_path
    ])
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("TESTING FUSED MODEL")
        print("=" * 70)
        
        test_prompt = "What is artificial intelligence?"
        formatted_prompt = f"<|user|>\n{test_prompt}<|end|>\n<|assistant|>\n"
        
        # Test the fused model
        subprocess.run([
            "python", "-m", "mlx_lm", "generate",
            "--model", fused_model_path,
            "--prompt", formatted_prompt,
            "--max-tokens", "300"
        ])
    else:
        print("Error: Failed to fuse model")

if __name__ == "__main__":
    fuse_model()