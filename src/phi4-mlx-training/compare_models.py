#!/usr/bin/env -S uv run --script

import subprocess
from train_config import TRAINING_CONFIG

def compare_models():
    test_prompt = "Explain machine learning in simple terms"
    formatted_prompt = f"<|user|>\n{test_prompt}<|end|>\n<|assistant|>\n"
    
    print("=" * 70)
    print("TESTING ORIGINAL PHI-4 MINI")
    print("=" * 70)
    
    # Test original model
    subprocess.run([
        "python", "-m", "mlx_lm", "generate",
        "--model", TRAINING_CONFIG["model"],
        "--prompt", formatted_prompt,
        "--max-tokens", "300"
    ])
    
    print("\n" + "=" * 70)
    print("TESTING FINE-TUNED VERSION")
    print("=" * 70)
    
    # Test fine-tuned model with adapter
    subprocess.run([
        "python", "-m", "mlx_lm", "generate",
        "--model", TRAINING_CONFIG["model"],
        "--adapter-path", TRAINING_CONFIG["adapter_path"],
        "--prompt", formatted_prompt,
        "--max-tokens", "300"
    ])

if __name__ == "__main__":
    compare_models()