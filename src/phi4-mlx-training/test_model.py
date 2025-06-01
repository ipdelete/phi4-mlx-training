#!/usr/bin/env -S uv run --script

import subprocess

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
            "python", "-m", "mlx_lm", "generate",
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