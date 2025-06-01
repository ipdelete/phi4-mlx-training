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