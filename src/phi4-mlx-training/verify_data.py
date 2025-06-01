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