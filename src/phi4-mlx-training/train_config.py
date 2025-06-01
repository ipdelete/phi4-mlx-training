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