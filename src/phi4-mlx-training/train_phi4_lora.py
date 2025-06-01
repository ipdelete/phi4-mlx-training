#!/usr/bin/env -S uv run --script

import subprocess
from train_config import TRAINING_CONFIG

subprocess.run([
    "python", "-m", "mlx_lm", "lora",
    "--model", TRAINING_CONFIG["model"],
    "--train",
    "--data", TRAINING_CONFIG["data_path"],
    "--batch-size", str(TRAINING_CONFIG["batch_size"]),
    "--num-layers", str(TRAINING_CONFIG["lora_layers"]),
    "--iters", str(TRAINING_CONFIG["num_iters"]),
    "--steps-per-report", str(TRAINING_CONFIG["steps_per_report"]),
    "--steps-per-eval", str(TRAINING_CONFIG["steps_per_eval"]),
    "--learning-rate", str(TRAINING_CONFIG["learning_rate"]),
    "--adapter-path", TRAINING_CONFIG["adapter_path"]
])