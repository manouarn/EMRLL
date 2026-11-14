import os
import json
import time
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


ROOT = "/data/anouar/EMRLL/E04"

MODEL_PATH = f"{ROOT}/model/Qwen2.5-7"
TRAIN_PATH = f"{ROOT}/data/E04_grpo_training.csv"

OUTPUT_DIR = f"{ROOT}/outputs/E06_Qwen2.5-7B_QLoRA"
CHECKPOINT_DIR = f"{ROOT}/checkpoints/E06_Qwen2.5-7B_QLoRA"
RESULTS_DIR = f"{ROOT}/results/E06"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 80)
print("E06 QWEN2.5-7B QLoRA FULL TRAINING")
print("=" * 80)

# ============================================================
# 1. GPU
# ============================================================

print("\n[1/8] ENVIRONMENT")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

gpu = torch.cuda.get_device_properties(0)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", gpu.name)
print("VRAM GB:", round(gpu.total_memory / 1024**3, 2))

# ============================================================
# 2. Dataset
# ============================================================

print("\n[2/8] DATASET")

df = pd.read_csv(TRAIN_PATH)

required = {
    "id",
    "category",
    "problem",
    "answer",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if len(df) != 3000:
    raise ValueError(
        f"Expected 3000 examples, found {len(df)}"
    )

if df[list(required)].isna().any().any():
    raise ValueError("Dataset contains missing values")

print("Training examples:", len(df))
print("\nCategory counts:")
print(df["category"].value_counts().sort_index())


def make_text(row):
    return (
        "Solve the following mathematical problem.\n\n"
        f"Problem: {row['problem']}\n\n"
        f"Answer: {row['answer']}"
    )


df = df.copy()
df["text"] = df.apply(make_text, axis=1)

dataset = Dataset.from_pandas(
    df[["text"]],
    preserve_index=False,
)

print("\nDataset:", dataset)


# ============================================================
# 3. Tokenizer
# ============================================================

print("\n[3/8] TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer:", tokenizer.__class__.__name__)
print("Vocabulary:", len(tokenizer))


# ============================================================
# 4. 4-bit model
# ============================================================

print("\n[4/8] 4-BIT MODEL")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
)

model.config.use_cache = False

print("Model:", model.__class__.__name__)
print(
    "Embedding device:",
    model.get_input_embeddings().weight.device,
)


# ============================================================
# 5. QLoRA
# ============================================================

print("\n[5/8] QLoRA")

peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.4,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

print("Rank:", 4)
print("Alpha:", 8)
print("Dropout:", 0.4)
print("Targets: q_proj, k_proj, v_proj")


# ============================================================
# 6. SFT configuration
# ============================================================

print("\n[6/8] SFT CONFIGURATION")

training_args = SFTConfig(
    output_dir=CHECKPOINT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,

    learning_rate=2e-5,

    max_length=256,

    bf16=True,
    fp16=False,

    gradient_checkpointing=True,

    logging_steps=10,

    save_strategy="steps",
    save_steps=250,

    report_to="none",

    seed=SEED,

    dataset_text_field="text",

    # Keep the complete 3000-example dataset.
    packing=False,
)

print("SFTConfig: OK")


# ============================================================
# 7. Trainer
# ============================================================

print("\n[7/8] SFT TRAINER")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

print("SFTTrainer: OK")


# ============================================================
# Metadata
# ============================================================

metadata = {
    "experiment": "E06",
    "model": "Qwen2.5-7B",
    "model_path": MODEL_PATH,
    "training_data": TRAIN_PATH,
    "training_examples": 3000,
    "epochs": 1,
    "method": "QLoRA",
    "quantization": "4-bit NF4",
    "compute_dtype": "bfloat16",
    "learning_rate": 2e-5,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_length": 256,
    "lora": {
        "r": 4,
        "alpha": 8,
        "dropout": 0.4,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    },
    "gpu": torch.cuda.get_device_name(0),
    "gpu_vram_gb": round(
        gpu.total_memory / 1024**3,
        2,
    ),
    "pytorch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}

metadata_path = f"{RESULTS_DIR}/E06_metadata.json"

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)


# ============================================================
# 8. TRAIN
# ============================================================

print("\n" + "=" * 80)
print("STARTING E06 FULL QLoRA TRAINING")
print("=" * 80)

print("Examples: 3000")
print("Epochs: 1")
print("Batch size: 1")
print("Learning rate:", 2e-5)
print("LoRA rank: 4")
print("LoRA alpha: 8")
print("LoRA dropout: 0.4")
print("Quantization: 4-bit NF4")
print("Precision: BF16")
print()

start = time.time()

result = trainer.train()

elapsed = time.time() - start


# ============================================================
# Save
# ============================================================

print("\n" + "=" * 80)
print("E06 TRAINING FINISHED")
print("=" * 80)

print("Elapsed hours:", round(elapsed / 3600, 3))
print("Global steps:", result.global_step)
print("Training loss:", result.training_loss)

print("\nSaving final adapter...")

trainer.save_model(OUTPUT_DIR)
trainer.save_state()

metrics = dict(result.metrics)

metrics["elapsed_seconds"] = elapsed
metrics["elapsed_hours"] = elapsed / 3600
metrics["global_step"] = result.global_step
metrics["training_loss"] = result.training_loss

metrics_path = f"{RESULTS_DIR}/E06_training_metrics.json"

with open(metrics_path, "w") as f:
    json.dump(
        metrics,
        f,
        indent=2,
        default=float,
    )

status = {
    "experiment": "E06",
    "status": "COMPLETED",
    "global_step": int(result.global_step),
    "training_loss": float(result.training_loss),
    "elapsed_seconds": float(elapsed),
    "elapsed_hours": float(elapsed / 3600),
}

status_path = f"{RESULTS_DIR}/E06_status.json"

with open(status_path, "w") as f:
    json.dump(status, f, indent=2)

print("\nFinal adapter:", OUTPUT_DIR)
print("Metrics:", metrics_path)
print("Status:", status_path)

print("\n" + "=" * 80)
print("E06 QLoRA FULL TRAINING: COMPLETED")
print("=" * 80)
