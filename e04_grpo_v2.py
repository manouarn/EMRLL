import os
import re
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
from trl import GRPOConfig, GRPOTrainer


# ============================================================
# E04 GRPO FULL EXPERIMENT
# ============================================================

E04_ROOT = "/data/anouar/EMRLL/E04"

MODEL_PATH = (
    f"{E04_ROOT}/model/"
    "DeepSeek-R1-Distill-Qwen-1.5B"
)

TRAIN_PATH = (
    f"{E04_ROOT}/data/E04_grpo_training_v2.csv"
)

OUTPUT_DIR = (
    f"{E04_ROOT}/outputs/E04_GRPO_v1"
)

CHECKPOINT_DIR = (
    f"{E04_ROOT}/checkpoints/E04_GRPO_v1"
)

RESULTS_DIR = (
    f"{E04_ROOT}/results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Experiment configuration
# ============================================================

NUM_EPOCHS = 1

TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 1

NUM_GENERATIONS = 4
GENERATION_BATCH_SIZE = 4

MAX_COMPLETION_LENGTH = 128

LEARNING_RATE = 5e-6

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LOGGING_STEPS = 10
SAVE_STEPS = 250

# ============================================================
# Header
# ============================================================

print("=" * 80)
print("E04 GRPO FULL EXPERIMENT")
print("=" * 80)

print("Root:", E04_ROOT)
print("Model:", MODEL_PATH)
print("Training data:", TRAIN_PATH)
print("Output:", OUTPUT_DIR)
print("Checkpoints:", CHECKPOINT_DIR)

print()


# ============================================================
# 1. Environment
# ============================================================

print("[1/9] ENVIRONMENT")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

gpu = torch.cuda.get_device_properties(0)

print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("GPU:", gpu.name)
print("VRAM GB:", round(gpu.total_memory / 1024**3, 2))
print("Compute capability:", f"{gpu.major}.{gpu.minor}")

if gpu.major < 8:
    raise RuntimeError(
        "This E04 configuration expects an Ampere-or-newer GPU."
    )

print("CUDA: OK")
print()


# ============================================================
# 2. Dataset
# ============================================================

print("[2/9] DATASET")

if not os.path.isfile(TRAIN_PATH):
    raise FileNotFoundError(TRAIN_PATH)

df = pd.read_csv(TRAIN_PATH)

required_columns = {
    "id",
    "category",
    "problem",
    "answer",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

if len(df) != 3000:
    raise ValueError(
        f"Expected exactly 3000 training examples; "
        f"found {len(df)}"
    )

if df[list(required_columns)].isna().any().any():
    raise ValueError(
        "Training dataset contains missing values."
    )

print("Training rows:", len(df))
print("Columns:", list(df.columns))

print()
print("Category counts:")
print(df["category"].value_counts().sort_index())

print()


# ============================================================
# 3. Dataset preparation
# ============================================================

print("[3/9] DATASET PREPARATION")


def make_prompt(problem):
    return (
        "Solve the following mathematical problem.\n\n"
        f"Problem:\n{problem}\n\n"
        "Give the final numerical answer clearly and concisely."
    )


df = df.copy()

df["prompt"] = df["problem"].astype(str).apply(make_prompt)
df["answer"] = df["answer"].astype(str).str.strip()

train_dataset = Dataset.from_pandas(
    df[
        [
            "prompt",
            "answer",
            "problem",
            "category",
        ]
    ],
    preserve_index=False,
)

print(train_dataset)
print("GRPO columns:", train_dataset.column_names)
print()


# ============================================================
# 4. Tokenizer
# ============================================================

print("[4/9] TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer:", tokenizer.__class__.__name__)
print("Vocabulary:", len(tokenizer))
print("PAD:", tokenizer.pad_token)
print("EOS:", tokenizer.eos_token)
print()


# ============================================================
# 5. 4-bit model
# ============================================================

print("[5/9] 4-BIT MODEL")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
    local_files_only=True,
    trust_remote_code=True,
)

model.config.use_cache = False

print("Model:", model.__class__.__name__)
print(
    "Embedding device:",
    model.get_input_embeddings().weight.device,
)
print(
    "Parameter dtype:",
    next(model.parameters()).dtype,
)

print()


# ============================================================
# 6. Reward
# ============================================================

print("[6/9] REWARD FUNCTION")


def extract_final_number(text):
    """
    Extract the final numerical answer from generated text.
    """

    if text is None:
        return None

    text = str(text)

    # First look for \boxed{...}
    boxed = re.findall(
        r"\\boxed\{([^{}]+)\}",
        text,
    )

    if boxed:
        numbers = re.findall(
            r"-?\d+(?:\.\d+)?",
            boxed[-1],
        )

        if numbers:
            return numbers[-1]

    # Otherwise use the final number in the response.
    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not numbers:
        return None

    return numbers[-1]


def exact_numeric_reward(
    completions,
    answer,
    **kwargs,
):
    rewards = []

    for completion, target in zip(
        completions,
        answer,
    ):

        if isinstance(completion, list):

            parts = []

            for item in completion:

                if isinstance(item, dict):
                    parts.append(
                        str(item.get("content", ""))
                    )
                else:
                    parts.append(str(item))

            text = " ".join(parts)

        else:
            text = str(completion)

        predicted = extract_final_number(text)
        target = str(target).strip()

        reward = 1.0 if predicted == target else 0.0

        rewards.append(reward)

    return rewards


print("Reward function: OK")
print()


# ============================================================
# 7. GRPO configuration
# ============================================================

print("[7/9] GRPO CONFIGURATION")

training_args = GRPOConfig(
    output_dir=CHECKPOINT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,

    # IMPORTANT:
    # generation_batch_size must be divisible by
    # num_generations.
    num_generations=NUM_GENERATIONS,
    generation_batch_size=GENERATION_BATCH_SIZE,

    max_completion_length=MAX_COMPLETION_LENGTH,

    learning_rate=LEARNING_RATE,

    # Consistent BF16 path.
    fp16=False,
    bf16=True,

    gradient_checkpointing=True,

    logging_steps=LOGGING_STEPS,

    save_strategy="steps",
    save_steps=SAVE_STEPS,

    report_to="none",

    remove_unused_columns=False,

    seed=SEED,
)

print("GRPOConfig: OK")
print()


# ============================================================
# LoRA
# ============================================================

print("[8/9] LoRA")

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)

print("LoRA: OK")
print()


# ============================================================
# Trainer
# ============================================================

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=exact_numeric_reward,
    peft_config=peft_config,
)

print("GRPOTrainer: OK")
print()


# ============================================================
# Save experiment metadata BEFORE training
# ============================================================

metadata = {
    "experiment": "E04_GRPO_v1",

    "seed": SEED,

    "model_path": MODEL_PATH,

    "training_data": TRAIN_PATH,
    "training_examples": int(len(df)),

    "num_train_epochs": NUM_EPOCHS,

    "per_device_train_batch_size": TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION,

    "num_generations": NUM_GENERATIONS,
    "generation_batch_size": GENERATION_BATCH_SIZE,

    "max_completion_length": MAX_COMPLETION_LENGTH,

    "learning_rate": LEARNING_RATE,

    "quantization": {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": True,
    },

    "precision": {
        "fp16": False,
        "bf16": True,
    },

    "lora": {
        "r": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
    },

    "gpu": torch.cuda.get_device_name(0),
    "gpu_vram_gb": round(
        gpu.total_memory / 1024**3,
        2,
    ),

    "pytorch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}

metadata_path = (
    f"{RESULTS_DIR}/E04_GRPO_v1_metadata.json"
)

with open(metadata_path, "w") as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )

print("Metadata:", metadata_path)
print()


# ============================================================
# TRAIN
# ============================================================

print("=" * 80)
print("STARTING COMPLETE E04 GRPO RUN")
print("=" * 80)

print("Examples:", len(df))
print("Epochs:", NUM_EPOCHS)
print("Generations:", NUM_GENERATIONS)
print("Generation batch:", GENERATION_BATCH_SIZE)
print("Max completion:", MAX_COMPLETION_LENGTH)
print("Precision: BF16")
print("Quantization: 4-bit NF4")
print()

start = time.time()

try:

    train_result = trainer.train()

except KeyboardInterrupt:

    print("\nTRAINING INTERRUPTED")

    raise

except Exception as exc:

    print("\nTRAINING FAILED")
    print(type(exc).__name__)
    print(str(exc))

    raise

elapsed = time.time() - start


# ============================================================
# SAVE FINAL ADAPTER
# ============================================================

print()
print("=" * 80)
print("TRAINING FINISHED")
print("=" * 80)

print(
    "Elapsed:",
    round(elapsed / 3600, 3),
    "hours",
)

print("Global steps:", train_result.global_step)
print("Training loss:", train_result.training_loss)

print()
print("Saving final adapter...")

trainer.save_model(OUTPUT_DIR)
trainer.save_state()


# ============================================================
# Save training metrics
# ============================================================

metrics = dict(train_result.metrics)

metrics["elapsed_seconds"] = elapsed
metrics["elapsed_hours"] = elapsed / 3600
metrics["global_step"] = train_result.global_step
metrics["training_loss"] = train_result.training_loss

metrics_path = (
    f"{RESULTS_DIR}/E04_GRPO_v1_training_metrics.json"
)

with open(metrics_path, "w") as f:
    json.dump(
        metrics,
        f,
        indent=2,
        default=float,
    )


# ============================================================
# Final status
# ============================================================

status = {
    "experiment": "E04_GRPO_v1",
    "status": "COMPLETED",
    "global_step": int(train_result.global_step),
    "training_loss": float(train_result.training_loss),
    "elapsed_seconds": float(elapsed),
    "elapsed_hours": float(elapsed / 3600),
}

status_path = (
    f"{RESULTS_DIR}/E04_GRPO_v1_status.json"
)

with open(status_path, "w") as f:
    json.dump(
        status,
        f,
        indent=2,
    )


print()
print("Final adapter:", OUTPUT_DIR)
print("Training metrics:", metrics_path)
print("Status:", status_path)

print()
print("=" * 80)
print("E04 GRPO FULL RUN: COMPLETED")
print("=" * 80)
