import os
import re
import json
import time
import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# ============================================================
# E04 GRPO SMOKE TEST
# ============================================================

E04_ROOT = "/data/anouar/EMRLL/E04"

MODEL_PATH = (
    f"{E04_ROOT}/model/"
    "DeepSeek-R1-Distill-Qwen-1.5B"
)

TRAIN_PATH = f"{E04_ROOT}/data/E04_grpo_training.csv"

OUTPUT_DIR = f"{E04_ROOT}/outputs/smoke_test"
CHECKPOINT_DIR = f"{E04_ROOT}/checkpoints/smoke_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 80)
print("E04 GRPO SMOKE TEST")
print("=" * 80)

# ------------------------------------------------------------
# 1. Environment
# ------------------------------------------------------------

print("\n[1/8] Environment")

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))

gpu = torch.cuda.get_device_properties(0)

print(
    "VRAM:",
    round(gpu.total_memory / 1024**3, 2),
    "GB"
)

# ------------------------------------------------------------
# 2. Dataset
# ------------------------------------------------------------

print("\n[2/8] Dataset")

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
        f"Missing dataset columns: {sorted(missing)}"
    )

print("Full dataset:", df.shape)

# Tiny smoke subset
smoke_df = df.head(8).copy()

print("Smoke dataset:", smoke_df.shape)

print(smoke_df[
    ["id", "category", "problem", "answer"]
].to_string(index=False))

# ------------------------------------------------------------
# 3. Convert to GRPO format
# ------------------------------------------------------------

print("\n[3/8] Preparing GRPO dataset")

def make_prompt(problem):
    return (
        "Solve the following mathematical problem.\n\n"
        f"Problem:\n{problem}\n\n"
        "Give the final numerical answer clearly."
    )

smoke_df["prompt"] = smoke_df["problem"].apply(make_prompt)

# Keep answer as string because reward comparison is textual.
smoke_df["answer"] = smoke_df["answer"].astype(str)

train_dataset = Dataset.from_pandas(
    smoke_df[
        ["prompt", "answer", "problem", "category"]
    ],
    preserve_index=False,
)

print(train_dataset)
print("Columns:", train_dataset.column_names)

# ------------------------------------------------------------
# 4. Tokenizer
# ------------------------------------------------------------

print("\n[4/8] Loading tokenizer")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer:", tokenizer.__class__.__name__)
print("Vocabulary:", len(tokenizer))

# ------------------------------------------------------------
# 5. Quantized model
# ------------------------------------------------------------

print("\n[5/8] Loading 4-bit model")

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

# ------------------------------------------------------------
# 6. Reward function
# ------------------------------------------------------------

print("\n[6/8] Reward function")

def extract_final_number(text):
    """
    Extract the last integer/decimal appearing in generated text.
    This is intentionally simple for the smoke test.
    """

    if text is None:
        return None

    text = str(text)

    # Prefer an explicit boxed answer if present.
    boxed = re.findall(
        r"\\boxed\{([^{}]+)\}",
        text,
    )

    if boxed:
        candidates = re.findall(
            r"-?\d+(?:\.\d+)?",
            boxed[-1],
        )
        if candidates:
            return candidates[-1]

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
        # TRL may provide either strings or message-like structures.
        if isinstance(completion, list):
            text_parts = []

            for item in completion:
                if isinstance(item, dict):
                    text_parts.append(
                        str(item.get("content", ""))
                    )
                else:
                    text_parts.append(str(item))

            text = " ".join(text_parts)
        else:
            text = str(completion)

        predicted = extract_final_number(text)
        target = str(target).strip()

        reward = 1.0 if predicted == target else 0.0

        rewards.append(reward)

        print(
            "\nREWARD CHECK"
            f"\nTarget: {target}"
            f"\nPredicted: {predicted}"
            f"\nReward: {reward}"
        )

    return rewards


# ------------------------------------------------------------
# 7. GRPO configuration
# ------------------------------------------------------------

print("\n[7/8] Building GRPO configuration")

training_args = GRPOConfig(
    output_dir=CHECKPOINT_DIR,

    # Tiny smoke test
    max_steps=2,

    # Memory-conscious configuration
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,

    # IMPORTANT:
    # generation_batch_size must be divisible by
    # num_generations.
    num_generations=4,
    generation_batch_size=4,

    # TRL 1.13.0 supports this.
    max_completion_length=256,

    learning_rate=5e-6,

    # Use FP16 on the RTX A5000 for this experiment.
    fp16=False,
    bf16=True,

    gradient_checkpointing=True,

    logging_steps=1,
    save_steps=1,

    save_strategy="steps",

    report_to="none",

    remove_unused_columns=False,

    seed=42,
)

# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

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

print("GRPOConfig: OK")
print("LoRA configuration: OK")

# ------------------------------------------------------------
# 8. Trainer
# ------------------------------------------------------------

print("\n[8/8] Creating GRPO trainer")

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=exact_numeric_reward,
    peft_config=peft_config,
)

print("GRPOTrainer: OK")

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("STARTING 2-STEP GRPO SMOKE TEST")
print("=" * 80)

start = time.time()

result = trainer.train()

elapsed = time.time() - start

print("\n" + "=" * 80)
print("SMOKE TEST TRAINING FINISHED")
print("=" * 80)

print("Elapsed seconds:", round(elapsed, 2))

print("\nTraining result:")
print(result)

# ------------------------------------------------------------
# Save trainer state
# ------------------------------------------------------------

trainer.save_model(OUTPUT_DIR)

trainer.save_state()

# ------------------------------------------------------------
# Save metadata
# ------------------------------------------------------------

metadata = {
    "experiment": "E04",
    "test": "GRPO smoke test",
    "model": MODEL_PATH,
    "training_data": TRAIN_PATH,
    "training_examples": len(smoke_df),
    "max_steps": 2,
    "num_generations": 4,
    "generation_batch_size": 4,
    "max_completion_length": 256,
    "learning_rate": 5e-6,
    "fp16": True,
    "bf16": False,
    "gpu": torch.cuda.get_device_name(0),
    "pytorch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}

metadata_path = (
    f"{OUTPUT_DIR}/smoke_test_metadata.json"
)

with open(metadata_path, "w") as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )

# ------------------------------------------------------------
# Final checks
# ------------------------------------------------------------

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nCheckpoint directory:")
print(CHECKPOINT_DIR)

print("\n" + "=" * 80)
print("E04 GRPO SMOKE TEST: PASSED")
print("=" * 80)
