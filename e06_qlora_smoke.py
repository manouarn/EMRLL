import os
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
from trl import SFTTrainer, SFTConfig


ROOT = "/data/anouar/EMRLL/E04"

MODEL_PATH = f"{ROOT}/model/Qwen2.5-7"
TRAIN_PATH = f"{ROOT}/data/E04_grpo_training.csv"

OUTPUT_DIR = f"{ROOT}/outputs/E06_Qwen2.5-7B_QLoRA/smoke"
CHECKPOINT_DIR = f"{ROOT}/checkpoints/E06_Qwen2.5-7B_QLoRA/smoke"


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


print("=" * 80)
print("E06 QWEN2.5-7B QLoRA SMOKE TEST")
print("=" * 80)


# ============================================================
# 1. GPU
# ============================================================

print("\n[1/7] GPU")

assert torch.cuda.is_available()

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        2,
    ),
    "GB",
)


# ============================================================
# 2. Dataset
# ============================================================

print("\n[2/7] DATASET")

df = pd.read_csv(TRAIN_PATH)

required = {
    "id",
    "category",
    "problem",
    "answer",
}

assert required.issubset(df.columns)

smoke_df = df.head(8).copy()

print("Full dataset:", df.shape)
print("Smoke dataset:", smoke_df.shape)


def make_text(row):
    return (
        "Solve the following mathematical problem.\n\n"
        f"Problem: {row['problem']}\n\n"
        f"Answer: {row['answer']}"
    )


smoke_df["text"] = smoke_df.apply(make_text, axis=1)

dataset = Dataset.from_pandas(
    smoke_df[["text"]],
    preserve_index=False,
)

print(dataset)


# ============================================================
# 3. Tokenizer
# ============================================================

print("\n[3/7] TOKENIZER")

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

print("\n[4/7] 4-BIT MODEL")

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

print("\n[5/7] QLoRA")

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

print("LoRA rank:", 4)
print("LoRA alpha:", 8)
print("LoRA dropout:", 0.4)
print("Target modules: q_proj, k_proj, v_proj")


# ============================================================
# 6. SFT configuration
# ============================================================

print("\n[6/7] SFT CONFIGURATION")

training_args = SFTConfig(
    output_dir=CHECKPOINT_DIR,

    # Tiny smoke test
    max_steps=2,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,

    learning_rate=2e-5,

    num_train_epochs=1,

    max_length=256,

    bf16=True,
    fp16=False,

    gradient_checkpointing=True,

    logging_steps=1,

    save_strategy="steps",
    save_steps=1,

    report_to="none",

    seed=42,

    dataset_text_field="text",
)

print("SFTConfig: OK")


# ============================================================
# 7. Trainer
# ============================================================

print("\n[7/7] SFT TRAINER")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

print("SFTTrainer: OK")


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 80)
print("STARTING E06 2-STEP QLoRA SMOKE TEST")
print("=" * 80)

start = time.time()

result = trainer.train()

elapsed = time.time() - start

print("\n" + "=" * 80)
print("E06 SMOKE TEST FINISHED")
print("=" * 80)

print("Elapsed seconds:", round(elapsed, 2))
print("Global steps:", result.global_step)
print("Training loss:", result.training_loss)

trainer.save_model(OUTPUT_DIR)

print("\nOutput:", OUTPUT_DIR)

print("\n" + "=" * 80)
print("E06 QLoRA SMOKE TEST: PASSED")
print("=" * 80)
