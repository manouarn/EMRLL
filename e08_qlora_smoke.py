import os
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

MODEL_PATH = f"{ROOT}/model/DeepSeek-R1-Distill-Qwen-14B"
TRAIN_PATH = f"{ROOT}/data/E04_grpo_training.csv"

OUTPUT_DIR = f"{ROOT}/outputs/E08_DeepSeek14B_QLoRA/smoke"
CHECKPOINT_DIR = f"{ROOT}/checkpoints/E08_DeepSeek14B_QLoRA/smoke"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("E08 DEEPSEEK-R1-DISTILL-QWEN-14B QLoRA SMOKE TEST")
print("=" * 80)

# ------------------------------------------------------------
# GPU
# ------------------------------------------------------------

print("\n[1/7] GPU")

assert torch.cuda.is_available()

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB",
)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

print("\n[2/7] DATASET")

df = pd.read_csv(TRAIN_PATH)

assert len(df) == 3000

smoke_df = df.head(8).copy()

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

print("Full dataset:", df.shape)
print("Smoke dataset:", dataset)

# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

print("\n[3/7] TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer:", tokenizer.__class__.__name__)
print("Vocabulary:", len(tokenizer))

# ------------------------------------------------------------
# 4-bit model
# ------------------------------------------------------------

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
    max_memory={
        0: "20GiB",
        "cpu": "80GiB",
    },
    offload_folder=f"{ROOT}/offload_e08",
    local_files_only=True,
)

model.config.use_cache = False

print("Model:", model.__class__.__name__)
print(
    "Embedding device:",
    model.get_input_embeddings().weight.device,
)

# ------------------------------------------------------------
# QLoRA
# ------------------------------------------------------------

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

print("Rank:", 4)
print("Alpha:", 8)
print("Dropout:", 0.4)
print("Targets: q_proj, k_proj, v_proj")

# ------------------------------------------------------------
# SFT
# ------------------------------------------------------------

print("\n[6/7] SFT CONFIGURATION")

training_args = SFTConfig(
    output_dir=CHECKPOINT_DIR,
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
    seed=SEED,
    dataset_text_field="text",
    packing=False,
)

print("SFTConfig: OK")

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

print("\n[7/7] SFT TRAINER")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

print("SFTTrainer: OK")

print("\n" + "=" * 80)
print("STARTING E08 2-STEP QLoRA SMOKE TEST")
print("=" * 80)

start = time.time()

result = trainer.train()

elapsed = time.time() - start

print("\n" + "=" * 80)
print("E08 SMOKE TEST FINISHED")
print("=" * 80)

print("Elapsed seconds:", round(elapsed, 2))
print("Global steps:", result.global_step)
print("Training loss:", result.training_loss)

trainer.save_model(OUTPUT_DIR)

print("Output:", OUTPUT_DIR)

print("\n" + "=" * 80)
print("E08 QLoRA SMOKE TEST: PASSED")
print("=" * 80)
