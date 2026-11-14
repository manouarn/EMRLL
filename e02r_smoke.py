from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

ROOT = Path("/data/anouar/EMRLL/E04")
MODEL = ROOT / "model/gpt2"
DATA = ROOT / "data/E02R_train.csv"

print("=" * 70)
print("E02-R GPT-2 FULL FINE-TUNING SMOKE TEST")
print("=" * 70)

assert torch.cuda.is_available()

print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_csv(DATA).head(8)

print("Smoke examples:", len(df))

texts = []

for _, row in df.iterrows():
    text = (
        "Instruction: Solve the following math problem.\n"
        f"Problem: [{row['category']}] {row['problem']}\n"
        f"Solution: {row['answer']}"
    )
    texts.append(text)

dataset = Dataset.from_dict({"text": texts})

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer:", tokenizer.__class__.__name__)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
)

model.config.pad_token_id = tokenizer.pad_token_id

print("Model:", model.__class__.__name__)
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=1024,
    )

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

args = TrainingArguments(
    output_dir=str(ROOT / "outputs/E02R_smoke"),
    max_steps=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    weight_decay=0.5,
    fp16=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=collator,
)

print("=" * 70)
print("STARTING 2-STEP SMOKE TEST")
print("=" * 70)

result = trainer.train()

print("=" * 70)
print("E02-R SMOKE TEST FINISHED")
print("=" * 70)

print("Global steps:", result.global_step)
print("Training loss:", result.training_loss)

print("=" * 70)
print("E02-R SMOKE TEST: PASSED")
print("=" * 70)
