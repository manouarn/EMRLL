from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

ROOT = Path("/data/anouar/EMRLL/E04")
MODEL = ROOT / "model/gpt2"
DATA = ROOT / "data/E02R_train.csv"

print("=" * 70)
print("E02-R2 GPT-2 BF16 STABILITY TEST")
print("=" * 70)

assert torch.cuda.is_available()

print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_csv(DATA).head(8)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer:", tokenizer.__class__.__name__)
print("EOS:", repr(tokenizer.eos_token))
print("EOS ID:", tokenizer.eos_token_id)

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


def prepare(example):

    prompt = (
        "Instruction: Solve the following math problem.\n"
        f"Problem: [{example['category']}] "
        f"{example['problem']}\n"
        "Solution:"
    )

    answer = f" {example['answer']}{tokenizer.eos_token}"

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    answer_ids = tokenizer(
        answer,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prompt_ids + answer_ids

    labels = (
        [-100] * len(prompt_ids)
        + answer_ids
    )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


dataset = Dataset.from_pandas(
    df,
    preserve_index=False,
)

dataset = dataset.map(
    prepare,
    remove_columns=dataset.column_names,
)

print("Smoke dataset:", len(dataset))

for i in range(min(2, len(dataset))):
    ids = dataset[i]["input_ids"]
    labels = dataset[i]["labels"]

    print("=" * 70)
    print("Example", i)

    print(
        "TEXT:",
        tokenizer.decode(ids)
    )

    print(
        "TRAINED TOKENS:",
        tokenizer.decode([
            x for x in ids
            if labels[ids.index(x)] != -100
        ])
    )

# Simple padding collator.
def collate(features):

    max_len = max(
        len(x["input_ids"])
        for x in features
    )

    input_ids = []
    attention = []
    labels = []

    for x in features:

        pad = max_len - len(x["input_ids"])

        input_ids.append(
            x["input_ids"]
            + [tokenizer.pad_token_id] * pad
        )

        attention.append(
            x["attention_mask"]
            + [0] * pad
        )

        labels.append(
            x["labels"]
            + [-100] * pad
        )

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
    }


args = TrainingArguments(
    output_dir=str(
        ROOT / "outputs/E02R1_smoke"
    ),
    max_steps=20,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    weight_decay=0.5,
    fp16=False,
    bf16=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate,
)

print("=" * 70)
print("STARTING E02-R2 20-STEP STABILITY TEST")
print("=" * 70)

result = trainer.train()

print("=" * 70)
print("E02-R2 STABILITY TEST FINISHED")
print("=" * 70)

print("Steps:", result.global_step)
print("Training loss:", result.training_loss)

print("=" * 70)
print("E02-R2 STABILITY TEST: PASSED")
print("=" * 70)
