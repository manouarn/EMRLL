from pathlib import Path
import json
import time

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

ROOT = Path("/data/anouar/EMRLL/E04")

MODEL = ROOT / "model/gpt2"
TRAIN_PATH = ROOT / "data/E02R_train.csv"
VALID_PATH = ROOT / "data/E02R_validation.csv"

OUTPUT_DIR = ROOT / "outputs/E02R_GPT2_FT"
CHECKPOINT_DIR = ROOT / "checkpoints/E02R_GPT2_FT"
RESULTS_DIR = ROOT / "results/E02R"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def make_text(row):
    return (
        "Instruction: Solve the following math problem.\n"
        f"Problem: [{row['category']}] {row['problem']}\n"
        f"Solution: {row['answer']}"
    )


print("=" * 80)
print("E02-R GPT-2 FULL SUPERVISED FINE-TUNING")
print("=" * 80)

assert torch.cuda.is_available()

print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))

train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)

assert len(train_df) == 1120
assert len(valid_df) == 280

print("Train:", train_df.shape)
print("Validation:", valid_df.shape)

print("\nTraining categories:")
print(train_df["category"].value_counts())

print("\nValidation categories:")
print(valid_df["category"].value_counts())

train_ds = Dataset.from_pandas(
    pd.DataFrame({
        "text": train_df.apply(make_text, axis=1)
    }),
    preserve_index=False,
)

valid_ds = Dataset.from_pandas(
    pd.DataFrame({
        "text": valid_df.apply(make_text, axis=1)
    }),
    preserve_index=False,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nTokenizer:", tokenizer.__class__.__name__)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
)

model.config.pad_token_id = tokenizer.pad_token_id

print("Model:", model.__class__.__name__)
print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ),
)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=1024,
    )


train_ds = train_ds.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

valid_ds = valid_ds.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


class BestLossCallback(TrainerCallback):

    def __init__(self):
        self.best = float("inf")
        self.best_step = None

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        **kwargs,
    ):
        if metrics and "eval_loss" in metrics:
            loss = metrics["eval_loss"]

            if loss < self.best:
                self.best = loss
                self.best_step = state.global_step

                print(
                    f"\nNEW BEST VALIDATION LOSS: "
                    f"{loss:.6f} "
                    f"at step {state.global_step}"
                )


args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),

    num_train_epochs=20,

    learning_rate=1e-5,
    weight_decay=0.5,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=1,

    warmup_steps=50,
    lr_scheduler_type="cosine",

    fp16=False,
    bf16=True,
    gradient_checkpointing=True,

    logging_steps=50,

    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,

    save_total_limit=3,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    seed=SEED,
    data_seed=20260825,

    report_to="none",

    optim="adamw_torch",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=collator,
    callbacks=[BestLossCallback()],
)

metadata = {
    "experiment": "E02-R",
    "type": "reproduction",
    "model": "GPT-2 small",
    "parameters": 124439808,
    "train_size": 1120,
    "validation_size": 280,
    "epochs": 20,
    "learning_rate": 1e-5,
    "weight_decay": 0.5,
    "optimizer": "AdamW",
    "scheduler": "cosine",
    "warmup_steps": 50,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_sequence_length": 1024,
    "fp16": False,
    "bf16": True,
    "gradient_checkpointing": True,
    "seed": 42,
    "data_seed": 20260825,
    "historical_reference": {
        "historical_gpu_count": 2,
        "historical_gpu": "Tesla T4",
        "historical_checkpoint": "checkpoint-6160",
        "historical_checkpoint_available": False,
    },
}

with open(
    RESULTS_DIR / "E02R_metadata.json",
    "w",
) as f:
    json.dump(metadata, f, indent=2)


print("=" * 80)
print("STARTING E02-R TRAINING")
print("=" * 80)

start = time.time()

result = trainer.train()

elapsed = time.time() - start

print("=" * 80)
print("E02-R TRAINING FINISHED")
print("=" * 80)

print("Elapsed hours:", elapsed / 3600)
print("Global steps:", result.global_step)
print("Training loss:", result.training_loss)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics = dict(result.metrics)
metrics["global_step"] = result.global_step
metrics["training_loss"] = result.training_loss
metrics["elapsed_seconds"] = elapsed
metrics["elapsed_hours"] = elapsed / 3600

with open(
    RESULTS_DIR / "E02R_training_metrics.json",
    "w",
) as f:
    json.dump(metrics, f, indent=2)

final_status = {
    "experiment": "E02-R",
    "status": "COMPLETED",
    "global_step": int(result.global_step),
    "training_loss": float(result.training_loss),
    "elapsed_seconds": elapsed,
    "elapsed_hours": elapsed / 3600,
}

with open(
    RESULTS_DIR / "E02R_status.json",
    "w",
) as f:
    json.dump(final_status, f, indent=2)

print("Final model:", OUTPUT_DIR)
print("Metrics:", RESULTS_DIR / "E02R_training_metrics.json")
print("Status:", RESULTS_DIR / "E02R_status.json")

print("=" * 80)
print("E02-R FULL TRAINING: COMPLETED")
print("=" * 80)
