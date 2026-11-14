import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


ROOT = Path("/data/anouar/EMRLL/E04")

MODEL_PATH = ROOT / "outputs/E02R_GPT2_FT"
HOLDOUT_PATH = ROOT / "data/paper_holdout.csv"
RESULTS_DIR = ROOT / "results/E02R"

RESULTS_PATH = RESULTS_DIR / "E01_holdout_results.csv"
SUMMARY_PATH = RESULTS_DIR / "E01_holdout_comparison.json"

MAX_NEW_TOKENS = 256


def extract_number(text):
    text = str(text)

    # Prefer boxed answers.
    matches = re.findall(
        r"\\boxed\{\s*(-?\d+(?:\.\d+)?)\s*\}",
        text,
    )

    if matches:
        return matches[-1]

    # Explicit answer patterns.
    patterns = [
        r"(?:final\s+answer|final)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"(?:answer)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"(?:therefore|thus|so)\s*[,:]?\s*(-?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if matches:
            return matches[-1]

    # Fallback: last numerical value.
    numbers = re.findall(
        r"(?<![\w.])-?\d+(?:\.\d+)?",
        text,
    )

    return numbers[-1] if numbers else None


def normalize_number(value):
    if value is None:
        return None

    value = float(value)

    if value.is_integer():
        return int(value)

    return value


print("=" * 70)
print("E02-R GPT-2 FINE-TUNED HOLDOUT EVALUATION")
print("=" * 70)

assert torch.cuda.is_available()

print("GPU:", torch.cuda.get_device_name(0))

df = pd.read_csv(HOLDOUT_PATH)

print("Holdout shape:", df.shape)

print("\nCategories:")
print(
    df["category"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nTokenizer:", tokenizer.__class__.__name__)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model.to("cuda")
model.eval()

print("Model:", model.__class__.__name__)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING E01 HOLDOUT EVALUATION")
print("=" * 70)

results = []

for i, row in df.iterrows():

    problem = str(row["problem"])
    expected = normalize_number(row["answer"])

    prompt = (
        "Solve the following mathematical problem. "
        "Give the final numeric answer clearly.\n\n"
        f"Problem: {problem}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to("cuda")

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    generated = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    prediction = normalize_number(
        extract_number(generated)
    )

    correct = (
        prediction is not None
        and expected is not None
        and prediction == expected
    )

    results.append({
        "id": row["id"],
        "category": row["category"],
        "problem": problem,
        "answer": expected,
        "prediction": prediction,
        "correct": correct,
        "generated": generated,
    })

    n = i + 1

    if n % 25 == 0 or n == len(df):

        accuracy = (
            sum(r["correct"] for r in results)
            / n
        )

        print(
            f"{n}/{len(df)} "
            f"accuracy={accuracy:.4f}"
        )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

result_df = pd.DataFrame(results)

result_df.to_csv(
    RESULTS_PATH,
    index=False,
)

accuracy = float(
    result_df["correct"].mean()
)

category_stats = (
    result_df
    .groupby("category")["correct"]
    .agg(["mean", "sum", "count"])
    .rename(columns={"mean": "accuracy"})
)

summary = {
    "experiment": "E02-R",
    "model": "GPT-2",
    "method": "base",
    "parameters": 124439808,
    "holdout": str(HOLDOUT_PATH),
    "examples": int(len(result_df)),
    "accuracy": accuracy,
    "correct": int(result_df["correct"].sum()),
    "max_new_tokens": MAX_NEW_TOKENS,
    "generation": {
        "do_sample": False,
    },
    "category_accuracy": {
        str(category): {
            "accuracy": float(row["accuracy"]),
            "correct": int(row["sum"]),
            "count": int(row["count"]),
        }
        for category, row
        in category_stats.iterrows()
    },
}

with open(
    SUMMARY_PATH,
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print("\n" + "-" * 70)
print("E02-R GPT-2 FINE-TUNED ACCURACY")
print(
    f"Accuracy: {accuracy:.4f}"
)
print(
    f"Correct: "
    f"{int(result_df['correct'].sum())}/"
    f"{len(result_df)}"
)

print("\nAccuracy by category:")
print(category_stats.to_string())

print("\n" + "=" * 70)
print("E01 HOLDOUT EVALUATION COMPLETE")
print("=" * 70)

print("Results:", RESULTS_PATH)
print("Summary:", SUMMARY_PATH)
