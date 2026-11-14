import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

ROOT = Path("/data/anouar/EMRLL/E04")

MODEL_PATH = ROOT / "model/Qwen2.5-7"
HOLDOUT_PATH = ROOT / "data/paper_holdout.csv"
RESULTS_DIR = ROOT / "results/E05"

BASE_RESULTS = RESULTS_DIR / "E05_holdout_results.csv"
SUMMARY_PATH = RESULTS_DIR / "E05_holdout_comparison.json"

MAX_NEW_TOKENS = 256


def extract_number(text):
    """
    Extract the final integer/decimal answer from generated text.
    Prefer boxed answers and explicit answer/final-answer patterns.
    """
    patterns = [
        r"\\boxed\{([-+]?\d+(?:\.\d+)?)\}",
        r"(?:final answer|answer)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        r"([-+]?\d+(?:\.\d+)?)\s*$",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return float(matches[-1])

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return float(numbers[-1])

    return None


def normalize_number(x):
    if x is None:
        return None

    x = float(x)

    if x.is_integer():
        return int(x)

    return x


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E05 QWEN2.5-7B BASE HOLDOUT EVALUATION")
    print("=" * 70)

    assert torch.cuda.is_available(), "CUDA unavailable"

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2,
        ),
        "GB",
    )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    df = pd.read_csv(HOLDOUT_PATH)

    print("\nDataset:", HOLDOUT_PATH)
    print("Shape:", df.shape)
    print("\nCategories:")
    print(df["category"].value_counts().to_string())

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer: OK")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading Qwen2.5-7B in 4-bit...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    )

    model.eval()

    print("Model:", model.__class__.__name__)
    print(
        "Embedding device:",
        model.get_input_embeddings().weight.device,
    )

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    results = []

    print("\n" + "=" * 70)
    print("STARTING E05 HOLDOUT EVALUATION")
    print("=" * 70)

    for i, row in df.iterrows():
        problem = str(row["problem"])
        expected = normalize_number(row["answer"])

        prompt = (
            "Solve the following problem. "
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
            )

        generated_tokens = output[0][inputs["input_ids"].shape[1]:]

        generated = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        prediction = normalize_number(extract_number(generated))

        correct = (
            prediction is not None
            and expected is not None
            and prediction == expected
        )

        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "problem": problem,
                "answer": expected,
                "prediction": prediction,
                "correct": correct,
                "generated": generated,
            }
        )

        n = i + 1

        if n % 25 == 0 or n == len(df):
            accuracy = sum(r["correct"] for r in results) / n
            print(
                f"{n}/{len(df)} accuracy={accuracy:.4f}"
            )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    result_df = pd.DataFrame(results)
    result_df.to_csv(BASE_RESULTS, index=False)

    overall_accuracy = float(result_df["correct"].mean())

    category_stats = (
        result_df.groupby("category")["correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
    )

    summary = {
        "experiment": "E05",
        "model": "Qwen2.5-7B",
        "model_path": str(MODEL_PATH),
        "evaluation_dataset": str(HOLDOUT_PATH),
        "num_examples": int(len(result_df)),
        "overall_accuracy": overall_accuracy,
        "correct": int(result_df["correct"].sum()),
        "max_new_tokens": MAX_NEW_TOKENS,
        "quantization": "4-bit NF4",
        "compute_dtype": "bfloat16",
        "sampling": False,
        "category_accuracy": {
            category: {
                "accuracy": float(row["accuracy"]),
                "correct": int(row["sum"]),
                "count": int(row["count"]),
            }
            for category, row in category_stats.iterrows()
        },
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 70)
    print("E05 QWEN2.5-7B BASE ACCURACY")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(
        f"Correct: {int(result_df['correct'].sum())}/{len(result_df)}"
    )

    print("\nAccuracy by category:")
    print(category_stats.to_string())

    print("\n" + "=" * 70)
    print("E05 HOLDOUT EVALUATION COMPLETE")
    print("=" * 70)

    print("\nResults:")
    print(BASE_RESULTS)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
