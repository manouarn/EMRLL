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

MODEL_PATH = ROOT / "model/DeepSeek-R1-Distill-Qwen-14B"
HOLDOUT_PATH = ROOT / "data/paper_holdout.csv"
RESULTS_DIR = ROOT / "results/E07"

RESULTS_PATH = RESULTS_DIR / "E07_holdout_results.csv"
SUMMARY_PATH = RESULTS_DIR / "E07_holdout_comparison.json"

MAX_NEW_TOKENS = 256


def extract_number(text):
    text = str(text)

    boxed = re.findall(
        r"\\boxed\{\s*(-?\d+(?:\.\d+)?)\s*\}",
        text,
    )

    if boxed:
        return boxed[-1]

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


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("E07 DEEPSEEK-R1-DISTILL-QWEN-14B HOLDOUT")
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

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nTokenizer: OK")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading 14B model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={
            0: "20GiB",
            "cpu": "80GiB",
        },
        offload_folder=str(
            ROOT / "offload_e07"
        ),
        local_files_only=True,
    )

    model.eval()

    print("Model:", model.__class__.__name__)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING E07 HOLDOUT EVALUATION")
    print("=" * 70)

    results = []

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
        "experiment": "E07",
        "model": "DeepSeek-R1-Distill-Qwen-14B",
        "method": "base",
        "holdout": str(HOLDOUT_PATH),
        "examples": int(len(result_df)),
        "accuracy": accuracy,
        "correct": int(
            result_df["correct"].sum()
        ),
        "max_new_tokens": MAX_NEW_TOKENS,
        "quantization": "4-bit NF4",
        "compute_dtype": "bfloat16",
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
    print("E07 ACCURACY")
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
    print("E07 HOLDOUT EVALUATION COMPLETE")
    print("=" * 70)

    print(RESULTS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
