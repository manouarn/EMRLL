import os
import re
import json
import torch
import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path("/data/anouar/EMRLL/E04")
MODEL_PATH = ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER_PATH = ROOT / "outputs/E04_GRPO_v1"
HOLDOUT_PATH = ROOT / "data/paper_holdout.csv"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_RESULTS = RESULTS_DIR / "E04_holdout_base.csv"
GRPO_RESULTS = RESULTS_DIR / "E04_holdout_grpo.csv"
SUMMARY_PATH = RESULTS_DIR / "E04_holdout_comparison.json"

MAX_NEW_TOKENS = 256


def extract_number(text):
    text = str(text)

    # Prefer the final integer/decimal in the response.
    matches = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text)

    if not matches:
        return None

    return matches[-1]


def load_model(with_adapter=False):
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 4-bit base model...")

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
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
    )

    if with_adapter:
        print("Loading E04 LoRA adapter...")
        model = PeftModel.from_pretrained(
            model,
            ADAPTER_PATH,
            is_trainable=False,
        )

    model.eval()

    return tokenizer, model


def make_prompt(problem):
    return (
        "Solve the following math problem.\n"
        "Give the final numeric answer clearly.\n\n"
        f"Problem: {problem}\n\n"
        "Answer:"
    )


@torch.inference_mode()
def evaluate(model_name, with_adapter):
    print("\n" + "=" * 70)
    print(f"E04 HOLDOUT EVALUATION: {model_name}")
    print("=" * 70)

    df = pd.read_csv(HOLDOUT_PATH)

    print("Holdout examples:", len(df))

    tokenizer, model = load_model(with_adapter)

    rows = []

    for i, row in df.iterrows():
        problem = str(row["problem"])
        target = str(row["answer"])

        prompt = make_prompt(problem)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        predicted = extract_number(generated)

        target_number = extract_number(target)

        correct = (
            predicted is not None
            and target_number is not None
            and predicted == target_number
        )

        rows.append({
            "id": row["id"],
            "category": row["category"],
            "problem": problem,
            "answer": target,
            "prediction": predicted,
            "generated": generated,
            "correct": correct,
        })

        if (i + 1) % 25 == 0:
            accuracy = sum(x["correct"] for x in rows) / len(rows)
            print(
                f"{i + 1:3d}/{len(df)} "
                f"accuracy={accuracy:.4f}"
            )

    result_df = pd.DataFrame(rows)

    accuracy = float(result_df["correct"].mean())

    category_accuracy = (
        result_df
        .groupby("category")["correct"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )

    print("\n" + "-" * 70)
    print(f"{model_name} accuracy: {accuracy:.4f}")
    print(
        f"Correct: {int(result_df['correct'].sum())}/"
        f"{len(result_df)}"
    )

    print("\nAccuracy by category:")
    print(category_accuracy.to_string(index=False))

    if with_adapter:
        result_df.to_csv(GRPO_RESULTS, index=False)
    else:
        result_df.to_csv(BASE_RESULTS, index=False)

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "accuracy": accuracy,
        "correct": int(result_df["correct"].sum()),
        "total": len(result_df),
        "category_accuracy": {
            str(row["category"]): {
                "accuracy": float(row["mean"]),
                "correct": int(row["sum"]),
                "total": int(row["count"]),
            }
            for _, row in category_accuracy.iterrows()
        },
    }


print("=" * 70)
print("E04 PAPER HOLDOUT EVALUATION")
print("=" * 70)

base_result = evaluate(
    "BASE MODEL",
    with_adapter=False,
)

grpo_result = evaluate(
    "E04 GRPO ADAPTER",
    with_adapter=True,
)

improvement = grpo_result["accuracy"] - base_result["accuracy"]

summary = {
    "holdout": str(HOLDOUT_PATH),
    "total_examples": 250,
    "base": base_result,
    "grpo": grpo_result,
    "accuracy_difference": improvement,
    "accuracy_difference_percentage_points": improvement * 100,
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("E04 HOLDOUT EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Base accuracy: {base_result['accuracy']:.4f} "
    f"({base_result['correct']}/{base_result['total']})"
)

print(
    f"GRPO accuracy: {grpo_result['accuracy']:.4f} "
    f"({grpo_result['correct']}/{grpo_result['total']})"
)

print(
    f"Difference: {improvement:+.4f} "
    f"({improvement * 100:+.2f} percentage points)"
)

print("\nResults:")
print(BASE_RESULTS)
print(GRPO_RESULTS)
print(SUMMARY_PATH)
