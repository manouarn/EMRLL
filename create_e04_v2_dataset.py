import random
from pathlib import Path

import pandas as pd


ROOT = Path("/data/anouar/EMRLL/E04")
SOURCE = ROOT / "data/E04_grpo_training.csv"
OUTPUT = ROOT / "data/E04_grpo_training_v2.csv"

SEED = 20260921
random.seed(SEED)


def make_problem(r1, r2, mode):
    # (x-r1)(x-r2) = x^2 -(r1+r2)x + r1*r2
    b = -(r1 + r2)
    c = r1 * r2

    b_text = f"+ {b}x" if b >= 0 else f"- {abs(b)}x"
    c_text = f"+ {c}" if c >= 0 else f"- {abs(c)}"

    equation = f"x^2 {b_text} {c_text} = 0"

    if mode == "positive":
        positive = r1 if r1 > 0 else r2
        problem = (
            f"For the quadratic equation {equation}, "
            f"determine the positive integer root."
        )
        answer = positive

    elif mode == "larger":
        problem = f"Find the larger integer root of {equation}."
        answer = max(r1, r2)

    elif mode == "quadratic_formula":
        problem = (
            f"Solve the quadratic equation {equation}. "
            f"Give the larger integer solution."
        )
        answer = max(r1, r2)

    elif mode == "integer_solution":
        positive = r1 if r1 > 0 else r2
        problem = (
            f"What is the positive integer solution of {equation}?"
        )
        answer = positive

    else:
        raise ValueError(mode)

    return problem, str(answer)


# Keep the four non-quadratic categories unchanged.
df = pd.read_csv(SOURCE)

non_quadratic = df[df["category"] != "quadratic"].copy()

assert len(non_quadratic) == 2400

for category in [
    "word_problem",
    "linear_equation",
    "arithmetic",
    "multi_step",
]:
    assert len(non_quadratic[non_quadratic["category"] == category]) == 600


# Generate 600 more representative quadratic examples.
quadratic_rows = []
seen = set()

modes = (
    ["positive"] * 180
    + ["larger"] * 180
    + ["quadratic_formula"] * 120
    + ["integer_solution"] * 120
)

random.shuffle(modes)

while len(quadratic_rows) < 600:
    r1 = random.randint(-45, 45)
    r2 = random.randint(-45, 45)

    if r1 == r2:
        continue

    # Avoid zero roots and overly trivial equations.
    if r1 == 0 or r2 == 0:
        continue

    if r1 + r2 == 0:
        continue

    mode = modes[len(quadratic_rows)]

    # Positive-root questions should have exactly one positive root.
    if mode in {"positive", "integer_solution"}:
        if not ((r1 > 0 and r2 < 0) or (r2 > 0 and r1 < 0)):
            continue

    key = (r1, r2, mode)

    if key in seen:
        continue

    seen.add(key)

    problem, answer = make_problem(r1, r2, mode)

    quadratic_rows.append({
        "id": f"V2Q{len(quadratic_rows) + 1:03d}",
        "category": "quadratic",
        "problem": problem,
        "answer": answer,
    })


quadratic = pd.DataFrame(quadratic_rows)

assert len(quadratic) == 600
assert quadratic["answer"].notna().all()


# Combine into exactly 3000 examples.
result = pd.concat(
    [non_quadratic, quadratic],
    ignore_index=True,
)

assert len(result) == 3000

expected_counts = {
    "word_problem": 600,
    "linear_equation": 600,
    "arithmetic": 600,
    "multi_step": 600,
    "quadratic": 600,
}

assert result["category"].value_counts().to_dict() == expected_counts

result.to_csv(OUTPUT, index=False)

print("=" * 70)
print("E04 V2 DATASET CREATED")
print("=" * 70)
print("Output:", OUTPUT)
print("Shape:", result.shape)
print()
print(result["category"].value_counts())
print()
print("Sample quadratic examples:")
print(
    quadratic[["problem", "answer"]]
    .head(10)
    .to_string(index=False)
)
