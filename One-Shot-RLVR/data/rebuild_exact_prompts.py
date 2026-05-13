import json
from pathlib import Path

import pandas as pd


TRAIN_IN = Path("data/train/exact_rlvr/exact_train.parquet")
VAL_IN = Path("data/val/exact_rlvr/exact_val.parquet")

TRAIN_OUT = Path("data/train/exact_rlvr/exact_train_router.parquet")
VAL_OUT = Path("data/val/exact_rlvr/exact_val_router.parquet")


def parse_extra(x):
    if isinstance(x, dict):
        return x

    try:
        return json.loads(x)
    except Exception:
        return {}


def get_content(prompt):
    if isinstance(prompt, list) and len(prompt) > 0:
        if isinstance(prompt[0], dict):
            return str(prompt[0].get("content", ""))

    return str(prompt)


def build_prompt(row):
    extra = parse_extra(row["extra_info"])
    task_type = extra.get("task_type", row.get("ability", ""))
    old_content = get_content(row["prompt"])

    if task_type == "physics":
        schema = """{
  "answer": "final numeric answer only",
  "explanation": "explain the formula, substitutions, and final computation",
  "fol": "symbolic formula used, e.g., V = IR",
  "cot": [
    "Step 1: identify known quantities",
    "Step 2: choose formula",
    "Step 3: substitute values",
    "Step 4: compute answer"
  ],
  "premises": [
    "formula used",
    "known quantities"
  ],
  "confidence": 0.0
}"""

        instruction = f"""You are an explainable physics QA system.

Task type: physics.

Solve the problem and return valid JSON only.
The answer must be numeric when possible.
Always include formula-based explanation, CoT steps, premises, and confidence.

Required JSON schema:
{schema}

Problem:
{extra.get("question", old_content)}
"""

    else:
        premises_nl = extra.get("premises_nl", [])
        premises_text = "\n".join([f"- {p}" for p in premises_nl]) if isinstance(premises_nl, list) else ""

        schema = """{
  "answer": "A/B/C/D or Yes/No/Unknown",
  "explanation": "explain which premises support the answer",
  "fol": "FOL/rule representation if available",
  "cot": [
    "Step 1: identify relevant premises",
    "Step 2: convert them into logical rules",
    "Step 3: apply the rules",
    "Step 4: derive the answer"
  ],
  "premises": [
    "premise used 1",
    "premise used 2"
  ],
  "confidence": 0.0
}"""

        instruction = f"""You are an explainable logic QA system.

Task type: logic.

Use the premises to answer the question.
Return valid JSON only.
Always include explanation, FOL/rule, CoT steps, premises, and confidence.

Premises:
{premises_text}

Required JSON schema:
{schema}

Question:
{extra.get("question", old_content)}
"""

    return [{"role": "user", "content": instruction}]


def rebuild(in_path, out_path):
    df = pd.read_parquet(in_path)
    df["prompt"] = df.apply(build_prompt, axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print("Saved:", out_path, df.shape)


def main():
    rebuild(TRAIN_IN, TRAIN_OUT)
    rebuild(VAL_IN, VAL_OUT)


if __name__ == "__main__":
    main()
    