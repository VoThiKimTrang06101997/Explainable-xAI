import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_LOGIC_PATHS = [
    "Logic_Based_Educational_Queries(1).json",
    "Logic_Based_Educational_Queries.json",
    "/content/Logic_Based_Educational_Queries(1).json",
    "/content/drive/MyDrive/Explainable_AI/Dataset/Logic_Based_Educational_Queries(1).json",
]

DEFAULT_PHYSICS_PATHS = [
    "Physics_Problems_Text_Only(1).csv",
    "Physics_Problems_Text_Only.csv",
    "/content/Physics_Problems_Text_Only(1).csv",
    "/content/drive/MyDrive/Explainable_AI/Dataset/Physics_Problems_Text_Only(1).csv",
]

OUT_TRAIN = Path("data/train/exact_rlvr")
OUT_VAL = Path("data/val/exact_rlvr")


def find_existing_path(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find any file from: {candidates}")


def is_nan_like(x):
    if x is None:
        return True
    s = str(x).strip().lower()
    return s in ["", "nan", "none", "null", "na"]


def normalize_answer(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ["true"]:
        return "Yes"
    if s.lower() in ["false"]:
        return "No"
    return s


def infer_answer_from_explanation(question, explanation):
    """
    Conservative label cleaner for noisy logic labels.
    Only override when explanation is explicit.
    """
    q = str(question or "")
    e = str(explanation or "").strip()

    has_options = all(opt in q for opt in ["A.", "B.", "C."])

    if has_options:
        patterns = [
            r"support(?:ing|s)? option\s+([A-D])",
            r"option\s+([A-D])\s+(?:is|as)\s+(?:correct|true|valid|logically valid)",
            r"\b([A-D])\s+is\s+(?:correct|true|valid|logically valid)",
            r"\banswer\s+is\s+([A-D])\b",
            r"supporting\s+([A-D])\b",
        ]
        for p in patterns:
            m = re.search(p, e, flags=re.I)
            if m:
                return m.group(1).upper()

        return None

    # Yes/No questions
    if re.match(r"^(yes|true)\b", e, flags=re.I):
        return "Yes"
    if re.match(r"^(no|false)\b", e, flags=re.I):
        return "No"

    if re.search(r"\bso the answer is\s+(yes|true)\b", e, flags=re.I):
        return "Yes"
    if re.search(r"\bso the answer is\s+(no|false)\b", e, flags=re.I):
        return "No"

    return None


def clean_logic_answer(question, raw_answer, explanation):
    raw = normalize_answer(raw_answer)
    inferred = infer_answer_from_explanation(question, explanation)

    if inferred is None:
        return raw, False

    # Override only suspicious labels or contradictory Yes/No.
    if raw in ["Unknown", "No", "Yes", "False", "True"] and raw != inferred:
        return inferred, True

    if raw in ["A", "B", "C", "D"] and inferred in ["A", "B", "C", "D"] and raw != inferred:
        return inferred, True

    return raw, False


def build_logic_prompt(question, premises_nl, premises_fol):
    premises_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_nl)])
    fol_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_fol)])

    return f"""You are an explainable logic QA system.

Use the given natural-language premises and FOL/rule premises to answer the question.
Return valid JSON only with:
answer, explanation, fol, cot, premises, confidence.

Natural-language premises:
{premises_text}

FOL/rule premises:
{fol_text}

Question:
{question}
"""


def build_physics_prompt(question, unit):
    return f"""You are an explainable physics QA system.

Solve the physics problem.
Return valid JSON only with:
answer, unit, explanation, fol, cot, premises, confidence.

The answer should be numeric when possible.
Expected unit if available: {unit}

Problem:
{question}
"""


def make_row(prompt, ability, gold, extra_info):
    return {
        "data_source": "exact",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": ability,
        "reward_model": {
            "ground_truth": str(gold),
            "style": "rule",
        },
        # Store as JSON string to avoid pyarrow struct/non-struct errors.
        "extra_info": json.dumps(extra_info, ensure_ascii=False),
    }


def load_logic_rows(logic_path):
    with open(logic_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    n_overridden = 0

    for item_id, item in enumerate(data):
        premises_fol = item.get("premises-FOL", []) or []
        premises_nl = item.get("premises-NL", []) or []
        questions = item.get("questions", []) or []
        answers = item.get("answers", []) or []
        explanations = item.get("explanation", []) or []

        for q_idx, question in enumerate(questions):
            raw_answer = answers[q_idx] if q_idx < len(answers) else ""
            explanation = explanations[q_idx] if q_idx < len(explanations) else ""

            clean_answer, overridden = clean_logic_answer(
                question=question,
                raw_answer=raw_answer,
                explanation=explanation,
            )
            n_overridden += int(overridden)

            prompt = build_logic_prompt(question, premises_nl, premises_fol)

            extra_info = {
                "index": f"LOGIC_{item_id}_{q_idx}",
                "task_type": "logic",
                "question": question,
                "premises_nl": premises_nl,
                "premises_fol": premises_fol,
                "gold_answer": clean_answer,
                "gold_answer_raw": normalize_answer(raw_answer),
                "gold_answer_overridden": overridden,
                "gold_explanation": explanation,
                "gold_valid": True,
            }

            rows.append(make_row(prompt, "logic", clean_answer, extra_info))

    print(f"Logic rows: {len(rows)}")
    print(f"Logic answer overrides from explanation: {n_overridden}")
    return rows


def load_physics_rows(physics_path, keep_unlabeled_physics=False):
    df = pd.read_csv(physics_path)

    rows = []
    skipped_nan = 0

    for _, r in df.iterrows():
        qid = str(r.get("id", "")).strip()
        question = str(r.get("question", "")).strip()
        cot = "" if pd.isna(r.get("cot", "")) else str(r.get("cot", ""))
        answer = r.get("answer", "")
        unit = r.get("unit", "")

        answer_is_nan = is_nan_like(answer)
        unit_is_nan = is_nan_like(unit)

        if answer_is_nan and not keep_unlabeled_physics:
            skipped_nan += 1
            continue

        gold_answer = "" if answer_is_nan else str(answer).strip()
        gold_unit = "" if unit_is_nan else str(unit).strip()

        prompt = build_physics_prompt(question, gold_unit)

        extra_info = {
            "index": qid,
            "task_type": "physics",
            "question": question,
            "gold_answer": gold_answer,
            "gold_answer_raw": "" if answer_is_nan else str(answer).strip(),
            "gold_explanation": "",
            "gold_unit": gold_unit,
            "gold_cot": cot,
            "gold_valid": not answer_is_nan,
            "id_prefix": re.sub(r"[^A-Za-z]", "", qid),
        }

        rows.append(make_row(prompt, "physics", gold_answer, extra_info))

    print(f"Physics rows kept: {len(rows)}")
    print(f"Physics rows skipped because answer is nan: {skipped_nan}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic_file", type=str, default=None)
    parser.add_argument("--physics_file", type=str, default=None)
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_unlabeled_physics", action="store_true")
    args = parser.parse_args()

    logic_path = Path(args.logic_file) if args.logic_file else find_existing_path(DEFAULT_LOGIC_PATHS)
    physics_path = Path(args.physics_file) if args.physics_file else find_existing_path(DEFAULT_PHYSICS_PATHS)

    print("Logic file:", logic_path)
    print("Physics file:", physics_path)

    logic_rows = load_logic_rows(logic_path)
    physics_rows = load_physics_rows(
        physics_path,
        keep_unlabeled_physics=args.keep_unlabeled_physics,
    )

    rows = logic_rows + physics_rows

    train_rows, val_rows = train_test_split(
        rows,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
        stratify=[r["ability"] for r in rows],
    )

    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_VAL.mkdir(parents=True, exist_ok=True)

    train_path = OUT_TRAIN / "exact_train.parquet"
    val_path = OUT_VAL / "exact_val.parquet"

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print("Total rows:", len(rows))
    print("Train rows:", len(train_rows))
    print("Val rows:", len(val_rows))
    print("Saved:", train_path)
    print("Saved:", val_path)


if __name__ == "__main__":
    main()



# Chạy câu lệnh:

# python data/prepare_exact_dataset.py \
#   --logic_file "/content/Logic_Based_Educational_Queries(1).json" \
#   --physics_file "/content/Physics_Problems_Text_Only(1).csv"

# python data/rebuild_exact_prompts.py