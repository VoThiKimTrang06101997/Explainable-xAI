# %%writefile /content/Explainable-xAI/One-Shot-RLVR/data/prepare_exact_dataset.py
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

DEFAULT_LOGIC_PATHS = [
    "Logic_Based_Educational_Queries.json",
    "/content/Explainable-xAI/One-Shot-RLVR/data/exact/raw/Logic_Based_Educational_Queries.json",
    "/content/Explainable-xAI/One-Shot-RLVR/data/raw/uploaded_exact/Logic_Based_Educational_Queries.json",
    "/content/drive/MyDrive/Explainable_AI/Dataset/Logic_Based_Educational_Queries_Text_Only/Logic_Based_Educational_Queries.json",
]

DEFAULT_PHYSICS_PATHS = [
    "Physics_Problems_Text_Only.csv",
    "/content/Explainable-xAI/One-Shot-RLVR/data/exact/raw/Physics_Problems_Text_Only.csv",
    "/content/Explainable-xAI/One-Shot-RLVR/data/raw/uploaded_exact/Physics_Problems_Text_Only.csv",
    "/content/drive/MyDrive/Explainable_AI/Dataset/Physics_Problems_Text_Only/Physics_Problems_Text_Only.csv",
]

OUT_TRAIN = Path("/content/Explainable-xAI/One-Shot-RLVR/data/train/exact_rlvr")
OUT_VAL = Path("/content/Explainable-xAI/One-Shot-RLVR/data/val/exact_rlvr")


# ============================================================
# Generic helpers
# ============================================================

def find_existing_path(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find any file from: {candidates}")


def is_nan_like(x):
    if x is None:
        return True

    try:
        if pd.isna(x):
            return True
    except Exception:
        pass

    s = str(x).strip().lower()
    return s in ["", "nan", "none", "null", "na"]


def normalize_answer(x):
    if x is None:
        return ""

    s = str(x).strip()

    if s.lower() == "true":
        return "Yes"
    if s.lower() == "false":
        return "No"
    if s.lower() == "uncertain":
        return "Unknown"
    if s.lower() == "unknown":
        return "Unknown"

    return s


def as_list(x):
    if x is None:
        return []

    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []

        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
        except Exception:
            pass

        return [s]

    return [str(x)]


def safe_get_list(item, keys):
    for key in keys:
        if key in item:
            values = as_list(item.get(key))
            if values:
                return values
    return []


def safe_get_value(row, col, default=""):
    if col is None:
        return default

    try:
        val = row.get(col, default)
    except Exception:
        return default

    if is_nan_like(val):
        return default

    return val


# ============================================================
# Logic answer cleaner
# ============================================================

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
            r"the correct answer is\s+([A-D])\b",
            r"supporting\s+([A-D])\b",
        ]

        for p in patterns:
            m = re.search(p, e, flags=re.I)
            if m:
                return m.group(1).upper()

        return None

    # Yes / No / Unknown questions
    if re.match(r"^(yes|true)\b", e, flags=re.I):
        return "Yes"
    if re.match(r"^(no|false)\b", e, flags=re.I):
        return "No"
    if re.match(r"^(unknown|uncertain)\b", e, flags=re.I):
        return "Unknown"

    if re.search(r"\bso the answer is\s+(yes|true)\b", e, flags=re.I):
        return "Yes"
    if re.search(r"\bso the answer is\s+(no|false)\b", e, flags=re.I):
        return "No"
    if re.search(r"\bso the answer is\s+(unknown|uncertain)\b", e, flags=re.I):
        return "Unknown"

    if re.search(r"\btherefore\s*,?\s+(yes|true)\b", e, flags=re.I):
        return "Yes"
    if re.search(r"\btherefore\s*,?\s+(no|false)\b", e, flags=re.I):
        return "No"
    if re.search(r"\btherefore\s*,?\s+(unknown|uncertain)\b", e, flags=re.I):
        return "Unknown"

    return None


def clean_logic_answer(question, raw_answer, explanation):
    raw = normalize_answer(raw_answer)
    inferred = infer_answer_from_explanation(question, explanation)

    if inferred is None:
        return raw, False

    # Override only suspicious labels or explicit contradictions.
    if raw in ["Unknown", "No", "Yes", "False", "True"] and raw != inferred:
        return inferred, True

    if raw in ["A", "B", "C", "D"] and inferred in ["A", "B", "C", "D"] and raw != inferred:
        return inferred, True

    return raw, False


# ============================================================
# Prompt builders
# ============================================================

def build_logic_prompt(question, premises_nl, premises_fol):
    premises_nl = as_list(premises_nl)
    premises_fol = as_list(premises_fol)

    premises_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_nl)])
    fol_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_fol)])

    return f"""You are an explainable logic QA system.

Use both natural-language premises and FOL/rule premises to answer the question.

Logicality pipeline:
1. Problem formalization
2. Evidence generation
3. Evidence evaluation
4. Inference
5. Conclusion

Return valid JSON only with:
answer, explanation, fol, cot, premises, premises_nl, premises_fol, confidence.

Natural-language premises:
{premises_text}

FOL/rule premises:
{fol_text}

Question:
{question}
"""


def build_physics_prompt(question, unit):
    return f"""You are an explainable physics QA system.

Solve the physics problem using this scientific-logicality pipeline:
1. Problem formalization
2. Evidence generation
3. Evidence evaluation
4. Calculation
5. Conclusion

Return valid JSON only with:
answer, unit, explanation, fol, cot, premises, premises_nl, premises_fol, confidence.

The answer should be numeric when possible.
Expected unit if available: {unit}

Problem:
{question}
"""


# ============================================================
# Row maker
# ============================================================

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


# ============================================================
# Logic loader
# ============================================================

def load_logic_rows(logic_path):
    with open(logic_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    n_overridden = 0
    n_missing_answer = 0
    n_missing_fol = 0
    n_missing_nl = 0

    for item_id, item in enumerate(data):
        premises_fol = safe_get_list(item, ["premises-FOL", "premises_fol", "premisesFOL"])
        premises_nl = safe_get_list(item, ["premises-NL", "premises_nl", "premisesNL", "premises"])

        questions = as_list(item.get("questions", []) or [])
        answers = as_list(item.get("answers", []) or [])
        explanations = as_list(item.get("explanation", []) or item.get("explanations", []) or [])

        if not premises_fol:
            n_missing_fol += 1
        if not premises_nl:
            n_missing_nl += 1

        for q_idx, question in enumerate(questions):
            raw_answer = answers[q_idx] if q_idx < len(answers) else ""
            explanation = explanations[q_idx] if q_idx < len(explanations) else ""

            clean_answer, overridden = clean_logic_answer(
                question=question,
                raw_answer=raw_answer,
                explanation=explanation,
            )

            if is_nan_like(clean_answer):
                n_missing_answer += 1
                clean_answer = ""

            n_overridden += int(overridden)

            prompt = build_logic_prompt(question, premises_nl, premises_fol)

            extra_info = {
                "index": f"LOGIC_{item_id}_{q_idx}",
                "task_type": "logic",
                "question": question,

                # Keep all versions for downstream modules.
                "premises": premises_nl,
                "premises_nl": premises_nl,
                "premises_fol": premises_fol,
                "premises-NL": premises_nl,
                "premises-FOL": premises_fol,

                "gold_answer": clean_answer,
                "gold_answer_raw": normalize_answer(raw_answer),
                "gold_answer_overridden": overridden,
                "gold_explanation": explanation,
                "gold_cot": "",
                "gold_valid": not is_nan_like(clean_answer),

                "item_id": item_id,
                "question_id": q_idx,
            }

            rows.append(make_row(prompt, "logic", clean_answer, extra_info))

    print(f"Logic rows: {len(rows)}")
    print(f"Logic answer overrides from explanation: {n_overridden}")
    print(f"Logic missing/empty answers: {n_missing_answer}")
    print(f"Logic items missing premises-FOL: {n_missing_fol}")
    print(f"Logic items missing premises-NL: {n_missing_nl}")

    return rows


# ============================================================
# Physics loader
# ============================================================

def detect_physics_columns(df):
    cols_lower = {c.lower().strip(): c for c in df.columns}

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
            key = c.lower().strip()
            if key in cols_lower:
                return cols_lower[key]
        return None

    q_col = find_col(["question", "questions", "problem", "text", "prompt"])
    a_col = find_col(["answer", "answers", "gold_answer", "final_answer"])
    u_col = find_col(["unit", "gold_unit"])
    cot_col = find_col(["cot", "CoT", "chain_of_thought", "solution", "explanation", "gold_cot"])
    id_col = find_col(["id", "qid", "index"])

    return q_col, a_col, u_col, cot_col, id_col


def fallback_physics_explanation(cot, question, answer, unit):
    cot = "" if is_nan_like(cot) else str(cot).strip()

    if cot:
        return cot

    answer = "" if is_nan_like(answer) else str(answer).strip()
    unit = "" if is_nan_like(unit) else str(unit).strip()

    if answer:
        return (
            "The answer is derived by identifying the target physical quantity, "
            "extracting the given values, selecting the appropriate formula, "
            "checking unit consistency, and calculating the final result "
            f"as {answer} {unit}."
        ).strip()

    return (
        "The answer is derived by formalizing the physics problem, extracting the relevant quantities, "
        "selecting the appropriate formula, checking units, and drawing the final conclusion."
    )


def load_physics_rows(physics_path, keep_unlabeled_physics=False):
    df = pd.read_csv(physics_path)

    q_col, a_col, u_col, cot_col, id_col = detect_physics_columns(df)

    if q_col is None:
        raise ValueError(f"Cannot detect physics question column. Columns: {list(df.columns)}")
    if a_col is None:
        raise ValueError(f"Cannot detect physics answer column. Columns: {list(df.columns)}")

    rows = []
    skipped_nan = 0
    skipped_empty_question = 0

    for idx, r in df.iterrows():
        qid = str(safe_get_value(r, id_col, f"PHY_{idx}") if id_col else f"PHY_{idx}").strip()
        question = str(safe_get_value(r, q_col, "")).strip()

        if not question:
            skipped_empty_question += 1
            continue

        answer = safe_get_value(r, a_col, "")
        unit = safe_get_value(r, u_col, "") if u_col else ""
        cot = safe_get_value(r, cot_col, "") if cot_col else ""

        answer_is_nan = is_nan_like(answer)
        unit_is_nan = is_nan_like(unit)

        if answer_is_nan and not keep_unlabeled_physics:
            skipped_nan += 1
            continue

        gold_answer = "" if answer_is_nan else str(answer).strip()
        gold_unit = "" if unit_is_nan else str(unit).strip()
        gold_cot = "" if is_nan_like(cot) else str(cot).strip()
        gold_explanation = fallback_physics_explanation(
            cot=gold_cot,
            question=question,
            answer=gold_answer,
            unit=gold_unit,
        )

        prompt = build_physics_prompt(question, gold_unit)

        physics_premises_nl = [
            "Official physics problem statement.",
            "Official physics dataset answer."
        ]

        extra_info = {
            "index": qid,
            "task_type": "physics",
            "question": question,

            "gold_answer": gold_answer,
            "gold_answer_raw": gold_answer,
            "gold_explanation": gold_explanation,
            "gold_unit": gold_unit,
            "gold_cot": gold_cot,
            "gold_valid": not answer_is_nan,

            # Keep unified premise fields for downstream modules.
            "premises": physics_premises_nl,
            "premises_nl": physics_premises_nl,
            "premises_fol": [],
            "premises-NL": physics_premises_nl,
            "premises-FOL": [],

            "id_prefix": re.sub(r"[^A-Za-z]", "", qid),
            "row_id": int(idx),
        }

        rows.append(make_row(prompt, "physics", gold_answer, extra_info))

    print(f"Physics rows kept: {len(rows)}")
    print(f"Physics rows skipped because answer is nan: {skipped_nan}")
    print(f"Physics rows skipped because question is empty: {skipped_empty_question}")

    return rows


# ============================================================
# Main
# ============================================================

def save_parquet(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print("Saved:", path)


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

    if not rows:
        raise RuntimeError("No rows were loaded. Please check input files.")

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

    save_parquet(train_rows, train_path)
    save_parquet(val_rows, val_path)

    print("Total rows:", len(rows))
    print("Train rows:", len(train_rows))
    print("Val rows:", len(val_rows))


if __name__ == "__main__":
    main()
    