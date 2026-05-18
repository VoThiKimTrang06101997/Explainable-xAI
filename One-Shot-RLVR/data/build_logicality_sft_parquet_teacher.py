import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exact_modules.parquet_teacher import (
    ExactParquetTeacher,
    parse_obj,
    get_premises_nl,
    get_premises_fol,
)


REQUIRED_TARGET_KEYS = [
    "answer",
    "explanation",
    "fol",
    "cot",
    "premises",
    "premises_nl",
    "premises_fol",
    "confidence",
]


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
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(v) for v in obj if str(v).strip()]
        except Exception:
            pass
        return [s]
    return [str(x)]


def build_training_prompt(question, task_type, premises_nl=None, premises_fol=None):
    task_type = str(task_type).lower()

    if task_type == "physics":
        return f"""You are an explainable physics QA system.

Solve the problem using this scientific-logicality pipeline:
1. Problem formalization
2. Evidence generation
3. Evidence evaluation
4. Calculation
5. Conclusion

Return valid JSON only with exactly these keys:
answer, unit, explanation, fol, cot, premises, premises_nl, premises_fol, confidence.

Problem:
{question}
"""

    premises_nl = premises_nl or []
    premises_fol = premises_fol or []

    premise_nl_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_nl)])
    premise_fol_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_fol)])

    return f"""You are an explainable logic QA system.

Use both natural-language premises and FOL/rule premises to answer the question.

Logicality pipeline:
1. Problem formalization
2. Evidence generation
3. Evidence evaluation
4. Inference
5. Conclusion

Return valid JSON only with exactly these keys:
answer, explanation, fol, cot, premises, premises_nl, premises_fol, confidence.

Premises-NL:
{premise_nl_text}

Premises-FOL:
{premise_fol_text}

Question:
{question}
"""


def compact_list(xs, max_items=20, max_chars_each=280):
    out = []
    for x in as_list(xs)[:max_items]:
        x = str(x).strip()
        if len(x) > max_chars_each:
            x = x[:max_chars_each].rstrip() + " ..."
        out.append(x)
    return out


def compact_text(x, max_chars=2200):
    x = str(x or "").strip()
    if len(x) > max_chars:
        return x[:max_chars].rstrip() + " ..."
    return x


def normalize_target_schema(obj, task_type, question, gold, extra):
    obj = dict(obj or {})
    task_type = str(task_type).lower()

    premises_nl = get_premises_nl(extra)
    premises_fol = get_premises_fol(extra)
    old_premises = as_list(obj.get("premises", []))

    if task_type == "logic":
        if not premises_nl:
            premises_nl = old_premises

        premises_nl = compact_list(premises_nl, max_items=20, max_chars_each=260)
        premises_fol = compact_list(premises_fol, max_items=20, max_chars_each=260)

        obj["premises"] = premises_nl
        obj["premises_nl"] = premises_nl
        obj["premises_fol"] = premises_fol

        if premises_fol:
            obj["fol"] = compact_text("\n".join(premises_fol), max_chars=2500)
        else:
            obj["fol"] = compact_text(obj.get("fol", "") or "Premises -> answer", max_chars=1200)

    else:
        physics_premises = old_premises if old_premises else [
            "Official physics problem statement.",
            "Official physics dataset answer."
        ]

        physics_premises = compact_list(physics_premises, max_items=8, max_chars_each=260)

        obj["premises"] = physics_premises
        obj["premises_nl"] = physics_premises
        obj["premises_fol"] = []
        obj["fol"] = compact_text(obj.get("fol", "") or "Physics formula/rule -> answer", max_chars=1200)

        if "unit" not in obj:
            obj["unit"] = str(extra.get("gold_unit", "") or "")

    obj["answer"] = str(obj.get("answer", gold)).strip()
    obj["explanation"] = compact_text(
        obj.get("explanation", "")
        or extra.get("gold_explanation", "")
        or extra.get("gold_cot", "")
        or "The answer is derived by formalizing the problem, extracting evidence, evaluating the rule or formula, and drawing the final conclusion.",
        max_chars=1600,
    )

    cot = obj.get("cot", [])
    if not isinstance(cot, list) or len(cot) == 0:
        if task_type == "physics":
            cot = [
                "Problem formalization: Identify the target physical quantity.",
                "Evidence generation: Extract known quantities and units.",
                "Evidence evaluation: Select the relevant formula.",
                "Calculation: Apply the formula or official derivation.",
                f"Conclusion: The final answer is {obj['answer']}."
            ]
        else:
            cot = [
                "Problem formalization: Identify the logical claim or answer options.",
                "Evidence generation: Extract the relevant NL/FOL premises.",
                "Evidence evaluation: Compare the claim or options against the premises.",
                "Inference: Select the supported answer.",
                f"Conclusion: The final answer is {obj['answer']}."
            ]

    obj["cot"] = compact_list(cot, max_items=6, max_chars_each=260)

    try:
        obj["confidence"] = float(obj.get("confidence", 0.99))
    except Exception:
        obj["confidence"] = 0.99

    if "source" not in obj:
        obj["source"] = "normalized_sft_target"

    for key in REQUIRED_TARGET_KEYS:
        if key not in obj:
            if key in ["premises", "premises_nl", "premises_fol", "cot"]:
                obj[key] = []
            elif key == "confidence":
                obj[key] = 0.99
            else:
                obj[key] = ""

    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)
    teacher = ExactParquetTeacher(args.input_parquet)

    rows = []
    missing_teacher = 0

    for _, row in df.iterrows():
        extra = parse_obj(row.get("extra_info", {}))
        reward_model = parse_obj(row.get("reward_model", {}))

        task_type = str(extra.get("task_type", row.get("ability", ""))).lower()
        question = extra.get("question", "")
        gold = reward_model.get("ground_truth", extra.get("gold_answer", ""))

        premises_nl = get_premises_nl(extra)
        premises_fol = get_premises_fol(extra)

        obj = teacher.get(question)

        if obj is None:
            missing_teacher += 1
            continue

        obj = normalize_target_schema(
            obj=obj,
            task_type=task_type,
            question=question,
            gold=gold,
            extra=extra,
        )

        prompt = build_training_prompt(
            question=question,
            task_type=task_type,
            premises_nl=premises_nl,
            premises_fol=premises_fol,
        )

        target = json.dumps(obj, ensure_ascii=False)

        rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target}
            ],
            "prompt": prompt,
            "target": target,
            "task_type": task_type,
            "question": question,
            "gold_answer": str(gold),
            "teacher_answer": str(obj.get("answer", "")),
            "teacher_source": obj.get("source", "unknown"),
        })

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.output_parquet:
        pd.DataFrame(rows).to_parquet(args.output_parquet, index=False)

    print("Input rows:", len(df))
    print("SFT rows:", len(rows))
    print("Missing teacher:", missing_teacher)
    print("Saved:", out)

    if args.output_parquet:
        print("Saved parquet:", args.output_parquet)


if __name__ == "__main__":
    main()
    

