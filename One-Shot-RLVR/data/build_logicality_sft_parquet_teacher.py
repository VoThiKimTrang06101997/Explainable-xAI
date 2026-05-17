import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exact_modules.parquet_teacher import ExactParquetTeacher, parse_obj, get_premises_nl, get_premises_fol


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

Return valid JSON only with:
answer, unit, explanation, fol, cot, premises, premises_nl, premises_fol, confidence.

Problem:
{question}
"""

    premises_nl = premises_nl or []
    premises_fol = premises_fol or []

    premise_nl_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_nl)])
    premise_fol_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_fol)])

    return f"""You are an explainable logic QA system.

Use both natural-language premises and FOL premises to answer the question.

Logicality pipeline:
1. Problem formalization
2. Evidence generation
3. Evidence evaluation
4. Inference
5. Conclusion

Return valid JSON only with:
answer, explanation, fol, cot, premises, premises_nl, premises_fol, confidence.

Premises-NL:
{premise_nl_text}

Premises-FOL:
{premise_fol_text}

Question:
{question}
"""


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

        # Ensure target always has NL/FOL premise fields.
        if task_type == "logic":
            obj["premises"] = premises_nl
            obj["premises_nl"] = premises_nl
            obj["premises_fol"] = premises_fol
            obj["fol"] = "\n".join(premises_fol).strip() or obj.get("fol", "")

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
    
