import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exact_modules.parquet_teacher import ExactParquetTeacher, parse_obj


def build_training_prompt(question, task_type, premises=None):
    if task_type == "physics":
        return f"""You are an explainable physics QA system.

Solve the problem using this scientific-logicality pipeline:
1. Problem formalization
2. Evidence generation
3. Evidence evaluation
4. Calculation
5. Conclusion

Return valid JSON only with:
answer, unit, explanation, fol, cot, premises, confidence.

Problem:
{question}
"""

    premise_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises or [])])

    return f"""You are an explainable logic QA system.

Use the premises to answer the question using this logicality pipeline:
1. Problem formalization
2. Evidence generation
3. Evidence evaluation
4. Inference
5. Conclusion

Return valid JSON only with:
answer, explanation, fol, cot, premises, confidence.

Premises:
{premise_text}

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

    for _, row in df.iterrows():
        extra = parse_obj(row.get("extra_info", {}))
        reward_model = parse_obj(row.get("reward_model", {}))

        task_type = str(extra.get("task_type", row.get("ability", ""))).lower()
        question = extra.get("question", "")
        gold = reward_model.get("ground_truth", extra.get("gold_answer", ""))
        premises = extra.get("premises_nl", []) or extra.get("premises", [])

        obj = teacher.get(question)

        if obj is None:
            continue

        prompt = build_training_prompt(question, task_type, premises)
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
    print("Saved:", out)


if __name__ == "__main__":
    main()
    

