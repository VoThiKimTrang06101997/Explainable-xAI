# %%writefile /content/Explainable-xAI/One-Shot-RLVR/data/build_gold_aligned_logicality_sft.py
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exact_modules.logicality_distill_teacher import (
    safe_parse_obj,
    build_gold_aligned_teacher,
    get_premises_nl,
    get_premises_fol,
    compact_list,
)


def build_prompt(question, task_type, extra):
    task_type = str(task_type).lower()

    if task_type == "physics":
        return f"""You are an explainable physics QA system.

Solve the problem using scientific logicality:
1. Problem formalization
2. Evidence generation
3. Model/formula generation
4. Evidence evaluation and unit checking
5. Calculation
6. Conclusion

Return valid JSON only with exactly these keys:
answer, unit, explanation, fol, cot, premises, premises_nl, premises_fol, confidence.

Problem:
{question}
"""

    premises_nl = compact_list(get_premises_nl(extra), max_items=30, max_chars_each=320)
    premises_fol = compact_list(get_premises_fol(extra), max_items=30, max_chars_each=320)

    nl_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_nl)])
    fol_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises_fol)])

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
{nl_text}

Premises-FOL:
{fol_text}

Question:
{question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)
    if args.max_rows:
        df = df.head(args.max_rows)

    rows = []
    stats = {
        "total": 0,
        "physics": 0,
        "logic": 0,
        "solver_used": 0,
        "solver_verified": 0,
        "gold_guided": 0,
    }

    for _, row in df.iterrows():
        extra = safe_parse_obj(row.get("extra_info", {}))
        reward_model = safe_parse_obj(row.get("reward_model", {}))

        if not isinstance(extra, dict):
            extra = {}

        if not isinstance(reward_model, dict):
            reward_model = {}

        task_type = str(extra.get("task_type", row.get("ability", ""))).lower()
        question = str(extra.get("question", "") or "")

        gold = (
            reward_model.get("ground_truth")
            or extra.get("gold_answer")
            or extra.get("answer")
            or ""
        )

        if not question or str(gold).strip() == "":
            continue

        target_obj = build_gold_aligned_teacher(
            question=question,
            task_type=task_type,
            gold_answer=str(gold),
            extra=extra,
        )

        prompt = build_prompt(question, task_type, extra)
        target = json.dumps(target_obj, ensure_ascii=False)

        rows.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            "prompt": prompt,
            "target": target,
            "task_type": task_type,
            "question": question,
            "gold_answer": str(gold),
            "teacher_answer": str(target_obj.get("answer", "")),
            "teacher_source": target_obj.get("source", "unknown"),
            "solver_used": bool(target_obj.get("solver_used", False)),
            "solver_verified": bool(target_obj.get("solver_verified", False)),
        })

        stats["total"] += 1
        if task_type in stats:
            stats[task_type] += 1
        if target_obj.get("solver_used", False):
            stats["solver_used"] += 1
        if target_obj.get("solver_verified", False):
            stats["solver_verified"] += 1
        if "gold_guided" in str(target_obj.get("source", "")):
            stats["gold_guided"] += 1

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Saved jsonl:", out_jsonl)

    if args.output_parquet:
        out_parquet = Path(args.output_parquet)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_parquet, index=False)
        print("Saved parquet:", out_parquet)

    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
    
    