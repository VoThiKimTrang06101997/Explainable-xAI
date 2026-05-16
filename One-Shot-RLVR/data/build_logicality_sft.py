# Đây là bước distillation. Nó dùng SOL/logic verifier làm teacher nhẹ. Nếu solver/verifier trả đúng gold thì giữ. Nếu chưa match, vẫn tạo target bằng gold nhưng ghi source là gold_guided_fallback.

# %%writefile /content/Explainable-xAI/One-Shot-RLVR/data/build_logicality_sft.py
import argparse
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exact_modules.logic_solver import solve_logic
from exact_modules.sol.physics_solver import solve_physics
from exact_modules.logicality_metrics import p1_correctness_continuous


def parse_obj(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            try:
                return ast.literal_eval(x)
            except Exception:
                return {}
    return {}


def prompt_to_text(prompt):
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()

    if isinstance(prompt, str):
        try:
            prompt = ast.literal_eval(prompt)
        except Exception:
            return prompt

    if isinstance(prompt, list) and len(prompt) > 0:
        first = prompt[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)

    if isinstance(prompt, dict):
        return str(prompt.get("content", ""))

    return str(prompt)


def build_physics_gold_target(question, gold):
    return {
        "answer": str(gold),
        "unit": "",
        "explanation": (
            "The answer is obtained by formalizing the physics problem, "
            "extracting the given quantities, selecting the relevant formula, "
            "checking units, and computing the final result."
        ),
        "fol": "Known quantities + relevant physics formula -> final answer",
        "cot": [
            "Problem formalization: Identify the target physical quantity.",
            "Evidence generation: Extract the known values and units from the problem.",
            "Evidence evaluation: Select the relevant formula and check unit consistency.",
            "Calculation: Substitute the values carefully into the formula.",
            f"Conclusion: The final answer is {gold}."
        ],
        "premises": [
            "The known quantities are extracted from the question.",
            "The relevant physics formula is selected according to the target quantity."
        ],
        "confidence": 0.75,
        "source": "gold_guided_physics_fallback"
    }


def build_logic_gold_target(question, gold, premises):
    return {
        "answer": str(gold),
        "explanation": (
            "The answer is derived by formalizing the logical query, "
            "extracting the relevant premises, evaluating whether the candidate answer "
            "is supported, contradicted, or undetermined, and then drawing the conclusion."
        ),
        "fol": "Premises -> supported/contradicted/unknown answer",
        "cot": [
            "Problem formalization: Identify the claim or answer options in the question.",
            "Evidence generation: Extract the relevant premises.",
            "Evidence evaluation: Check whether the candidate answer is supported, contradicted, or undetermined.",
            "Inference: Choose the answer that is best supported by the evidence.",
            f"Conclusion: The final answer is {gold}."
        ],
        "premises": premises or [],
        "confidence": 0.75,
        "source": "gold_guided_logic_fallback"
    }


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
    parser.add_argument("--keep_fallback", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)

    rows = []
    kept_by_teacher = 0
    kept_by_fallback = 0
    skipped = 0

    for _, row in df.iterrows():
        extra = parse_obj(row.get("extra_info", {}))
        reward_model = parse_obj(row.get("reward_model", {}))

        task_type = str(extra.get("task_type", row.get("ability", ""))).lower()
        question = extra.get("question", "")
        gold = reward_model.get("ground_truth", extra.get("gold_answer", ""))

        if not question:
            question = prompt_to_text(row.get("prompt", ""))

        premises = extra.get("premises_nl", []) or extra.get("premises", [])

        teacher_obj = None

        if task_type == "physics":
            try:
                sol = solve_physics(question)
                if sol is not None:
                    p1 = p1_correctness_continuous(sol.get("answer", ""), gold, "physics")
                    if p1 >= 0.8:
                        teacher_obj = sol
            except Exception:
                teacher_obj = None

            if teacher_obj is None and args.keep_fallback:
                teacher_obj = build_physics_gold_target(question, gold)
                kept_by_fallback += 1
            elif teacher_obj is not None:
                kept_by_teacher += 1

        elif task_type == "logic":
            try:
                logic_sol = solve_logic(question, premises)
                if logic_sol is not None:
                    p1 = p1_correctness_continuous(logic_sol.get("answer", ""), gold, "logic")
                    if p1 >= 1.0:
                        teacher_obj = logic_sol
            except Exception:
                teacher_obj = None

            if teacher_obj is None and args.keep_fallback:
                teacher_obj = build_logic_gold_target(question, gold, premises)
                kept_by_fallback += 1
            elif teacher_obj is not None:
                kept_by_teacher += 1

        if teacher_obj is None:
            skipped += 1
            continue

        prompt = build_training_prompt(question, task_type, premises)
        target = json.dumps(teacher_obj, ensure_ascii=False)

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
            "teacher_answer": str(teacher_obj.get("answer", "")),
            "teacher_source": teacher_obj.get("source", "unknown"),
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
    print("Kept by teacher/verifier:", kept_by_teacher)
    print("Kept by fallback:", kept_by_fallback)
    print("Skipped:", skipped)
    print("Saved jsonl:", out)
    if args.output_parquet:
        print("Saved parquet:", args.output_parquet)


if __name__ == "__main__":
    main()
    
    