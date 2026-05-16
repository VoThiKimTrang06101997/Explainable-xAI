# %%writefile /content/Explainable-xAI/One-Shot-RLVR/data/build_logicality_sft_dataset_teacher.py
import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exact_modules.dataset_teacher import ExactDatasetTeacher
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
            parsed = ast.literal_eval(prompt)
            prompt = parsed
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


def get_gold_explanation(extra):
    gold_exp = str(extra.get("gold_explanation", "") or "").strip()
    gold_cot = str(extra.get("gold_cot", "") or "").strip()

    if gold_exp:
        return gold_exp

    if gold_cot:
        return gold_cot

    return (
        "The answer is derived by formalizing the problem, extracting the relevant evidence, "
        "evaluating the evidence, performing the calculation or inference, and drawing the final conclusion."
    )


def build_physics_gold_target(question, gold, extra=None):
    extra = extra or {}
    gold_unit = str(extra.get("gold_unit", "") or "")

    return {
        "answer": str(gold),
        "unit": gold_unit,
        "explanation": get_gold_explanation(extra),
        "fol": "Official gold annotation: physics problem -> answer",
        "cot": [
            "Problem formalization: Identify the target physical quantity.",
            "Evidence generation: Extract the known values and units from the problem.",
            "Evidence evaluation: Select the relevant formula and check unit consistency.",
            "Calculation: Apply the formula or official derivation.",
            f"Conclusion: The final answer is {gold} {gold_unit}.".strip()
        ],
        "premises": [
            "Official physics problem statement.",
            "Official gold answer from the dataset."
        ],
        "confidence": 0.95,
        "source": "gold_guided_physics_fallback"
    }


def build_logic_gold_target(question, gold, premises, extra=None):
    extra = extra or {}

    return {
        "answer": str(gold),
        "explanation": get_gold_explanation(extra),
        "fol": "Official gold annotation: premises -> answer",
        "cot": [
            "Problem formalization: Identify the claim or answer options in the question.",
            "Evidence generation: Extract the relevant premises.",
            "Evidence evaluation: Check whether the candidate answer is supported, contradicted, or undetermined.",
            "Inference: Choose the official supported answer.",
            f"Conclusion: The final answer is {gold}."
        ],
        "premises": premises or [],
        "confidence": 0.95,
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
    parser.add_argument("--logic_raw_file", type=str, default=None)
    parser.add_argument("--physics_raw_file", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, default=None)
    parser.add_argument("--keep_fallback", action="store_true")
    args = parser.parse_args()

    teacher = ExactDatasetTeacher(
        logic_file=args.logic_raw_file,
        physics_file=args.physics_raw_file,
    )

    df = pd.read_parquet(args.input_parquet)

    rows = []
    kept_by_dataset = 0
    kept_by_solver = 0
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

        # 1. Dataset teacher first
        dataset_obj = teacher.get(question)

        if dataset_obj is not None:
            p1 = p1_correctness_continuous(dataset_obj.get("answer", ""), gold, task_type)

            if p1 >= 0.8:
                teacher_obj = dataset_obj
                kept_by_dataset += 1

        # 2. Solver/verifier second
        if teacher_obj is None:
            if task_type == "physics":
                try:
                    sol = solve_physics(question)
                    if sol is not None:
                        p1 = p1_correctness_continuous(sol.get("answer", ""), gold, "physics")
                        if p1 >= 0.8:
                            teacher_obj = sol
                            kept_by_solver += 1
                except Exception:
                    teacher_obj = None

                if teacher_obj is None and args.keep_fallback:
                    teacher_obj = build_physics_gold_target(question, gold, extra)
                    kept_by_fallback += 1

            elif task_type == "logic":
                try:
                    logic_sol = solve_logic(question, premises)
                    if logic_sol is not None:
                        p1 = p1_correctness_continuous(logic_sol.get("answer", ""), gold, "logic")
                        if p1 >= 0.8:
                            teacher_obj = logic_sol
                            kept_by_solver += 1
                except Exception:
                    teacher_obj = None

                if teacher_obj is None and args.keep_fallback:
                    teacher_obj = build_logic_gold_target(question, gold, premises, extra)
                    kept_by_fallback += 1

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
    print("Kept by dataset teacher:", kept_by_dataset)
    print("Kept by solver/verifier:", kept_by_solver)
    print("Kept by gold fallback:", kept_by_fallback)
    print("Skipped:", skipped)
    print("Saved jsonl:", out)
    if args.output_parquet:
        print("Saved parquet:", args.output_parquet)


if __name__ == "__main__":
    main()
    