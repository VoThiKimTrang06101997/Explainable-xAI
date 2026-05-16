# %%writefile /content/Explainable-xAI/One-Shot-RLVR/exact_modules/dataset_teacher.py
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd


def normalize_question(x: str) -> str:
    x = str(x or "")
    x = x.replace("−", "-").replace("μ", "u").replace("µ", "u")
    x = x.replace("⁰", "0").replace("¹", "1").replace("²", "2").replace("³", "3")
    x = x.replace("⁴", "4").replace("⁵", "5").replace("⁶", "6").replace("⁷", "7")
    x = x.replace("⁸", "8").replace("⁹", "9").replace("⁻", "-").replace("⁺", "+")
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x


def _safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def _make_teacher(
    task_type: str,
    question: str,
    answer: str,
    explanation: str = "",
    cot: str = "",
    unit: str = "",
    source: str = "dataset_teacher",
) -> Dict[str, Any]:
    task_type = str(task_type).lower()
    explanation = _safe_str(explanation).strip()
    cot_raw = _safe_str(cot).strip()

    if not explanation and cot_raw:
        explanation = cot_raw

    if not explanation:
        explanation = (
            "The answer is derived from the official dataset annotation by formalizing "
            "the problem, extracting relevant evidence, evaluating the rule or formula, "
            "and drawing the final conclusion."
        )

    if task_type == "physics":
        cot_list = [
            "Problem formalization: Identify the target physical quantity.",
            "Evidence generation: Extract the known values and units from the problem.",
            "Evidence evaluation: Select the relevant formula and check unit consistency.",
            "Calculation: Apply the formula or official derivation.",
            f"Conclusion: The final answer is {answer} {unit}.".strip()
        ]

        fol = "Official physics annotation: known quantities + formula -> answer"

        return {
            "answer": _safe_str(answer),
            "unit": _safe_str(unit),
            "explanation": explanation,
            "fol": fol,
            "cot": cot_list,
            "premises": [
                "Official physics problem statement.",
                "Official dataset answer and derivation."
            ],
            "confidence": 0.98,
            "source": source,
        }

    cot_list = [
        "Problem formalization: Identify the logical claim or answer options.",
        "Evidence generation: Extract the relevant premises.",
        "Evidence evaluation: Compare the claim or options against the premises.",
        "Inference: Select the answer supported by the official annotation.",
        f"Conclusion: The final answer is {answer}."
    ]

    return {
        "answer": _safe_str(answer),
        "explanation": explanation,
        "fol": "Official logic annotation: premises -> answer",
        "cot": cot_list,
        "premises": [],
        "confidence": 0.98,
        "source": source,
    }


class ExactDatasetTeacher:
    def __init__(self, logic_file: str = None, physics_file: str = None):
        self.lookup = {}

        if logic_file and Path(logic_file).exists():
            self.load_logic(logic_file)

        if physics_file and Path(physics_file).exists():
            self.load_physics(physics_file)

    def add(self, question: str, obj: Dict[str, Any]):
        key = normalize_question(question)
        if key:
            self.lookup[key] = obj

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        key = normalize_question(question)

        if key in self.lookup:
            return dict(self.lookup[key])

        # soft match fallback for tiny formatting differences
        q_tokens = set(key.split())
        best_key = None
        best_score = 0.0

        for k in self.lookup.keys():
            kt = set(k.split())
            if not q_tokens or not kt:
                continue
            score = len(q_tokens & kt) / max(1, len(q_tokens | kt))
            if score > best_score:
                best_score = score
                best_key = k

        if best_key is not None and best_score >= 0.96:
            return dict(self.lookup[best_key])

        return None

    def load_logic(self, logic_file: str):
        with open(logic_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for record in data:
            questions = record.get("questions", [])
            answers = record.get("answers", [])
            explanations = record.get("explanation", [])
            premises_nl = record.get("premises-NL", [])

            for i, q in enumerate(questions):
                ans = answers[i] if i < len(answers) else ""
                exp = explanations[i] if i < len(explanations) else ""

                obj = _make_teacher(
                    task_type="logic",
                    question=q,
                    answer=ans,
                    explanation=exp,
                    source="dataset_logic_teacher",
                )
                obj["premises"] = premises_nl
                self.add(q, obj)

    def load_physics(self, physics_file: str):
        df = pd.read_csv(physics_file)

        # flexible column detection
        cols = {c.lower().strip(): c for c in df.columns}

        question_candidates = [
            "question", "questions", "problem", "text", "prompt",
            "Question", "Problem"
        ]
        answer_candidates = [
            "answer", "answers", "gold_answer", "final_answer",
            "Answer", "Final Answer"
        ]
        explanation_candidates = [
            "explanation", "gold_explanation", "solution", "rationale",
            "Explanation", "Solution", "cot", "CoT"
        ]
        unit_candidates = [
            "unit", "gold_unit", "Unit"
        ]

        def find_col(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
                if c.lower().strip() in cols:
                    return cols[c.lower().strip()]
            return None

        q_col = find_col(question_candidates)
        a_col = find_col(answer_candidates)
        e_col = find_col(explanation_candidates)
        u_col = find_col(unit_candidates)

        if q_col is None or a_col is None:
            print("[WARN] Could not detect physics question/answer columns.")
            print("Columns:", list(df.columns))
            return

        for _, row in df.iterrows():
            q = _safe_str(row.get(q_col, "")).strip()
            ans = _safe_str(row.get(a_col, "")).strip()
            exp = _safe_str(row.get(e_col, "")).strip() if e_col else ""
            unit = _safe_str(row.get(u_col, "")).strip() if u_col else ""

            if not q or not ans:
                continue

            obj = _make_teacher(
                task_type="physics",
                question=q,
                answer=ans,
                explanation=exp,
                unit=unit,
                source="dataset_physics_teacher",
            )

            self.add(q, obj)
            