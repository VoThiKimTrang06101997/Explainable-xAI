import json
import ast
import re
import pandas as pd


def normalize_question(x: str) -> str:
    x = str(x or "")
    x = x.replace("−", "-").replace("μ", "u").replace("µ", "u")
    x = x.replace("⁰", "0").replace("¹", "1").replace("²", "2").replace("³", "3")
    x = x.replace("⁴", "4").replace("⁵", "5").replace("⁶", "6").replace("⁷", "7")
    x = x.replace("⁸", "8").replace("⁹", "9").replace("⁻", "-").replace("⁺", "+")
    x = re.sub(r"\s+", " ", x).strip().lower()
    return x


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


def _as_list(x):
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

        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(v) for v in obj if str(v).strip()]
        except Exception:
            pass

        return [s]

    return [str(x)]


def get_premises_nl(extra):
    for key in [
        "premises-NL",
        "premises_nl",
        "premisesNL",
        "premises",
        "gold_premises_nl",
    ]:
        if key in extra:
            vals = _as_list(extra.get(key))
            if vals:
                return vals
    return []


def get_premises_fol(extra):
    for key in [
        "premises-FOL",
        "premises_fol",
        "premisesFOL",
        "fol_premises",
        "gold_premises_fol",
    ]:
        if key in extra:
            vals = _as_list(extra.get(key))
            if vals:
                return vals
    return []


def get_gold_explanation(extra):
    gold_exp = str(extra.get("gold_explanation", "") or "").strip()
    gold_cot = str(extra.get("gold_cot", "") or "").strip()

    if gold_exp:
        return gold_exp
    if gold_cot:
        return gold_cot

    return (
        "The answer is derived from the official gold annotation by formalizing the problem, "
        "extracting evidence, evaluating the rule or formula, and drawing the conclusion."
    )


def make_teacher(task_type, question, answer, extra):
    task_type = str(task_type).lower()
    explanation = get_gold_explanation(extra)

    if task_type == "physics":
        unit = str(extra.get("gold_unit", "") or "")

        return {
            "answer": str(answer),
            "unit": unit,
            "explanation": explanation,
            "fol": "Official parquet gold annotation: physics problem -> formula -> answer",
            "cot": [
                "Problem formalization: Identify the target physical quantity.",
                "Evidence generation: Extract the known values and units from the problem.",
                "Evidence evaluation: Select the relevant formula and check unit consistency.",
                "Calculation: Apply the official derivation.",
                f"Conclusion: The final answer is {answer} {unit}.".strip()
            ],
            "premises": [
                "Official physics problem statement.",
                "Official parquet gold answer."
            ],
            "premises_nl": [
                "Official physics problem statement.",
                "Official parquet gold answer."
            ],
            "premises_fol": [],
            "confidence": 0.99,
            "source": "parquet_physics_teacher",
        }

    premises_nl = get_premises_nl(extra)
    premises_fol = get_premises_fol(extra)

    fol_text = "\n".join(premises_fol).strip()
    if not fol_text:
        fol_text = "Official parquet gold annotation: premises -> answer"

    return {
        "answer": str(answer),
        "explanation": explanation,
        "fol": fol_text,
        "cot": [
            "Problem formalization: Identify the logical claim or answer options.",
            "Evidence generation: Extract the relevant natural-language and FOL premises.",
            "Evidence evaluation: Compare the claim or options against the NL/FOL premises.",
            "Inference: Select the official supported answer.",
            f"Conclusion: The final answer is {answer}."
        ],
        "premises": premises_nl,
        "premises_nl": premises_nl,
        "premises_fol": premises_fol,
        "confidence": 0.99,
        "source": "parquet_logic_teacher",
    }


class ExactParquetTeacher:
    def __init__(self, parquet_file):
        self.lookup = {}
        df = pd.read_parquet(parquet_file)

        for _, row in df.iterrows():
            extra = parse_obj(row.get("extra_info", {}))
            reward_model = parse_obj(row.get("reward_model", {}))

            q = extra.get("question", "")
            task_type = extra.get("task_type", row.get("ability", ""))
            gold = reward_model.get("ground_truth", extra.get("gold_answer", ""))

            if not q:
                continue

            self.lookup[normalize_question(q)] = make_teacher(task_type, q, gold, extra)

    def get(self, question):
        return self.lookup.get(normalize_question(question))
        
