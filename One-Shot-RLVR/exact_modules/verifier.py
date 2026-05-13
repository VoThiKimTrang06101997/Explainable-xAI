from typing import Any, Dict

from exact_modules.common import answer_score, clamp01, is_nan_like, text_quality_score
from exact_modules.fol.z3_light import light_z3_verify
from exact_modules.sol.physics_solver import solve_physics


def format_score(obj: Dict[str, Any], raw_text: str = ""):
    if obj is None:
        return min(0.35, text_quality_score(raw_text))

    keys = ["answer", "explanation", "fol", "cot", "premises", "confidence"]
    present = sum(int(k in obj) for k in keys)

    non_empty = 0
    for k in keys:
        v = obj.get(k, None)
        if isinstance(v, list):
            non_empty += int(len(v) > 0)
        else:
            non_empty += int(v not in [None, "", []])

    return clamp01(0.5 * present / len(keys) + 0.5 * non_empty / len(keys))


def explanation_score(obj: Dict[str, Any], raw_text: str = ""):
    if obj is None:
        return text_quality_score(raw_text)

    exp = str(obj.get("explanation", ""))
    cot = obj.get("cot", [])
    premises = obj.get("premises", [])

    score = 0.0
    if len(exp.split()) >= 8:
        score += 0.25
    if len(exp.split()) >= 16:
        score += 0.20
    if any(k in exp.lower() for k in [
        "because", "therefore", "thus", "hence", "formula",
        "premise", "substitute", "derive", "calculate", "apply",
    ]):
        score += 0.25
    if isinstance(cot, list) and len(cot) >= 2:
        score += 0.20
    if isinstance(premises, list) and len(premises) >= 1:
        score += 0.10

    return clamp01(score)


def reasoning_score(obj: Dict[str, Any]):
    if obj is None:
        return 0.0

    score = 0.0
    if isinstance(obj.get("cot"), list) and len(obj["cot"]) >= 2:
        score += 0.35
    if isinstance(obj.get("premises"), list) and len(obj["premises"]) >= 1:
        score += 0.25
    if isinstance(obj.get("fol"), str) and len(obj["fol"].strip()) > 0:
        score += 0.25
    if "confidence" in obj:
        score += 0.15

    return clamp01(score)


def physics_sol_score(pred_answer: str, question: str):
    sol = solve_physics(question)
    if sol is None:
        return 0.0
    return answer_score(pred_answer, sol["answer"], "physics")


def symbolic_light_score(obj: Dict[str, Any], extra_info: Dict[str, Any], task_type: str):
    if task_type != "logic":
        return 0.0

    z3_result = light_z3_verify(extra_info)
    return float(z3_result.get("score", 0.0))


def dense_nonzero_floor(obj, raw_text):
    if obj is not None:
        return 0.05

    tq = text_quality_score(raw_text)
    if tq > 0.2:
        return 0.03

    return 0.0


def exact_reward_components(obj, raw_text, pred_answer, gold, task_type, question, extra_info):
    gold_valid = bool(extra_info.get("gold_valid", not is_nan_like(gold)))

    if gold_valid:
        r_answer = answer_score(pred_answer, gold, task_type)
    else:
        r_answer = 0.0

    r_format = format_score(obj, raw_text)
    r_expl = explanation_score(obj, raw_text)
    r_reason = reasoning_score(obj)
    r_symbolic = symbolic_light_score(obj or {}, extra_info, task_type)
    r_sol = physics_sol_score(pred_answer, question) if task_type == "physics" else 0.0
    r_floor = dense_nonzero_floor(obj, raw_text)

    return {
        "answer": r_answer,
        "format": r_format,
        "explanation": r_expl,
        "reasoning": r_reason,
        "symbolic": r_symbolic,
        "sol": r_sol,
        "floor": r_floor,
        "gold_valid": gold_valid,
    }
    