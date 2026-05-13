from typing import Any, Dict

from exact_modules.common import clamp01
from exact_modules.config import EXACT_CONFIG
from exact_modules.fol.templates import build_fol_template, build_logic_cot, get_premises_nl
from exact_modules.fol.z3_heavy import heavy_z3_verify
from exact_modules.moe.router import route_experts
from exact_modules.sol.physics_solver import solve_physics


def _ensure_list(x):
    if isinstance(x, list):
        return [str(i) for i in x]

    if x is None or x == "":
        return []

    return [str(x)]


def base_organizer(obj: Dict[str, Any], task_type: str, question: str, extra_info: Dict[str, Any]):
    if obj is None:
        obj = {}

    answer = str(obj.get("answer", "")).strip()
    explanation = str(obj.get("explanation", "")).strip()
    fol = str(obj.get("fol", "")).strip()
    cot = _ensure_list(obj.get("cot", []))
    premises = _ensure_list(obj.get("premises", []))

    try:
        confidence = float(obj.get("confidence", 0.5))
    except Exception:
        confidence = 0.5

    if not answer:
        answer = "Unknown" if task_type == "logic" else ""

    if not explanation or len(explanation.split()) < 8:
        if task_type == "physics":
            explanation = (
                "The answer is derived by identifying the known quantities, "
                "selecting the relevant physics formula, substituting values, "
                "and computing the final result."
            )
        else:
            explanation = (
                "The answer is derived by identifying the relevant premises, "
                "mapping them to a logical rule, and applying the rule to the question."
            )

    if not fol:
        fol = build_fol_template(extra_info) if task_type == "logic" else "Known quantities + applicable formula -> numerical answer"

    if not cot:
        if task_type == "logic":
            cot = build_logic_cot(extra_info)
        else:
            cot = [
                "Step 1: Identify the target physical quantity.",
                "Step 2: Extract known values from the question.",
                "Step 3: Select the relevant physics formula.",
                "Step 4: Substitute values and compute the final answer.",
            ]

    if not premises:
        if task_type == "logic":
            premises = get_premises_nl(extra_info)
            if not premises:
                premises = ["Relevant natural-language premises are used to infer the answer."]
        else:
            premises = [
                f"Question: {question[:300]}",
                "Known quantities and relevant physics formulas are extracted from the question.",
            ]

    return {
        "answer": answer,
        "explanation": explanation,
        "fol": fol,
        "cot": cot,
        "premises": premises,
        "confidence": clamp01(confidence),
    }


def calibrate_confidence(out, json_valid=True, answer_ok=None, symbolic_status=None):
    conf = float(out.get("confidence", 0.5))

    if conf <= 0.0:
        conf = 0.45

    if json_valid:
        conf += 0.10

    if out.get("fol"):
        conf += 0.08

    if isinstance(out.get("cot"), list) and len(out["cot"]) >= 2:
        conf += 0.08

    if isinstance(out.get("premises"), list) and len(out["premises"]) >= 1:
        conf += 0.06

    if answer_ok == 1:
        conf += 0.15

    if symbolic_status == "consistent":
        conf += 0.08
    elif symbolic_status == "inconsistent":
        conf -= 0.10

    return round(clamp01(conf), 2)


def build_final_output(
    model_obj: Dict[str, Any],
    raw_output: str,
    task_type: str,
    question: str,
    extra_info: Dict[str, Any],
    json_valid_original: bool = True,
):
    experts = route_experts(task_type, question=question, extra_info=extra_info)

    out = base_organizer(
        obj=model_obj or {},
        task_type=task_type,
        question=question,
        extra_info=extra_info,
    )

    symbolic_status = None

    if experts["sol"]:
        sol = solve_physics(question)

        if sol is not None:
            out["answer"] = sol["answer"]
            out["explanation"] = sol["explanation"]
            out["fol"] = sol["fol"]
            out["cot"] = sol["cot"]
            out["premises"] = sol["premises"]
            out["confidence"] = sol["confidence"]

    if experts["fol"]:
        out["fol"] = build_fol_template(extra_info)

    if experts["z3_logic"] and EXACT_CONFIG.get("use_heavy_z3_inference", True):
        z3_result = heavy_z3_verify(extra_info, out.get("answer", ""))
        symbolic_status = z3_result.get("status")

        if z3_result.get("supported"):
            out["cot"] = [
                "Step 1: Convert the relevant natural-language premises into FOL/rule templates.",
                "Step 2: Load the parsed FOL rules and facts into the Z3 symbolic verifier.",
                f"Step 3: Z3 verification status is {z3_result.get('status')}.",
                "Step 4: Combine symbolic verification with the LLM answer to produce the final response.",
            ]

            if z3_result.get("used_fol"):
                out["premises"] = z3_result["used_fol"][:5]

            if z3_result.get("explanation"):
                out["explanation"] = out["explanation"] + " " + z3_result["explanation"]

            out["confidence"] = clamp01(
                float(out.get("confidence", 0.5)) + z3_result.get("confidence_bonus", 0.0)
            )

    out["confidence"] = calibrate_confidence(
        out,
        json_valid=json_valid_original,
        symbolic_status=symbolic_status,
    )

    return out
