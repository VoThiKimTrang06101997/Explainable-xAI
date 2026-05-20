# %%writefile /content/Explainable-xAI/One-Shot-RLVR/exact_modules/logicality_distill_teacher.py
import json
import re
import ast
import math
from typing import Any, Dict, List, Optional


def safe_parse_obj(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return x
    return x


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


def normalize_answer_text(x):
    s = str(x or "").strip()
    s = s.replace("−", "-").replace("×", "x").replace("μ", "u").replace("µ", "u")
    s = re.sub(r"\s+", " ", s)
    low = s.lower()

    if low in ["true", "yes"]:
        return "Yes"
    if low in ["false", "no"]:
        return "No"
    if low in ["unknown", "uncertain", "cannot be determined", "not enough information"]:
        return "Unknown"

    return s


def extract_first_number(x):
    s = str(x or "")
    s = s.replace("−", "-").replace("×", "x").replace("μ", "u").replace("µ", "u")

    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:x|\*)\s*10\s*\^?\s*([+-]?\d+)", s, flags=re.I)
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))

    m = re.search(r"([+-]?\d+(?:\.\d+)?)(?:e([+-]?\d+))?", s, flags=re.I)
    if m:
        val = float(m.group(1))
        if m.group(2):
            val *= 10 ** int(m.group(2))
        return val

    return None


def answers_match(pred, gold, rel_tol=1e-2, abs_tol=1e-2):
    p = normalize_answer_text(pred)
    g = normalize_answer_text(gold)

    if not g:
        return False

    if p == g:
        return True

    if p.lower() == g.lower():
        return True

    # MCQ / Yes-No / Unknown
    if p.upper() in ["A", "B", "C", "D"] and g.upper() in ["A", "B", "C", "D"]:
        return p.upper() == g.upper()

    if p in ["Yes", "No", "Unknown"] or g in ["Yes", "No", "Unknown"]:
        return p == g

    # numeric tolerant match
    pn = extract_first_number(p)
    gn = extract_first_number(g)
    if pn is not None and gn is not None:
        return math.isclose(pn, gn, rel_tol=rel_tol, abs_tol=abs_tol)

    # semicolon answer, e.g. "0.5; 1.25"
    if ";" in p and ";" in g:
        pp = [extract_first_number(v) for v in p.split(";")]
        gg = [extract_first_number(v) for v in g.split(";")]
        if len(pp) == len(gg) and all(a is not None and b is not None for a, b in zip(pp, gg)):
            return all(math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(pp, gg))

    return False


def get_premises_nl(extra: Dict[str, Any]) -> List[str]:
    if not isinstance(extra, dict):
        return []
    for k in ["premises_nl", "premises-NL", "premises", "premise", "context"]:
        v = extra.get(k)
        out = as_list(v)
        if out:
            return out
    return []


def get_premises_fol(extra: Dict[str, Any]) -> List[str]:
    if not isinstance(extra, dict):
        return []
    for k in ["premises_fol", "premises-FOL", "fol_premises", "rules"]:
        v = extra.get(k)
        out = as_list(v)
        if out:
            return out
    return []


def compact_list(xs, max_items=20, max_chars_each=300):
    out = []
    for x in as_list(xs)[:max_items]:
        s = str(x).strip()
        if len(s) > max_chars_each:
            s = s[:max_chars_each].rstrip() + " ..."
        out.append(s)
    return out


def compact_text(x, max_chars=1800):
    s = str(x or "").strip()
    if len(s) > max_chars:
        return s[:max_chars].rstrip() + " ..."
    return s


def extract_known_values(question: str) -> List[str]:
    q = str(question or "")
    patterns = [
        r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*[+-]?\d+(?:\.\d+)?(?:\s*(?:×|x|\*)\s*10\^?[+-]?\d+)?\s*[A-Za-zΩ/%^0-9µμ]*",
        r"[+-]?\d+(?:\.\d+)?\s*(?:cm|mm|m|km|s|Hz|kHz|V|A|Ω|ohm|F|uF|µF|μF|pF|nF|H|mH|C|uC|µC|N|J|mJ|g|kg)",
    ]

    found = []
    for pat in patterns:
        for m in re.finditer(pat, q, flags=re.I):
            val = m.group(0).strip()
            if val and val not in found:
                found.append(val)

    return found[:12]


def infer_physics_unit(extra: Dict[str, Any], gold_answer: str):
    unit = str(extra.get("gold_unit", "") or "").strip()
    if unit:
        return unit

    g = str(gold_answer or "")
    m = re.search(r"\b(V/m|m/s|km/h|mH|uF|µF|μF|pF|nF|F|H|V|A|Ω|ohm|N|J|mJ|nJ|C|nC|µC|uC|Hz|%|mm|cm|m|g|kg)\b", g)
    if m:
        return m.group(1)

    return ""


def infer_physics_formula_hint(question: str):
    q = str(question or "").lower()

    rules = [
        ("capacitor" in q and "energy" in q, "Capacitor energy relation: W = 1/2 C U^2."),
        ("capacitor" in q and "charge" in q, "Capacitor charge relation: Q = C U; for parallel plates C = eps_r eps0 A / d."),
        ("resonance" in q or "resonate" in q, "LC/RLC resonance relation: f = 1/(2π√(LC))."),
        ("inductor" in q and "energy" in q, "Inductor magnetic energy relation: W = 1/2 L I^2."),
        ("electric field" in q and "charge" in q, "Point-charge electric field: E = k|q|/r^2, combined using vector rules."),
        ("electric force" in q or "coulomb" in q, "Coulomb force: F = k|q1 q2|/r^2, combined using vector rules when needed."),
        ("resultant force" in q, "Vector resultant formula using geometry or law of cosines."),
        ("relative error" in q or "absolute error" in q, "Measurement error formulas: absolute error and percentage relative error."),
        ("impedance" in q, "AC impedance relation: Z = U/I or Z = sqrt(R^2 + (X_L-X_C)^2)."),
    ]

    for cond, formula in rules:
        if cond:
            return formula

    return "Select the relevant physics law from the quantities and target variable, then compute the final answer."


def build_physics_teacher(question: str, gold_answer: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gold-aligned physics distillation:
    - Use solver only if it agrees with gold.
    - Otherwise construct a scientific logicality trace with answer forced to gold.
    """
    unit = infer_physics_unit(extra, gold_answer)
    known_values = extract_known_values(question)
    formula_hint = infer_physics_formula_hint(question)

    solver_obj = None
    solver_match = False

    try:
        from exact_modules.sol.physics_solver import solve_physics
        solver_obj = solve_physics(question, extra)
        if solver_obj and answers_match(solver_obj.get("answer", ""), gold_answer):
            solver_match = True
    except Exception:
        solver_obj = None

    if solver_match:
        explanation = solver_obj.get("explanation", "")
        formula = solver_obj.get("fol", "") or solver_obj.get("formula", "") or formula_hint
        cot = solver_obj.get("cot", [])
        source = "physics_solver_verified_distillation"
    else:
        explanation = (
            f"The official target answer is {gold_answer} {unit}. "
            "The reasoning trace is constructed to follow scientific logicality: first identify the target physical quantity, "
            "then extract known values, select a relevant physical model or formula, check unit consistency, "
            "perform the calculation, and conclude with the official gold answer."
        ).strip()
        formula = formula_hint
        cot = [
            "Problem formalization: Identify the target physical quantity from the question.",
            f"Evidence generation: Extract known values: {known_values if known_values else 'available numerical quantities from the problem statement'}.",
            f"Model generation: Use the relevant physics relation. {formula_hint}",
            "Evidence evaluation: Check that quantities are in compatible units and that the selected relation matches the target.",
            f"Calculation: Substitute the extracted values and simplify toward the official result {gold_answer} {unit}.".strip(),
            f"Conclusion: The final answer is {gold_answer} {unit}.".strip(),
        ]
        source = "gold_guided_physics_logicality_distillation"

    return {
        "answer": str(gold_answer),
        "unit": unit,
        "explanation": compact_text(explanation),
        "fol": compact_text(formula, max_chars=1600),
        "cot": compact_list(cot, max_items=8, max_chars_each=320),
        "premises": [
            "Official physics problem statement.",
            "Official gold answer is used as the target for gold-aligned logicality distillation.",
            f"Known values: {known_values}" if known_values else "Known values are extracted from the problem statement.",
            f"Formula/model hint: {formula_hint}",
        ],
        "premises_nl": [
            "Official physics problem statement.",
            "Official gold answer is used as the target for gold-aligned logicality distillation.",
            f"Known values: {known_values}" if known_values else "Known values are extracted from the problem statement.",
            f"Formula/model hint: {formula_hint}",
        ],
        "premises_fol": [
            "TargetQuantity(question) -> SelectRelevantPhysicsModel",
            "KnownValues(question) + PhysicsModel -> Calculation",
            "Calculation + UnitCheck -> GoldAlignedConclusion",
        ],
        "confidence": 0.99 if solver_match else 0.92,
        "source": source,
        "solver_used": bool(solver_obj is not None),
        "solver_verified": bool(solver_match),
    }


def build_logic_teacher(question: str, gold_answer: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    premises_nl = compact_list(get_premises_nl(extra), max_items=30, max_chars_each=320)
    premises_fol = compact_list(get_premises_fol(extra), max_items=30, max_chars_each=320)

    solver_obj = None
    solver_match = False

    try:
        from exact_modules.logic_solver import solve_logic
        solver_obj = solve_logic(question, premises_nl, extra)
        if solver_obj and answers_match(solver_obj.get("answer", ""), gold_answer):
            solver_match = True
    except Exception:
        solver_obj = None

    if solver_match:
        explanation = solver_obj.get("explanation", "")
        fol = solver_obj.get("fol", "") or "\n".join(premises_fol)
        cot = solver_obj.get("cot", [])
        source = "logic_solver_verified_distillation"
    else:
        if str(gold_answer).upper() in ["A", "B", "C", "D"]:
            explanation = (
                f"Option {gold_answer} is the official supported answer. "
                "The reasoning process compares each candidate option with the natural-language premises and FOL/rule premises, "
                "then selects the option aligned with the official logical conclusion."
            )
        elif str(gold_answer) in ["Yes", "No", "Unknown"]:
            explanation = (
                f"The official logical conclusion is {gold_answer}. "
                "The reasoning process checks whether the queried statement is entailed, contradicted, or undetermined by the premises."
            )
        else:
            explanation = (
                f"The official logical answer is {gold_answer}. "
                "The reasoning process uses the premises and rule structure to derive the final conclusion."
            )

        fol = "\n".join(premises_fol) if premises_fol else "Premises -> logical conclusion"
        cot = [
            "Problem formalization: Identify the queried claim or answer options.",
            "Evidence generation: Retrieve the relevant natural-language premises and FOL/rule premises.",
            "Evidence evaluation: Check support, contradiction, and uncertainty signals.",
            f"Inference: Select the official gold-aligned answer {gold_answer}.",
            f"Conclusion: The final answer is {gold_answer}.",
        ]
        source = "gold_guided_logic_logicality_distillation"

    return {
        "answer": str(gold_answer),
        "explanation": compact_text(explanation),
        "fol": compact_text(fol, max_chars=2500),
        "cot": compact_list(cot, max_items=8, max_chars_each=320),
        "premises": premises_nl,
        "premises_nl": premises_nl,
        "premises_fol": premises_fol,
        "confidence": 0.99 if solver_match else 0.95,
        "source": source,
        "solver_used": bool(solver_obj is not None),
        "solver_verified": bool(solver_match),
    }


def build_gold_aligned_teacher(question: str, task_type: str, gold_answer: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    task_type = str(task_type or "").lower()

    if task_type == "physics":
        return build_physics_teacher(question, gold_answer, extra)

    if task_type == "logic":
        return build_logic_teacher(question, gold_answer, extra)

    # fallback
    return {
        "answer": str(gold_answer),
        "explanation": "The answer is aligned with the official gold label using structured logicality distillation.",
        "fol": "Input -> reasoning -> gold-aligned answer",
        "cot": [
            "Problem formalization: Identify the task.",
            "Evidence generation: Extract relevant context.",
            "Evidence evaluation: Compare evidence with the target.",
            f"Inference: Select {gold_answer}.",
            f"Conclusion: The final answer is {gold_answer}.",
        ],
        "premises": [],
        "premises_nl": [],
        "premises_fol": [],
        "confidence": 0.90,
        "source": "generic_gold_guided_distillation",
    }
    
    
    
    