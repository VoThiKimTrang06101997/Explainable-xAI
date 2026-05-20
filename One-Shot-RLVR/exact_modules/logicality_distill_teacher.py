import json
import re
import ast
import math
from typing import Any, Dict, List


# ============================================================
# Basic parsing utilities
# ============================================================

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

    if isinstance(x, tuple):
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


def compact_list(xs, max_items=80, max_chars_each=1000):
    out = []
    for x in as_list(xs)[:max_items]:
        s = str(x).strip()
        if len(s) > max_chars_each:
            s = s[:max_chars_each].rstrip() + " ..."
        if s:
            out.append(s)
    return out


def compact_text(x, max_chars=4000):
    s = str(x or "").strip()
    if len(s) > max_chars:
        return s[:max_chars].rstrip() + " ..."
    return s


# ============================================================
# Answer matching
# ============================================================

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

    if p == g or p.lower() == g.lower():
        return True

    if p.upper() in ["A", "B", "C", "D"] and g.upper() in ["A", "B", "C", "D"]:
        return p.upper() == g.upper()

    if p in ["Yes", "No", "Unknown"] or g in ["Yes", "No", "Unknown"]:
        return p == g

    pn = extract_first_number(p)
    gn = extract_first_number(g)
    if pn is not None and gn is not None:
        return math.isclose(pn, gn, rel_tol=rel_tol, abs_tol=abs_tol)

    return False


# ============================================================
# Premise extraction
# ============================================================

SOURCE_NL_KEYS = [
    "_source_premises_nl",
    "source_premises_nl",
    "premises-NL",
    "premises_nl",
    "premises_NL",
    "premise-NL",
    "premise_nl",
    "premises",
    "premise",
    "context",
]

SOURCE_FOL_KEYS = [
    "_source_premises_fol",
    "source_premises_fol",
    "premises-FOL",
    "premises_fol",
    "premises_FOL",
    "premise-FOL",
    "premise_fol",
    "fol_premises",
    "rules",
]


def get_premises_nl(extra: Dict[str, Any]) -> List[str]:
    if not isinstance(extra, dict):
        return []

    for k in SOURCE_NL_KEYS:
        if k in extra:
            out = as_list(extra.get(k))
            if out:
                return out

    return []


def get_premises_fol(extra: Dict[str, Any]) -> List[str]:
    if not isinstance(extra, dict):
        return []

    for k in SOURCE_FOL_KEYS:
        if k in extra:
            out = as_list(extra.get(k))
            if out:
                return out

    return []


def is_bad_fol_placeholder(x):
    xs = as_list(x)
    if not xs:
        return True

    joined = " ".join(str(v) for v in xs).lower()
    bad = [
        "priority dataset-compatible",
        "conservative nl/fol",
        "verifier",
        "premises -> logical conclusion",
        "parsed simple rules",
        "dataset-compatible nl/fol",
    ]
    return any(b in joined for b in bad)


# ============================================================
# Fallback NL -> FOL converter
# Only used if original dataset FOL is not available.
# ============================================================

def _strip_prefix(s):
    s = str(s or "").strip()
    s = re.sub(r"^At the .*?,\s*", "", s, flags=re.I).strip()
    return s


def nl_premise_to_fol(premise):
    s = _strip_prefix(premise)
    if not s:
        return ""

    # property dataset patterns
    m = re.search(r"there exists at least one object x that has property ([A-Z])", s, flags=re.I)
    if m:
        a = m.group(1).upper()
        return f"Exists(x, {a}(x))"

    m = re.search(r"every object x has property ([A-Z])", s, flags=re.I)
    if m:
        a = m.group(1).upper()
        return f"ForAll(x, {a}(x))"

    m = re.search(r"if an object x does not have property ([A-Z]), then it does not have property ([A-Z])", s, flags=re.I)
    if m:
        a, b = m.group(1).upper(), m.group(2).upper()
        return f"ForAll(x, ¬{a}(x) → ¬{b}(x))"

    m = re.search(r"if an object x has property ([A-Z]), then it has property ([A-Z])", s, flags=re.I)
    if m:
        a, b = m.group(1).upper(), m.group(2).upper()
        return f"ForAll(x, {a}(x) → {b}(x))"

    m = re.search(
        r"if there exists at least one object x that has property ([A-Z]), then if an object x does not have property ([A-Z]) it does not have property ([A-Z])",
        s,
        flags=re.I,
    )
    if m:
        a, b, c = m.group(1).upper(), m.group(2).upper(), m.group(3).upper()
        return f"Exists(x, {a}(x)) → ForAll(x, ¬{b}(x) → ¬{c}(x))"

    m = re.search(
        r"if it is true that if an object x does not have property ([A-Z]) then it does not have property ([A-Z]), then there exists at least one object x that has property ([A-Z])",
        s,
        flags=re.I,
    )
    if m:
        a, b, c = m.group(1).upper(), m.group(2).upper(), m.group(3).upper()
        return f"ForAll(x, ¬{a}(x) → ¬{b}(x)) → Exists(x, {c}(x))"

    return ""


def convert_premises_nl_to_fol(premises_nl):
    out = []
    for p in as_list(premises_nl):
        f = nl_premise_to_fol(p)
        if f and f not in out:
            out.append(f)
    return out


# ============================================================
# Physics helpers
# ============================================================

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
    m = re.search(
        r"\b(V/m|m/s|km/h|mH|uF|µF|μF|pF|nF|F|H|V|A|Ω|ohm|N|J|mJ|nJ|C|nC|µC|uC|Hz|%|mm|cm|m|g|kg)\b",
        g,
    )
    if m:
        return m.group(1)

    return ""


def infer_physics_formula_hint(question: str):
    q = str(question or "").lower()

    if "capacitor" in q and "energy" in q:
        return "Capacitor energy relation: W = 1/2 C U^2."
    if "capacitor" in q and "charge" in q:
        return "Capacitor charge relation: Q = C U; for parallel plates C = eps_r eps0 A / d."
    if "resonance" in q or "resonate" in q:
        return "LC/RLC resonance relation: f = 1/(2π√(LC))."
    if "inductor" in q and "energy" in q:
        return "Inductor magnetic energy relation: W = 1/2 L I^2."
    if "electric field" in q and "charge" in q:
        return "Point-charge electric field: E = k|q|/r^2, combined using vector rules."
    if "electric force" in q or "coulomb" in q:
        return "Coulomb force: F = k|q1 q2|/r^2, combined using vector rules when needed."
    if "resultant force" in q:
        return "Vector resultant formula using geometry or law of cosines."
    if "relative error" in q or "absolute error" in q:
        return "Measurement error formulas: absolute error and percentage relative error."
    if "impedance" in q:
        return "AC impedance relation: Z = U/I or Z = sqrt(R^2 + (X_L-X_C)^2)."

    return "Select the relevant physics law from the quantities and target variable, then compute the final answer."


# ============================================================
# Teacher builders
# ============================================================

def answer_aligned_logic_explanation(answer, question, premises_nl, premises_fol):
    answer = str(answer).strip()

    if answer in ["A", "B", "C", "D"]:
        return (
            f"Option {answer} is the official gold-aligned answer. "
            "The reasoning compares each candidate option against the natural-language premises and FOL premises."
        )

    if answer == "Yes":
        return (
            "The queried statement is treated as true under the official annotation. "
            "The reasoning should connect the relevant premises through the FOL/rule structure and conclude Yes."
        )

    if answer == "No":
        return (
            "The queried statement is treated as false or not supported as true under the official annotation. "
            "The final answer must remain aligned with the official gold label No."
        )

    if answer == "Unknown":
        return (
            "The premise set does not provide enough decisive evidence to prove or disprove the queried statement. "
            "Therefore, the official gold-aligned answer is Unknown."
        )

    return (
        f"The official logical answer is {answer}. "
        "The reasoning process uses the premises and rule structure to derive the final gold-aligned conclusion."
    )


def build_physics_teacher(question: str, gold_answer: str, extra: Dict[str, Any]) -> Dict[str, Any]:
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
            "The reasoning trace is constructed using scientific logicality: identify the target physical quantity, "
            "extract known values, select the relevant physical model or formula, check unit consistency, perform calculation, "
            "and conclude with the official gold answer."
        ).strip()
        formula = formula_hint
        cot = [
            "Problem formalization: Identify the target physical quantity from the question.",
            f"Evidence generation: Extract known values: {known_values if known_values else 'available numerical quantities from the problem statement'}.",
            f"Model generation: Use the relevant physics relation. {formula_hint}",
            "Evidence evaluation: Check unit consistency and whether the formula matches the target.",
            f"Calculation: Substitute the extracted values and simplify toward the official result {gold_answer} {unit}.".strip(),
            f"Conclusion: The final answer is {gold_answer} {unit}.".strip(),
        ]
        source = "gold_guided_physics_logicality_distillation"

    premises_nl = [
        "Official physics problem statement.",
        "Official gold answer is used as the target for gold-aligned logicality distillation.",
        f"Known values: {known_values}" if known_values else "Known values are extracted from the problem statement.",
        f"Formula/model hint: {formula_hint}",
    ]

    premises_fol = [
        "TargetQuantity(question) -> SelectRelevantPhysicsModel",
        "KnownValues(question) + PhysicsModel -> Calculation",
        "Calculation + UnitCheck -> GoldAlignedConclusion",
    ]

    return {
        "answer": str(gold_answer),
        "unit": unit,
        "explanation": compact_text(explanation),
        "fol": compact_text(formula, max_chars=3000),
        "cot": compact_list(cot, max_items=8, max_chars_each=500),
        "premises": premises_nl,
        "premises_nl": premises_nl,
        "premises_fol": premises_fol,
        "premises-NL": premises_nl,
        "premises-FOL": premises_fol,
        "confidence": 0.99 if solver_match else 0.92,
        "source": source,
        "solver_used": bool(solver_obj is not None),
        "solver_verified": bool(solver_match),
    }


def build_logic_teacher(question: str, gold_answer: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    # Source dataset premises are authoritative.
    premises_nl = compact_list(get_premises_nl(extra), max_items=80, max_chars_each=1000)
    premises_fol = compact_list(get_premises_fol(extra), max_items=80, max_chars_each=1000)

    # If original FOL is missing, fallback to conversion from NL.
    if is_bad_fol_placeholder(premises_fol):
        converted = convert_premises_nl_to_fol(premises_nl)
        if converted:
            premises_fol = converted

    solver_obj = None
    solver_match = False

    try:
        from exact_modules.logic_solver import solve_logic
        solver_obj = solve_logic(question, premises_nl, extra)
        if solver_obj and answers_match(solver_obj.get("answer", ""), gold_answer):
            solver_match = True
    except Exception:
        solver_obj = None

    # Important:
    # We never use solver_obj["fol"] as the authoritative premises-FOL,
    # because solvers often return placeholders such as:
    # "Conservative NL/FOL rule verification."
    source = "logic_solver_verified_distillation" if solver_match else "gold_guided_logic_logicality_distillation"

    explanation = ""
    if solver_match:
        explanation = str(solver_obj.get("explanation", "") or "").strip()

    if not explanation:
        explanation = answer_aligned_logic_explanation(
            answer=gold_answer,
            question=question,
            premises_nl=premises_nl,
            premises_fol=premises_fol,
        )

    if solver_match:
        cot = solver_obj.get("cot", [])
    else:
        cot = []

    if not cot:
        cot = [
            "Problem formalization: Identify the queried claim or answer options.",
            "Evidence generation: Retrieve the relevant natural-language premises and FOL/rule premises.",
            "Evidence evaluation: Check whether the claim is supported, contradicted, or undetermined by the official premise structure.",
            f"Inference: Select the official gold-aligned answer {gold_answer}.",
            f"Conclusion: The final answer is {gold_answer}.",
        ]

    fol_text = "\n".join(premises_fol) if premises_fol else "Premises -> logical conclusion"

    return {
        "answer": str(gold_answer),
        "explanation": compact_text(explanation, max_chars=4000),
        "fol": compact_text(fol_text, max_chars=8000),
        "cot": compact_list(cot, max_items=8, max_chars_each=500),
        "premises": premises_nl,
        "premises_nl": premises_nl,
        "premises_fol": premises_fol,
        "premises-NL": premises_nl,
        "premises-FOL": premises_fol,
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
        "premises-NL": [],
        "premises-FOL": [],
        "confidence": 0.90,
        "source": "generic_gold_guided_distillation",
    }
    

