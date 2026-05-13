import re
from typing import Any, Dict, List, Tuple

from z3 import And, Bool, Implies, Not, Solver, sat, unsat


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(s))


def var(pred, const="Entity"):
    return Bool(f"{safe_name(pred)}_{safe_name(const)}")


def extract_atoms(expr: str) -> List[Tuple[str, str, bool]]:
    expr = str(expr)
    expr = expr.replace("¬", " not ")

    atoms = []
    for m in re.finditer(r"(not\s+)?([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)", expr):
        neg = bool(m.group(1))
        pred = m.group(2)
        const = m.group(3).split(",")[0].strip()

        if const in ["x", "y", "z", "a", "c", "s", ""]:
            const = "Entity"

        atoms.append((pred, const, neg))

    return atoms


def atom_to_z3(atom):
    pred, const, neg = atom
    b = var(pred, const)
    return Not(b) if neg else b


def parse_rule(expr: str):
    s = str(expr)

    if "→" not in s and "->" not in s:
        return None

    parts = re.split(r"→|->", s, maxsplit=1)
    left, right = parts[0], parts[1]

    left_atoms = extract_atoms(left)
    right_atoms = extract_atoms(right)

    if not left_atoms or not right_atoms:
        return None

    antecedent = And(*[atom_to_z3(a) for a in left_atoms]) if len(left_atoms) > 1 else atom_to_z3(left_atoms[0])
    consequent = And(*[atom_to_z3(a) for a in right_atoms]) if len(right_atoms) > 1 else atom_to_z3(right_atoms[0])

    return Implies(antecedent, consequent)


def parse_fact(expr: str):
    s = str(expr)

    if "→" in s or "->" in s or "∀" in s or "ForAll" in s:
        return None

    atoms = extract_atoms(s)
    if not atoms:
        return None

    z3_atoms = [atom_to_z3(a) for a in atoms]
    return And(*z3_atoms) if len(z3_atoms) > 1 else z3_atoms[0]


def heavy_z3_verify(extra_info: Dict[str, Any], answer: str = "") -> Dict[str, Any]:
    """
    Inference/evaluation-time heavy verifier.
    More expensive and more descriptive than light_z3_verify.
    """
    premises = extra_info.get("premises_fol", [])

    if not isinstance(premises, list) or not premises:
        return {
            "supported": False,
            "status": "no_fol",
            "confidence_bonus": 0.0,
            "used_fol": [],
            "explanation": "No FOL premises are available for symbolic verification.",
        }

    solver = Solver()
    used = []
    parsed_rules = 0
    parsed_facts = 0

    for p in premises:
        rule = parse_rule(p)
        fact = parse_fact(p)

        if rule is not None:
            solver.add(rule)
            used.append(str(p))
            parsed_rules += 1
        elif fact is not None:
            solver.add(fact)
            used.append(str(p))
            parsed_facts += 1

    if not used:
        return {
            "supported": False,
            "status": "unsupported_fol",
            "confidence_bonus": 0.0,
            "used_fol": [],
            "explanation": "FOL strings were provided but not supported by the parser.",
        }

    status = solver.check()

    if status == sat:
        return {
            "supported": True,
            "status": "consistent",
            "confidence_bonus": 0.08,
            "used_fol": used[:8],
            "explanation": (
                f"Z3 parsed {parsed_rules} rules and {parsed_facts} facts. "
                f"The premise set is logically consistent."
            ),
        }

    if status == unsat:
        return {
            "supported": True,
            "status": "inconsistent",
            "confidence_bonus": -0.08,
            "used_fol": used[:8],
            "explanation": (
                f"Z3 parsed {parsed_rules} rules and {parsed_facts} facts, "
                f"but the premise set is inconsistent."
            ),
        }

    return {
        "supported": True,
        "status": "unknown",
        "confidence_bonus": 0.0,
        "used_fol": used[:8],
        "explanation": "Z3 could not determine satisfiability.",
    }
    