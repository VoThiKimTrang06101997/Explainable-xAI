import re
from typing import Any, Dict

from z3 import And, Bool, Implies, Not, Solver, sat, unsat


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(s))


def _bool(pred, const="Entity"):
    return Bool(f"{_safe(pred)}_{_safe(const)}")


def _extract_atoms(expr):
    expr = str(expr).replace("¬", " not ")
    atoms = []

    for m in re.finditer(r"(not\s+)?([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)", expr):
        neg = bool(m.group(1))
        pred = m.group(2)
        const = m.group(3).split(",")[0].strip()

        if const in ["x", "y", "z", "s", "a", "c", ""]:
            const = "Entity"

        atom = _bool(pred, const)
        atoms.append(Not(atom) if neg else atom)

    return atoms


def _parse_implication(expr):
    s = str(expr)
    if "→" not in s and "->" not in s:
        return None

    parts = re.split(r"→|->", s, maxsplit=1)
    left_atoms = _extract_atoms(parts[0])
    right_atoms = _extract_atoms(parts[1])

    if not left_atoms or not right_atoms:
        return None

    left = And(*left_atoms) if len(left_atoms) > 1 else left_atoms[0]
    right = And(*right_atoms) if len(right_atoms) > 1 else right_atoms[0]

    return Implies(left, right)


def _parse_fact(expr):
    s = str(expr)
    if "→" in s or "->" in s or "ForAll" in s or "∀" in s:
        return None

    atoms = _extract_atoms(s)
    if not atoms:
        return None

    return And(*atoms) if len(atoms) > 1 else atoms[0]


def light_z3_verify(extra_info: Dict[str, Any]) -> Dict[str, Any]:
    premises = extra_info.get("premises_fol", [])

    if not isinstance(premises, list) or len(premises) == 0:
        return {
            "supported": False,
            "status": "no_fol",
            "score": 0.0,
            "used": [],
        }

    solver = Solver()
    used = []

    for p in premises[:10]:
        rule = _parse_implication(p)
        fact = _parse_fact(p)

        if rule is not None:
            solver.add(rule)
            used.append(str(p))
        elif fact is not None:
            solver.add(fact)
            used.append(str(p))

    if not used:
        return {
            "supported": False,
            "status": "unsupported_fol",
            "score": 0.0,
            "used": [],
        }

    status = solver.check()

    if status == sat:
        return {
            "supported": True,
            "status": "consistent",
            "score": 0.7,
            "used": used[:5],
        }

    if status == unsat:
        return {
            "supported": True,
            "status": "inconsistent",
            "score": 0.2,
            "used": used[:5],
        }

    return {
        "supported": True,
        "status": "unknown",
        "score": 0.4,
        "used": used[:5],
    }
    