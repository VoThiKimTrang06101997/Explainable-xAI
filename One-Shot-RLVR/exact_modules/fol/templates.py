from typing import Any, Dict, List


def get_premises_nl(extra_info: Dict[str, Any]) -> List[str]:
    p = extra_info.get("premises_nl", [])
    if isinstance(p, list):
        return [str(x) for x in p]
    return []


def get_premises_fol(extra_info: Dict[str, Any]) -> List[str]:
    p = extra_info.get("premises_fol", [])
    if isinstance(p, list):
        return [str(x) for x in p]
    return []


def build_fol_template(extra_info: Dict[str, Any]) -> str:
    premises_fol = get_premises_fol(extra_info)
    if premises_fol:
        return " ∧ ".join(premises_fol[:3]) + " → Answer"

    return "∀p∀q ((RelevantPremise(p) ∧ Supports(p, q)) → Answer(q))"


def build_logic_cot(extra_info: Dict[str, Any], z3_status: str = ""):
    cot = [
        "Step 1: Read the natural-language premises and identify the relevant rules.",
        "Step 2: Convert the relevant rules into FOL or rule templates.",
        "Step 3: Apply the rules to the question or candidate answer.",
    ]

    if z3_status:
        cot.append(f"Step 4: Use the symbolic verifier; Z3 status is {z3_status}.")
    else:
        cot.append("Step 4: Derive the supported answer from the premises.")

    return cot

