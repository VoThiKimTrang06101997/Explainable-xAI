def route_experts(task_type, question="", extra_info=None):
    task_type = str(task_type).lower()
    q = str(question or "").lower()

    if task_type == "physics":
        experts = {
            "llm": True,
            "sol": True,
            "unit_converter": True,
            "formula_bank": True,
            "z3_numeric_optional": False,
            "fol": False,
            "z3_logic": False,
            "format": True,
        }

        # Optional numeric-Z3-like algebra route for simple circuit equations.
        # We keep it false by default because SOL is better for physics formulas.
        if any(k in q for k in [
            "ohm",
            "v =",
            "i =",
            "r =",
            "voltage",
            "current",
            "resistance",
        ]):
            experts["z3_numeric_optional"] = True

        return experts

    if task_type == "logic":
        return {
            "llm": True,
            "sol": False,
            "unit_converter": False,
            "formula_bank": False,
            "z3_numeric_optional": False,
            "fol": True,
            "z3_logic": True,
            "format": True,
        }

    return {
        "llm": True,
        "sol": False,
        "unit_converter": False,
        "formula_bank": False,
        "z3_numeric_optional": False,
        "fol": False,
        "z3_logic": False,
        "format": True,
    }
    