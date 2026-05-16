# %%writefile /content/Explainable-xAI/One-Shot-RLVR/exact_modules/sol/physics_solver.py
import math
import re
from typing import Dict, Any, Optional, Tuple, List


# ============================================================
# 0. Constants
# ============================================================

K_COULOMB = 9.0e9
EPS0 = 8.854e-12
PI = math.pi


# ============================================================
# 1. Text / number utilities
# ============================================================

def normalize_text(x: str) -> str:
    x = str(x or "")
    x = x.replace("−", "-")
    x = x.replace("μ", "u").replace("µ", "u")
    x = x.replace("×", "x")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def normalize_superscript(s: str) -> str:
    table = str.maketrans({
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁻": "-", "⁺": "+",
    })
    return str(s).translate(table)


def to_float(num_str: str) -> Optional[float]:
    if num_str is None:
        return None

    s = normalize_superscript(str(num_str))
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("×", "x").replace("\\times", "x")
    s = s.replace("{", "").replace("}", "")

    # 5.10^-16 means 5 * 10^-16 in some dataset lines
    m = re.match(r"^(-?\d+(?:\.\d+)?)\.10\^?(-?\d+)$", s)
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))

    # 5 x 10^-16
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*(?:x|\*)\s*10\s*(?:\^|\*\*)?\s*(-?\d+)$", s, flags=re.I)
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))

    try:
        return float(s)
    except Exception:
        return None


def fmt(x, decimals: int = 4) -> str:
    try:
        x = float(x)

        if abs(x) >= 1e5 or (abs(x) > 0 and abs(x) < 1e-3):
            return f"{x:.6g}"

        return f"{x:.{decimals}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def make_result(answer, unit, formula, explanation, cot, premises, confidence=0.90, source="physics_solver") -> Dict[str, Any]:
    return {
        "answer": str(answer),
        "unit": unit or "",
        "explanation": explanation,
        "fol": formula,
        "cot": cot,
        "premises": premises,
        "confidence": float(confidence),
        "source": source,
    }


def find_value(question: str, symbol: str, units: Optional[List[str]] = None) -> Optional[Tuple[float, str]]:
    """
    Finds patterns like:
    U = 100 V
    C=30 μF
    q1 = q2 = 5.10^-16 C
    """
    q = normalize_text(question)
    units = units or []

    unit_pattern = "|".join([re.escape(u) for u in units]) if units else r"[a-zA-ZΩ%]+"

    patterns = [
        rf"\b{re.escape(symbol)}\s*=\s*([+\-]?\d+(?:\.\d+)?(?:\s*(?:x|\*)\s*10\s*\^?\s*[+\-]?\d+|\.10\^?[+\-]?\d+|e[+\-]?\d+)?)\s*({unit_pattern})?",
        rf"\b{re.escape(symbol)}\s*is\s*([+\-]?\d+(?:\.\d+)?(?:\s*(?:x|\*)\s*10\s*\^?\s*[+\-]?\d+|\.10\^?[+\-]?\d+|e[+\-]?\d+)?)\s*({unit_pattern})?",
    ]

    for p in patterns:
        m = re.search(p, q, flags=re.I)
        if m:
            val = to_float(m.group(1))
            unit = m.group(2) or ""
            if val is not None:
                return val, unit

    return None


def find_first_number_with_unit(question: str, units: List[str]) -> Optional[Tuple[float, str]]:
    q = normalize_text(question)
    unit_pattern = "|".join([re.escape(u) for u in units])

    p = rf"([+\-]?\d+(?:\.\d+)?(?:\s*(?:x|\*)\s*10\s*\^?\s*[+\-]?\d+|\.10\^?[+\-]?\d+|e[+\-]?\d+)?)\s*({unit_pattern})"
    m = re.search(p, q, flags=re.I)

    if m:
        val = to_float(m.group(1))
        unit = m.group(2)
        if val is not None:
            return val, unit

    return None


def unit_scale(unit: str) -> float:
    u = str(unit or "").lower().strip()
    u = u.replace("μ", "u").replace("µ", "u")

    scales = {
        "pf": 1e-12,
        "nf": 1e-9,
        "uf": 1e-6,
        "microf": 1e-6,
        "mf": 1e-3,
        "f": 1.0,
        "mh": 1e-3,
        "uh": 1e-6,
        "h": 1.0,
        "cm": 1e-2,
        "mm": 1e-3,
        "m": 1.0,
        "km": 1e3,
        "ma": 1e-3,
        "a": 1.0,
        "mv": 1e-3,
        "v": 1.0,
        "kv": 1e3,
        "uc": 1e-6,
        "microc": 1e-6,
        "nc": 1e-9,
        "pc": 1e-12,
        "c": 1.0,
        "kj": 1e3,
        "j": 1.0,
        "hz": 1.0,
        "khz": 1e3,
        "mhz": 1e6,
        "n": 1.0,
        "ohm": 1.0,
        "ω": 1.0,
        "Ω": 1.0,
        "%": 1.0,
    }

    return scales.get(u, 1.0)


def convert_value(value: float, unit: str) -> float:
    return value * unit_scale(unit)


def extract_all_numbers(question: str) -> List[float]:
    q = normalize_text(normalize_superscript(question))
    nums = []

    sci_pattern = r"([+\-]?\d+(?:\.\d+)?)\s*(?:x|\*)\s*10\s*\^?\s*([+\-]?\d+)"
    for m in re.finditer(sci_pattern, q, flags=re.I):
        nums.append(float(m.group(1)) * (10 ** int(m.group(2))))

    for m in re.finditer(r"[+\-]?\d+(?:\.\d+)?(?:e[+\-]?\d+)?", q, flags=re.I):
        val = to_float(m.group(0))
        if val is not None:
            nums.append(val)

    return nums


# ============================================================
# 2. Solver templates
# ============================================================

def solve_impedance_ohm(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "impedance" not in q and " z" not in q and " z." not in q:
        return None

    U = find_value(question, "U", ["V", "mV", "kV"])
    I = find_value(question, "I", ["A", "mA"])

    if U is None:
        U = find_first_number_with_unit(question, ["V", "mV", "kV"])
    if I is None:
        I = find_first_number_with_unit(question, ["A", "mA"])

    if U is None or I is None:
        return None

    u = convert_value(U[0], U[1])
    i = convert_value(I[0], I[1])
    if i == 0:
        return None

    z = u / i

    return make_result(
        answer=fmt(z, 4),
        unit="Ω",
        formula="Z = U / I",
        explanation=f"The total impedance is given by Z = U/I. With U = {u} V and I = {i} A, Z = {u}/{i} = {fmt(z, 4)} Ω.",
        cot=[
            "Problem formalization: The target quantity is total impedance Z.",
            f"Evidence generation: Extract U = {u} V and I = {i} A.",
            "Evidence evaluation: For an AC circuit, impedance is voltage divided by current.",
            f"Calculation: Z = U/I = {u}/{i} = {fmt(z, 4)}.",
            f"Conclusion: The total impedance is {fmt(z, 4)} Ω."
        ],
        premises=["Impedance formula: Z = U/I.", f"U = {u} V.", f"I = {i} A."],
        confidence=0.96,
        source="sol_impedance_ohm",
    )


def solve_ohm_resistance(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "resistance" not in q and "resistor" not in q:
        return None

    V = find_value(question, "V", ["V", "mV", "kV"]) or find_value(question, "U", ["V", "mV", "kV"])
    I = find_value(question, "I", ["A", "mA"])

    if V is None or I is None:
        return None

    v = convert_value(V[0], V[1])
    i = convert_value(I[0], I[1])

    if i == 0:
        return None

    r = v / i

    return make_result(
        answer=fmt(r, 4),
        unit="Ω",
        formula="R = V / I",
        explanation=f"Using Ohm's law, R = V/I. With V = {v} V and I = {i} A, R = {fmt(r, 4)} Ω.",
        cot=[
            "Problem formalization: The target quantity is resistance R.",
            f"Evidence generation: Extract V = {v} V and I = {i} A.",
            "Evidence evaluation: Ohm's law applies.",
            f"Calculation: R = V/I = {v}/{i} = {fmt(r, 4)}.",
            f"Conclusion: The resistance is {fmt(r, 4)} Ω."
        ],
        premises=["Ohm's law: R = V/I.", f"V = {v} V.", f"I = {i} A."],
        confidence=0.93,
        source="sol_ohm_resistance",
    )


def solve_capacitor_energy(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "energy" not in q or ("capacitor" not in q and "capacitance" not in q):
        return None

    C = find_value(question, "C", ["F", "uF", "µF", "μF", "microF", "mF", "nF", "pF"])
    U = find_value(question, "U", ["V", "mV", "kV"]) or find_value(question, "V", ["V", "mV", "kV"])

    if C is None or U is None:
        return None

    c = convert_value(C[0], C[1])
    u = convert_value(U[0], U[1])
    e = 0.5 * c * (u ** 2)

    return make_result(
        answer=fmt(e, 6),
        unit="J",
        formula="E = 1/2 C U^2",
        explanation=f"Energy stored in a capacitor is E = 1/2 C U². With C = {c} F and U = {u} V, E = {fmt(e, 6)} J.",
        cot=[
            "Problem formalization: The target quantity is capacitor energy.",
            f"Evidence generation: Extract C = {c} F and U = {u} V.",
            "Evidence evaluation: Use capacitor energy formula E = 1/2 C U².",
            f"Calculation: E = 0.5 × {c} × {u}² = {fmt(e, 6)}.",
            f"Conclusion: The energy is {fmt(e, 6)} J."
        ],
        premises=["Capacitor energy formula: E = 1/2 C U².", f"C = {c} F.", f"U = {u} V."],
        confidence=0.96,
        source="sol_capacitor_energy",
    )


def solve_capacitor_charge(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "charge" not in q or ("capacitor" not in q and "capacitance" not in q):
        return None

    C = find_value(question, "C", ["F", "uF", "µF", "μF", "microF", "mF", "nF", "pF"])
    U = find_value(question, "U", ["V", "mV", "kV"]) or find_value(question, "V", ["V", "mV", "kV"])

    if C is None or U is None:
        return None

    c = convert_value(C[0], C[1])
    u = convert_value(U[0], U[1])
    q_charge = c * u

    return make_result(
        answer=fmt(q_charge, 6),
        unit="C",
        formula="Q = C U",
        explanation=f"The charge on a capacitor is Q = CU. With C = {c} F and U = {u} V, Q = {fmt(q_charge, 6)} C.",
        cot=[
            "Problem formalization: The target quantity is capacitor charge.",
            f"Evidence generation: Extract C = {c} F and U = {u} V.",
            "Evidence evaluation: Use Q = CU.",
            f"Calculation: Q = {c} × {u} = {fmt(q_charge, 6)}.",
            f"Conclusion: The charge is {fmt(q_charge, 6)} C."
        ],
        premises=["Capacitor charge formula: Q = CU.", f"C = {c} F.", f"U = {u} V."],
        confidence=0.94,
        source="sol_capacitor_charge",
    )


def solve_lc_resonance(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "resonance" not in q and "resonant" not in q:
        return None

    L = find_value(question, "L", ["H", "mH", "uH", "µH", "μH"])
    C = find_value(question, "C", ["F", "uF", "µF", "μF", "microF", "mF", "nF", "pF"])

    if L is None or C is None:
        return None

    l = convert_value(L[0], L[1])
    c = convert_value(C[0], C[1])

    if l <= 0 or c <= 0:
        return None

    f = 1.0 / (2.0 * PI * math.sqrt(l * c))

    return make_result(
        answer=fmt(f, 3),
        unit="Hz",
        formula="f = 1 / (2π√(LC))",
        explanation=f"The resonance frequency is f = 1/(2π√LC). With L = {l} H and C = {c} F, f = {fmt(f, 3)} Hz.",
        cot=[
            "Problem formalization: The target quantity is resonance frequency f.",
            f"Evidence generation: Extract L = {l} H and C = {c} F.",
            "Evidence evaluation: Convert capacitance/inductance to SI units and use f = 1/(2π√LC).",
            f"Calculation: f = 1/(2π√({l}×{c})) = {fmt(f, 3)}.",
            f"Conclusion: The resonance frequency is {fmt(f, 3)} Hz."
        ],
        premises=["LC resonance formula: f = 1/(2π√LC).", f"L = {l} H.", f"C = {c} F."],
        confidence=0.96,
        source="sol_lc_resonance",
    )


def solve_resultant_two_forces(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "resultant" not in q and "net force" not in q:
        return None

    if "force" not in q:
        return None

    # captures "magnitudes of 2 N and 12 N" or "F1 = 6 N and F2 = 5 N"
    nums_n = []
    for m in re.finditer(r"([+\-]?\d+(?:\.\d+)?)\s*N\b", normalize_text(question), flags=re.I):
        nums_n.append(float(m.group(1)))

    angle_match = re.search(r"angle(?:\s+of)?\s*([+\-]?\d+(?:\.\d+)?)\s*(?:degree|degrees|°)", question, flags=re.I)
    if angle_match is None:
        angle_match = re.search(r"([+\-]?\d+(?:\.\d+)?)\s*(?:degree|degrees|°)\s*(?:to each other|between)", question, flags=re.I)

    if len(nums_n) < 2 or angle_match is None:
        return None

    f1, f2 = nums_n[0], nums_n[1]
    theta = float(angle_match.group(1))
    r = math.sqrt(f1 ** 2 + f2 ** 2 + 2 * f1 * f2 * math.cos(math.radians(theta)))

    return make_result(
        answer=fmt(r, 4),
        unit="N",
        formula="R^2 = F1^2 + F2^2 + 2F1F2cosθ",
        explanation=f"The resultant of two forces is R = sqrt(F1² + F2² + 2F1F2cosθ). With F1 = {f1} N, F2 = {f2} N, θ = {theta}°, R = {fmt(r, 4)} N.",
        cot=[
            "Problem formalization: The target quantity is the resultant force magnitude.",
            f"Evidence generation: Extract F1 = {f1} N, F2 = {f2} N, and θ = {theta}°.",
            "Evidence evaluation: Use the law of cosines for two force vectors.",
            f"Calculation: R = sqrt({f1}² + {f2}² + 2×{f1}×{f2}×cos({theta}°)) = {fmt(r, 4)}.",
            f"Conclusion: The resultant force is {fmt(r, 4)} N."
        ],
        premises=["Resultant force formula: R² = F1² + F2² + 2F1F2cosθ."],
        confidence=0.95,
        source="sol_resultant_two_forces",
    )


def solve_coulomb_force(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "coulomb" not in q and "electric force" not in q:
        return None

    if "resultant" in q or "net" in q:
        # let vector solvers handle it
        return None

    # q1, q2
    q_values = []
    for m in re.finditer(r"q\d?\s*=?\s*([+\-]?\d+(?:\.\d+)?(?:\s*(?:x|\*)\s*10\s*\^?\s*[+\-]?\d+|\.10\^?[+\-]?\d+|e[+\-]?\d+)?)\s*C", normalize_text(question), flags=re.I):
        val = to_float(m.group(1))
        if val is not None:
            q_values.append(abs(val))

    if len(q_values) == 1 and re.search(r"q1\s*=\s*q2", question, flags=re.I):
        q_values = [q_values[0], q_values[0]]

    # distance
    d = find_first_number_with_unit(question, ["m", "cm", "mm", "km"])
    if len(q_values) < 2 or d is None:
        return None

    q1, q2 = q_values[0], q_values[1]
    r = convert_value(d[0], d[1])

    if r <= 0:
        return None

    f = K_COULOMB * q1 * q2 / (r ** 2)

    return make_result(
        answer=fmt(f, 6),
        unit="N",
        formula="F = k |q1 q2| / r^2",
        explanation=f"By Coulomb's law, F = k|q1q2|/r². With q1 = {q1} C, q2 = {q2} C, r = {r} m, F = {fmt(f, 6)} N.",
        cot=[
            "Problem formalization: The target quantity is the electrostatic force magnitude.",
            f"Evidence generation: Extract q1 = {q1} C, q2 = {q2} C, and r = {r} m.",
            "Evidence evaluation: Coulomb's law applies to point charges.",
            f"Calculation: F = 9e9×{q1}×{q2}/{r}² = {fmt(f, 6)}.",
            f"Conclusion: The electric force is {fmt(f, 6)} N."
        ],
        premises=["Coulomb's law: F = k|q1q2|/r².", "k = 9×10⁹ N·m²/C²."],
        confidence=0.92,
        source="sol_coulomb_force",
    )


def solve_right_angle_three_charges(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "right-angle" not in q and "right angle" not in q:
        return None

    if "charge" not in q or "force" not in q:
        return None

    # charge q = +5 × 10^-6 C, side length a = 10 cm
    q_match = re.search(r"q\s*=\s*([+\-]?\d+(?:\.\d+)?(?:\s*(?:x|\*)\s*10\s*\^?\s*[+\-]?\d+|\.10\^?[+\-]?\d+|e[+\-]?\d+)?)\s*C", normalize_text(question), flags=re.I)
    d = find_value(question, "a", ["m", "cm", "mm"]) or find_first_number_with_unit(question, ["cm", "m", "mm"])

    if q_match is None or d is None:
        return None

    qval = abs(to_float(q_match.group(1)))
    a = convert_value(d[0], d[1])

    if qval is None or a <= 0:
        return None

    f_one = K_COULOMB * qval * qval / (a ** 2)
    net = math.sqrt(2) * f_one

    return make_result(
        answer=fmt(net, 3),
        unit="N",
        formula="F_net = √2 · kq²/a²",
        explanation=f"At the right-angle vertex, the two equal repulsive forces are perpendicular. Each force is F = kq²/a². Thus F_net = √2F = {fmt(net, 3)} N.",
        cot=[
            "Problem formalization: The target quantity is the net force on the charge at the right-angle vertex.",
            f"Evidence generation: Extract q = {qval} C and a = {a} m.",
            "Evidence evaluation: The two equal forces are perpendicular, so vector addition gives √2 times one force.",
            f"Calculation: F_net = √2 × 9e9 × {qval}² / {a}² = {fmt(net, 3)}.",
            f"Conclusion: The net force is {fmt(net, 3)} N."
        ],
        premises=["Coulomb's law: F = kq²/a².", "Perpendicular equal vectors combine as F_net = √2F."],
        confidence=0.93,
        source="sol_right_angle_three_charges",
    )


def solve_equilateral_electric_field(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "equilateral triangle" not in q:
        return None

    if "electric field" not in q:
        return None

    q_match = re.search(r"q1\s*=\s*q2\s*=\s*([+\-]?\d+(?:\.\d+)?(?:\s*(?:x|\*)\s*10\s*\^?\s*[+\-]?\d+|\.10\^?[+\-]?\d+|e[+\-]?\d+)?)\s*C", normalize_text(question), flags=re.I)
    d = find_first_number_with_unit(question, ["cm", "m", "mm"])

    if q_match is None or d is None:
        return None

    qval = abs(to_float(q_match.group(1)))
    a = convert_value(d[0], d[1])

    if qval is None or a <= 0:
        return None

    e_single = K_COULOMB * qval / (a ** 2)
    # two equal field vectors at 60 degrees -> resultant = sqrt(3) E
    e_net = math.sqrt(3) * e_single

    return make_result(
        answer=fmt(e_net, 6),
        unit="V/m",
        formula="E_net = √3 · kq/a²",
        explanation=f"At the third vertex of an equilateral triangle, two equal electric fields form a 60° angle. Therefore E_net = √3·kq/a² = {fmt(e_net, 6)} V/m.",
        cot=[
            "Problem formalization: The target quantity is the electric field magnitude at the third vertex.",
            f"Evidence generation: Extract q = {qval} C and side length a = {a} m.",
            "Evidence evaluation: The two field vectors have equal magnitude and angle 60°.",
            f"Calculation: E_net = √3 × 9e9 × {qval}/{a}² = {fmt(e_net, 6)}.",
            f"Conclusion: The electric field magnitude is {fmt(e_net, 6)} V/m."
        ],
        premises=["Electric field of point charge: E = kq/r².", "Two equal vectors separated by 60° combine to √3 times one vector."],
        confidence=0.92,
        source="sol_equilateral_electric_field",
    )


def solve_relative_error(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "relative error" not in q:
        return None

    lc = re.search(r"least count(?:\s+of)?\s*([0-9.]+)\s*([a-zA-Z%]+)?", question, flags=re.I)
    read = re.search(r"(?:reads?|reading|measurement|measured value)\s*(?:is|=)?\s*([0-9.]+)\s*([a-zA-Z%]+)?", question, flags=re.I)

    if lc is None or read is None:
        nums = extract_all_numbers(question)
        if len(nums) >= 2:
            abs_err, measured = nums[0], nums[1]
        else:
            return None
    else:
        abs_err = float(lc.group(1))
        measured = float(read.group(1))

    if measured == 0:
        return None

    rel = abs_err / measured

    return make_result(
        answer=fmt(rel, 6),
        unit="",
        formula="relative error = absolute error / measured value",
        explanation=f"Relative error is Δx/x. With absolute error {abs_err} and measured value {measured}, relative error = {fmt(rel, 6)}.",
        cot=[
            "Problem formalization: The target quantity is relative error.",
            f"Evidence generation: Extract absolute error = {abs_err} and measured value = {measured}.",
            "Evidence evaluation: Relative error is the ratio of absolute error to measured value.",
            f"Calculation: relative error = {abs_err}/{measured} = {fmt(rel, 6)}.",
            f"Conclusion: The relative error is {fmt(rel, 6)}."
        ],
        premises=["Relative error formula: Δx/x."],
        confidence=0.88,
        source="sol_relative_error",
    )


def solve_percentage_uncertainty(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "percentage uncertainty" not in q and "percentage relative uncertainty" not in q:
        return None

    m = re.search(r"([0-9.]+)\s*±\s*([0-9.]+)", question)

    if m:
        measured = float(m.group(1))
        uncertainty = float(m.group(2))
    else:
        nums = extract_all_numbers(question)
        if len(nums) < 2:
            return None
        measured, uncertainty = nums[0], nums[1]

    if measured == 0:
        return None

    pct = uncertainty / measured * 100.0

    return make_result(
        answer=fmt(pct, 4),
        unit="%",
        formula="percentage uncertainty = Δx / x × 100%",
        explanation=f"Percentage uncertainty is Δx/x × 100%. With x = {measured} and Δx = {uncertainty}, it is {fmt(pct, 4)}%.",
        cot=[
            "Problem formalization: The target quantity is percentage uncertainty.",
            f"Evidence generation: Extract x = {measured} and Δx = {uncertainty}.",
            "Evidence evaluation: Use percentage uncertainty = Δx/x × 100%.",
            f"Calculation: {uncertainty}/{measured} × 100% = {fmt(pct, 4)}%.",
            f"Conclusion: The percentage uncertainty is {fmt(pct, 4)}%."
        ],
        premises=["Percentage uncertainty formula: Δx/x × 100%."],
        confidence=0.90,
        source="sol_percentage_uncertainty",
    )


def solve_inductor_energy(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "energy" not in q:
        return None

    if "inductor" not in q and "magnetic field energy" not in q and "magnetic energy" not in q:
        return None

    if "halved" in q and "current" in q:
        return make_result(
            answer="0.25",
            unit="of original",
            formula="W ∝ I²",
            explanation="Magnetic energy in an inductor is W = 1/2 LI². If current is halved, energy becomes (1/2)² = 1/4 = 0.25 of the original.",
            cot=[
                "Problem formalization: The target is the change in magnetic energy.",
                "Evidence generation: Current is halved.",
                "Evidence evaluation: Magnetic energy is proportional to I².",
                "Calculation: (1/2)² = 1/4 = 0.25.",
                "Conclusion: The energy becomes 0.25 of the original."
            ],
            premises=["Inductor energy: W = 1/2 LI²."],
            confidence=0.94,
            source="sol_inductor_energy_ratio",
        )

    L = find_value(question, "L", ["H", "mH", "uH", "µH", "μH"])
    I = find_value(question, "I", ["A", "mA"])

    if L is None or I is None:
        return None

    l = convert_value(L[0], L[1])
    i = convert_value(I[0], I[1])
    w = 0.5 * l * i * i

    return make_result(
        answer=fmt(w, 6),
        unit="J",
        formula="W = 1/2 L I²",
        explanation=f"Magnetic energy stored in an inductor is W = 1/2 LI². With L = {l} H and I = {i} A, W = {fmt(w, 6)} J.",
        cot=[
            "Problem formalization: The target quantity is magnetic energy.",
            f"Evidence generation: Extract L = {l} H and I = {i} A.",
            "Evidence evaluation: Use inductor energy formula W = 1/2 LI².",
            f"Calculation: W = 0.5 × {l} × {i}² = {fmt(w, 6)}.",
            f"Conclusion: The magnetic energy is {fmt(w, 6)} J."
        ],
        premises=["Inductor energy formula: W = 1/2 LI²."],
        confidence=0.94,
        source="sol_inductor_energy",
    )


def solve_train_relative_speed(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if "train" not in q or "opposite direction" not in q:
        return None

    car = re.search(r"car.*?speed\s*of\s*([0-9.]+)\s*km/h", question, flags=re.I)
    length = re.search(r"([0-9.]+)\s*m\s*long\s*train", question, flags=re.I)
    time = re.search(r"passes\s*it\s*in\s*([0-9.]+)\s*seconds", question, flags=re.I)

    if car is None or length is None or time is None:
        return None

    v_car_kmh = float(car.group(1))
    d = float(length.group(1))
    t = float(time.group(1))

    rel_ms = d / t
    rel_kmh = rel_ms * 3.6
    v_train = rel_kmh - v_car_kmh

    return make_result(
        answer=fmt(v_train, 3),
        unit="km/h",
        formula="v_train = (L/t)·3.6 - v_car",
        explanation=f"In opposite directions, relative speed is v_train + v_car. The train covers its length in time t, so relative speed = L/t = {rel_ms} m/s = {fmt(rel_kmh,3)} km/h. Thus v_train = {fmt(v_train,3)} km/h.",
        cot=[
            "Problem formalization: The target quantity is train speed.",
            f"Evidence generation: Extract car speed = {v_car_kmh} km/h, train length = {d} m, time = {t} s.",
            "Evidence evaluation: Opposite-direction passing uses relative speed v_train + v_car.",
            f"Calculation: relative speed = {d}/{t}×3.6 = {fmt(rel_kmh,3)} km/h, so train speed = {fmt(rel_kmh,3)} - {v_car_kmh} = {fmt(v_train,3)}.",
            f"Conclusion: The train speed is {fmt(v_train,3)} km/h."
        ],
        premises=["Opposite direction relative speed: v_rel = v_train + v_car."],
        confidence=0.90,
        source="sol_train_relative_speed",
    )


def solve_reactance(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    f_val = find_value(question, "f", ["Hz", "kHz", "MHz"])
    L = find_value(question, "L", ["H", "mH", "uH", "µH", "μH"])
    C = find_value(question, "C", ["F", "uF", "µF", "μF", "mF", "nF", "pF"])

    if f_val is None:
        return None

    f = convert_value(f_val[0], f_val[1])

    if ("inductive reactance" in q or "xl" in q) and L is not None:
        l = convert_value(L[0], L[1])
        xl = 2 * PI * f * l

        return make_result(
            answer=fmt(xl, 4),
            unit="Ω",
            formula="X_L = 2πfL",
            explanation=f"Inductive reactance is X_L = 2πfL. With f = {f} Hz and L = {l} H, X_L = {fmt(xl, 4)} Ω.",
            cot=[
                "Problem formalization: The target quantity is inductive reactance.",
                f"Evidence generation: Extract f = {f} Hz and L = {l} H.",
                "Evidence evaluation: Use X_L = 2πfL.",
                f"Calculation: X_L = 2π×{f}×{l} = {fmt(xl, 4)}.",
                f"Conclusion: The inductive reactance is {fmt(xl, 4)} Ω."
            ],
            premises=["Inductive reactance formula: X_L = 2πfL."],
            confidence=0.92,
            source="sol_inductive_reactance",
        )

    if ("capacitive reactance" in q or "xc" in q) and C is not None:
        c = convert_value(C[0], C[1])
        if f <= 0 or c <= 0:
            return None

        xc = 1 / (2 * PI * f * c)

        return make_result(
            answer=fmt(xc, 4),
            unit="Ω",
            formula="X_C = 1/(2πfC)",
            explanation=f"Capacitive reactance is X_C = 1/(2πfC). With f = {f} Hz and C = {c} F, X_C = {fmt(xc, 4)} Ω.",
            cot=[
                "Problem formalization: The target quantity is capacitive reactance.",
                f"Evidence generation: Extract f = {f} Hz and C = {c} F.",
                "Evidence evaluation: Use X_C = 1/(2πfC).",
                f"Calculation: X_C = 1/(2π×{f}×{c}) = {fmt(xc, 4)}.",
                f"Conclusion: The capacitive reactance is {fmt(xc, 4)} Ω."
            ],
            premises=["Capacitive reactance formula: X_C = 1/(2πfC)."],
            confidence=0.92,
            source="sol_capacitive_reactance",
        )

    return None


# ============================================================
# 3. Main router
# ============================================================

SOLVERS = [
    solve_impedance_ohm,
    solve_lc_resonance,
    solve_capacitor_energy,
    solve_capacitor_charge,
    solve_resultant_two_forces,
    solve_right_angle_three_charges,
    solve_equilateral_electric_field,
    solve_coulomb_force,
    solve_relative_error,
    solve_percentage_uncertainty,
    solve_inductor_energy,
    solve_train_relative_speed,
    solve_reactance,
    solve_ohm_resistance,
]


def solve_physics(question: str, extra_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Main physics SOL solver used by:
    - SFT distillation
    - RL reward
    - inference/evaluation/API override

    It returns None if no reliable formula template matches.
    """
    question = normalize_text(question)

    if not question:
        return None

    for solver in SOLVERS:
        try:
            out = solver(question)
            if out is not None and out.get("answer") not in [None, ""]:
                out["question"] = question
                return out
        except Exception as e:
            # Keep solver robust; never crash pipeline.
            continue

    return None
