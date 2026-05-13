import math
import re
from typing import Any, Dict, Optional

from exact_modules.common import normalize_superscript
from exact_modules.sol.unit_converter import convert_capacitance, convert_length, mps_to_kmh


def _extract(pattern, text, flags=re.I):
    return re.search(pattern, text, flags)


def solve_lc_resonance(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()
    if "resonance frequency" not in q and "resonant frequency" not in q:
        return None

    m_l = _extract(r"L\s*=\s*([0-9.]+)\s*H", question)
    m_c = _extract(r"C\s*=\s*([0-9.]+)\s*(μF|uF|microF|nF|pF|F)", question)

    if not (m_l and m_c):
        return None

    L = float(m_l.group(1))
    C_raw = float(m_c.group(1))
    C_unit = m_c.group(2)
    C = convert_capacitance(C_raw, C_unit)

    f = 1.0 / (2.0 * math.pi * math.sqrt(L * C))

    return {
        "answer": f"{f:.2f}",
        "unit": "Hz",
        "fol": "f = 1 / (2π√(LC))",
        "explanation": (
            f"The resonance frequency is calculated using f = 1 / (2π√(LC)). "
            f"Given L = {L} H and C = {C_raw} {C_unit}, C = {C} F. "
            f"Substituting these values gives f = {f:.2f} Hz."
        ),
        "cot": [
            "Step 1: Identify the problem as an LC resonance frequency problem.",
            "Step 2: Use the formula f = 1 / (2π√(LC)).",
            f"Step 3: Convert capacitance: C = {C_raw} {C_unit} = {C} F.",
            f"Step 4: Substitute L = {L} H and C = {C} F.",
            f"Step 5: Compute f = {f:.2f} Hz.",
        ],
        "premises": [
            "LC resonance formula: f = 1 / (2π√(LC)).",
            f"Given inductance: L = {L} H.",
            f"Given capacitance: C = {C_raw} {C_unit}.",
        ],
        "confidence": 0.92,
        "source": "sol_lc_resonance",
    }


def solve_train_relative_speed(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()
    if not ("car" in q and "train" in q and "opposite direction" in q):
        return None

    m_car = _extract(r"car.*?speed of\s*([0-9.]+)\s*km/h", question)
    m_len = _extract(r"([0-9.]+)\s*m\s*long train", question)
    m_time = _extract(r"passes it in\s*([0-9.]+)\s*seconds", question)

    if not (m_car and m_len and m_time):
        return None

    car_kmh = float(m_car.group(1))
    train_len_m = float(m_len.group(1))
    time_s = float(m_time.group(1))

    relative_mps = train_len_m / time_s
    relative_kmh = mps_to_kmh(relative_mps)
    train_kmh = relative_kmh - car_kmh

    return {
        "answer": f"{train_kmh:.2f}",
        "unit": "km/h",
        "fol": "v_relative = d / t ∧ v_train = v_relative - v_car",
        "explanation": (
            f"The train passes the car over a distance equal to the train length. "
            f"The relative speed is {train_len_m}/{time_s} = {relative_mps:.2f} m/s "
            f"= {relative_kmh:.2f} km/h. Since they move in opposite directions, "
            f"v_train = {relative_kmh:.2f} - {car_kmh} = {train_kmh:.2f} km/h."
        ),
        "cot": [
            "Step 1: Treat the passing event as a relative motion problem.",
            f"Step 2: Compute relative speed = {train_len_m} m / {time_s} s.",
            f"Step 3: Convert relative speed to {relative_kmh:.2f} km/h.",
            "Step 4: Since directions are opposite, v_train = v_relative - v_car.",
            f"Step 5: Compute v_train = {train_kmh:.2f} km/h.",
        ],
        "premises": [
            "Relative speed = distance / time.",
            "For opposite directions: v_relative = v_train + v_car.",
            f"Car speed = {car_kmh} km/h.",
            f"Train length = {train_len_m} m.",
            f"Passing time = {time_s} s.",
        ],
        "confidence": 0.91,
        "source": "sol_train_relative_speed",
    }


def solve_equal_charges_equilateral(question: str) -> Optional[Dict[str, Any]]:
    q = normalize_superscript(question)
    q_low = q.lower()

    if "equilateral triangle" not in q_low or "electric field at vertex a" not in q_low:
        return None

    m_q = re.search(
        r"q1\s*=\s*q2\s*=\s*([0-9.]+)\s*(?:[x×.]?\s*10\^?\{?(-?[0-9]+)\}?)?\s*C",
        q,
        flags=re.I,
    )
    m_side = re.search(r"side length of\s*([0-9.]+)\s*(cm|m)", q, flags=re.I)

    if not (m_q and m_side):
        return None

    charge = float(m_q.group(1))
    if m_q.group(2):
        charge *= 10 ** int(m_q.group(2))

    side = convert_length(float(m_side.group(1)), m_side.group(2))

    k = 8.99e9
    e_each = k * charge / (side ** 2)
    e_total = math.sqrt(3) * e_each

    return {
        "answer": f"{e_total:.3g}",
        "unit": "N/C",
        "fol": "E_total = √3 · kq / a²",
        "explanation": (
            f"Each charge produces an electric field E = kq/a² at vertex A. "
            f"Because the triangle is equilateral and the charges are equal, the two "
            f"field vectors combine symmetrically, giving E_total = √3·kq/a² = {e_total:.3g} N/C."
        ),
        "cot": [
            "Step 1: Identify that two equal charges are placed symmetrically at B and C.",
            "Step 2: Use the point-charge electric field formula E = kq/r².",
            "Step 3: Apply symmetry for an equilateral triangle.",
            "Step 4: Combine the two equal field vectors: E_total = √3·kq/a².",
            f"Step 5: Compute E_total = {e_total:.3g} N/C.",
        ],
        "premises": [
            "Electric field of a point charge: E = kq/r².",
            "The two charges are equal.",
            "The triangle is equilateral, so the resultant field follows from symmetry.",
            f"q = {charge} C.",
            f"a = {side} m.",
        ],
        "confidence": 0.90,
        "source": "sol_equal_charges_equilateral",
    }


def solve_capacitor_energy(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()
    if "energy stored" not in q or "capacitor" not in q:
        return None

    m_c = re.search(r"C\s*=\s*([0-9.]+)\s*(μF|uF|microF|nF|pF|F)", question, re.I)
    m_u = re.search(r"(?:U|V)\s*=\s*([0-9.]+)\s*V", question, re.I)

    if not (m_c and m_u):
        return None

    C_raw = float(m_c.group(1))
    C = convert_capacitance(C_raw, m_c.group(2))
    U = float(m_u.group(1))
    energy = 0.5 * C * U * U

    return {
        "answer": f"{energy:g}",
        "unit": "J",
        "fol": "E = 1/2 · C · U²",
        "explanation": (
            f"The energy stored in a capacitor is E = 1/2·C·U². "
            f"With C = {C_raw} {m_c.group(2)} = {C} F and U = {U} V, "
            f"E = {energy:g} J."
        ),
        "cot": [
            "Step 1: Identify the problem as capacitor energy calculation.",
            "Step 2: Use the formula E = 1/2·C·U².",
            f"Step 3: Convert capacitance to farads: C = {C} F.",
            f"Step 4: Substitute U = {U} V.",
            f"Step 5: Compute E = {energy:g} J.",
        ],
        "premises": [
            "Capacitor energy formula: E = 1/2·C·U².",
            f"C = {C_raw} {m_c.group(2)}.",
            f"U = {U} V.",
        ],
        "confidence": 0.92,
        "source": "sol_capacitor_energy",
    }


def solve_ohm_law(question: str) -> Optional[Dict[str, Any]]:
    q = question.lower()

    if not any(k in q for k in ["ohm", "resistor", "voltage", "current", "resistance"]):
        return None

    m_i = re.search(r"I\s*=\s*([0-9.]+)\s*A", question, re.I)
    m_r = re.search(r"R\s*=\s*([0-9.]+)\s*(ohm|Ω)", question, re.I)

    if m_i and m_r and ("voltage" in q or "potential difference" in q):
        current = float(m_i.group(1))
        resistance = float(m_r.group(1))
        voltage = current * resistance

        return {
            "answer": f"{voltage:g}",
            "unit": "V",
            "fol": "V = I · R",
            "explanation": (
                f"Using Ohm's law V = IR, with I = {current} A and R = {resistance} Ω, "
                f"V = {voltage:g} V."
            ),
            "cot": [
                "Step 1: Identify the problem as an Ohm's law problem.",
                "Step 2: Use V = IR.",
                f"Step 3: Substitute I = {current} A and R = {resistance} Ω.",
                f"Step 4: Compute V = {voltage:g} V.",
            ],
            "premises": [
                "Ohm's law: V = IR.",
                f"Current I = {current} A.",
                f"Resistance R = {resistance} Ω.",
            ],
            "confidence": 0.90,
            "source": "sol_ohm_law",
        }

    return None


def solve_capacitance_from_charge_voltage(question: str):
    q = question.lower()
    if "capacitance" not in q and "calculate the capacitance" not in q:
        return None

    m_q = re.search(r"Q\s*=\s*([0-9.]+)\s*(mC|μC|uC|nC|C)", question, re.I)
    m_u = re.search(r"U\s*=\s*([0-9.]+)\s*V", question, re.I)

    if not (m_q and m_u):
        return None

    charge = float(m_q.group(1))
    unit = m_q.group(2).lower()

    if unit == "mc":
        charge *= 1e-3
    elif unit in ["μc", "uc"]:
        charge *= 1e-6
    elif unit == "nc":
        charge *= 1e-9

    voltage = float(m_u.group(1))
    cap_f = charge / voltage
    cap_uf = cap_f * 1e6

    return {
        "answer": f"{cap_uf:g}",
        "unit": "μF",
        "fol": "C = Q / U",
        "explanation": f"Capacitance is C = Q/U. With Q = {charge} C and U = {voltage} V, C = {cap_f} F = {cap_uf:g} μF.",
        "cot": [
            "Step 1: Identify Q and U.",
            "Step 2: Use C = Q/U.",
            "Step 3: Convert charge to coulombs.",
            f"Step 4: Compute C = {cap_uf:g} μF.",
        ],
        "premises": ["Capacitance formula: C = Q/U."],
        "confidence": 0.90,
        "source": "sol_capacitance_from_charge_voltage",
    }


def solve_inductor_energy(question: str):
    q = question.lower()
    if "inductor" not in q and "magnetic field energy" not in q:
        return None

    m_l = re.search(r"L\s*=\s*([0-9.]+)\s*(H|mH)", question, re.I)
    m_i = re.search(r"(?:I|current).*?([0-9.]+)\s*A", question, re.I)

    if not (m_l and m_i):
        return None

    L = float(m_l.group(1))
    if m_l.group(2).lower() == "mh":
        L *= 1e-3

    I = float(m_i.group(1))
    W = 0.5 * L * I * I
    W_mj = W * 1000

    # Many NL rows expect mJ.
    unit = "mJ" if "mj" in q else "J"
    ans = W_mj if unit == "mJ" else W

    return {
        "answer": f"{ans:.2f}" if unit == "mJ" else f"{ans:g}",
        "unit": unit,
        "fol": "W = 1/2 · L · I²",
        "explanation": f"Magnetic energy in an inductor is W = 1/2·L·I². With L = {L} H and I = {I} A, W = {W:g} J.",
        "cot": [
            "Step 1: Identify inductance and current.",
            "Step 2: Use W = 1/2·L·I².",
            f"Step 3: Substitute L = {L} H and I = {I} A.",
            f"Step 4: Compute W = {ans:g} {unit}.",
        ],
        "premises": ["Inductor energy formula: W = 1/2·L·I²."],
        "confidence": 0.90,
        "source": "sol_inductor_energy",
    }


def solve_rlc_resonance_yesno(question: str):
    q = question.lower()
    if "resonance" not in q or "frequency" not in q:
        return None
    if not any(x in q for x in ["does", "will", "determine if"]):
        return None

    m_l = re.search(r"L\s*=\s*([0-9.]+)\s*H", question, re.I)
    m_c = re.search(r"C\s*=\s*([0-9.]+)\s*(μF|uF|microF|F)", question, re.I)
    m_f = re.search(r"(?:frequency|f)\s*=\s*([0-9.]+)\s*Hz", question, re.I)

    if not (m_l and m_c and m_f):
        return None

    L = float(m_l.group(1))
    C = float(m_c.group(1))
    if m_c.group(2).lower() in ["μf", "uf", "microf"]:
        C *= 1e-6

    f = float(m_f.group(1))
    f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))

    ok = abs(f - f0) / max(1.0, abs(f0)) <= 0.02
    ans = "Yes" if ok else "No"

    return {
        "answer": ans,
        "unit": "-",
        "fol": "f0 = 1 / (2π√(LC)); resonance iff f ≈ f0",
        "explanation": f"The resonant frequency is f0 = 1/(2π√LC) = {f0:.2f} Hz. The operating frequency is {f} Hz, so the answer is {ans}.",
        "cot": [
            "Step 1: Identify L, C, and operating frequency.",
            "Step 2: Compute f0 = 1/(2π√LC).",
            "Step 3: Compare f with f0.",
            f"Step 4: Return {ans}.",
        ],
        "premises": ["RLC resonance condition: f = 1/(2π√LC)."],
        "confidence": 0.90,
        "source": "sol_rlc_resonance_yesno",
    }


def solve_resistance_at_resonance(question: str):
    q = question.lower()
    if "resonance" not in q and "resonant" not in q:
        return None
    if "impedance" not in q:
        return None

    m_z = re.search(r"Z\s*=?\s*([0-9.]+)\s*Ω", question, re.I)
    if not m_z:
        m_z = re.search(r"impedance\s*(?:is|=)?\s*([0-9.]+)\s*Ω", question, re.I)

    if not m_z:
        return None

    R = float(m_z.group(1))

    return {
        "answer": f"{R:g}",
        "unit": "Ω",
        "fol": "At resonance in series RLC, Z = R",
        "explanation": f"At resonance in a series RLC circuit, the inductive and capacitive reactances cancel, so the impedance equals the resistance: R = Z = {R:g} Ω.",
        "cot": [
            "Step 1: Identify the circuit is at resonance.",
            "Step 2: Use the resonance property Z = R.",
            f"Step 3: Therefore R = {R:g} Ω.",
        ],
        "premises": ["At resonance in a series RLC circuit, Z = R."],
        "confidence": 0.92,
        "source": "sol_resistance_at_resonance",
    }


def solve_solenoid_turn_density(question: str):
    q = question.lower()
    if "turns per meter" not in q and "turns/m" not in q:
        return None

    m_n = re.search(r"([0-9.]+)\s*turns", question, re.I)
    m_l = re.search(r"length\s*(?:of)?\s*([0-9.]+)\s*m", question, re.I)
    if not (m_n and m_l):
        m_l = re.search(r"is\s*([0-9.]+)\s*m\s*long", question, re.I)

    if not (m_n and m_l):
        return None

    N = float(m_n.group(1))
    L = float(m_l.group(1))
    n = N / L

    return {
        "answer": f"{n:g}",
        "unit": "turns/m",
        "fol": "n = N / l",
        "explanation": f"The turn density is n = N/l. With N = {N} turns and l = {L} m, n = {n:g} turns/m.",
        "cot": [
            "Step 1: Identify number of turns and solenoid length.",
            "Step 2: Use n = N/l.",
            f"Step 3: Compute n = {n:g} turns/m.",
        ],
        "premises": ["Solenoid turn density: n = N/l."],
        "confidence": 0.90,
        "source": "sol_solenoid_turn_density",
    }


def solve_absolute_error_least_count(question: str):
    q = question.lower()
    if "absolute error" not in q and "least count" not in q:
        return None

    m_lc = re.search(r"least count of\s*([0-9.]+)\s*([A-Za-z%Ω]+)", question, re.I)
    if not m_lc:
        return None

    err = float(m_lc.group(1))
    unit = m_lc.group(2)

    return {
        "answer": f"{err:g}",
        "unit": unit,
        "fol": "absolute error = least count",
        "explanation": f"For a direct instrument reading, the absolute error is taken as the least count, so Δ = {err:g} {unit}.",
        "cot": [
            "Step 1: Identify the instrument least count.",
            "Step 2: Use absolute error equal to least count.",
            f"Step 3: Return {err:g} {unit}.",
        ],
        "premises": ["Absolute error of a direct reading is approximated by the least count."],
        "confidence": 0.88,
        "source": "sol_absolute_error_least_count",
    }
    

def solve_physics(question: str):
    solvers = [
        solve_resistance_at_resonance,
        solve_rlc_resonance_yesno,
        solve_lc_resonance,
        solve_capacitance_from_charge_voltage,
        solve_capacitor_energy,
        solve_inductor_energy,
        solve_solenoid_turn_density,
        solve_absolute_error_least_count,
        solve_train_relative_speed,
        solve_equal_charges_equilateral,
        solve_ohm_law,
    ]

    for solver in solvers:
        try:
            result = solver(question)
            if result is not None:
                return result
        except Exception:
            continue

    return None

