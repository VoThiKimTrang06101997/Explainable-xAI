def convert_capacitance(value, unit):
    unit = str(unit).lower()

    if unit in ["μf", "uf", "microf"]:
        return float(value) * 1e-6

    if unit == "nf":
        return float(value) * 1e-9

    if unit == "pf":
        return float(value) * 1e-12

    return float(value)


def convert_length(value, unit):
    unit = str(unit).lower()

    if unit == "cm":
        return float(value) / 100.0

    if unit == "mm":
        return float(value) / 1000.0

    if unit == "km":
        return float(value) * 1000.0

    return float(value)


def kmh_to_mps(v):
    return float(v) / 3.6


def mps_to_kmh(v):
    return float(v) * 3.6
