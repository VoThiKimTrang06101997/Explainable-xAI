# %%writefile /content/Explainable-xAI/One-Shot-RLVR/exact_modules/logicality_metrics.py
import re


def normalize_text(x):
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = x.replace("−", "-").replace("μ", "u").replace("µ", "u")
    x = re.sub(r"\s+", " ", x)
    return x


def is_nan_like(x):
    if x is None:
        return True
    return str(x).strip().lower() in ["", "nan", "none", "null", "na"]


def normalize_superscript(s):
    table = str.maketrans({
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁻": "-", "⁺": "+",
    })
    return str(s).translate(table)


def extract_first_number(x):
    if x is None:
        return None

    s = normalize_superscript(str(x))
    s = s.replace("\\times", "x").replace("×", "x")
    s = s.replace("−", "-").replace("{", "").replace("}", "")

    sci = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:x|×|\*|\.)\s*10\s*(?:\^|\*\*)?\s*(-?\d+)",
        s,
        flags=re.I,
    )
    if sci:
        return float(sci.group(1)) * (10 ** int(sci.group(2)))

    nums = re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s, flags=re.I)
    if not nums:
        return None

    try:
        return float(nums[0])
    except Exception:
        return None




def extract_all_numbers_metric(x):
    if x is None:
        return []

    s = normalize_superscript(str(x))
    s = s.replace("\\times", "x").replace("×", "x")
    s = s.replace("−", "-").replace("{", "").replace("}", "")

    vals = []

    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:x|×|\*|\.)\s*10\s*(?:\^|\*\*)?\s*(-?\d+)", s, flags=re.I):
        vals.append(float(m.group(1)) * (10 ** int(m.group(2))))

    for m in re.finditer(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s, flags=re.I):
        try:
            vals.append(float(m.group(0)))
        except Exception:
            pass

    return vals


def p1_correctness_continuous(pred, gold, task_type):
    if is_nan_like(gold):
        return 0.0

    task_type = str(task_type).lower()

    if task_type == "physics":
        ps = extract_all_numbers_metric(pred)
        gs = extract_all_numbers_metric(gold)

        if len(ps) == 0 or len(gs) == 0:
            return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

        # Multi-answer case: require all gold numbers to be close to predicted numbers.
        if len(gs) >= 2 and len(ps) >= 2:
            scores = []
            for g in gs[:len(ps)]:
                best = max(
                    [
                        1.0 if abs(p - g) / max(1.0, abs(g)) <= 0.01 else
                        0.9 if abs(p - g) / max(1.0, abs(g)) <= 0.03 else
                        0.8 if abs(p - g) / max(1.0, abs(g)) <= 0.05 else
                        0.6 if abs(p - g) / max(1.0, abs(g)) <= 0.10 else
                        0.4 if abs(p - g) / max(1.0, abs(g)) <= 0.20 else
                        0.2 if abs(p - g) / max(1.0, abs(g)) <= 0.50 else
                        0.0
                        for p in ps
                    ]
                )
                scores.append(best)
            return sum(scores) / max(1, len(scores))

        p = ps[0]
        g = gs[0]
        rel_err = abs(p - g) / max(1.0, abs(g))

        if rel_err <= 0.01:
            return 1.0
        if rel_err <= 0.03:
            return 0.9
        if rel_err <= 0.05:
            return 0.8
        if rel_err <= 0.10:
            return 0.6
        if rel_err <= 0.20:
            return 0.4
        if rel_err <= 0.50:
            return 0.2
        return 0.0

    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)

    label_map = {"true": "yes", "false": "no"}
    pred_n = label_map.get(pred_n, pred_n)
    gold_n = label_map.get(gold_n, gold_n)

    if gold_n in ["a", "b", "c", "d"]:
        if pred_n[:1] == gold_n:
            return 1.0
        if pred_n[:1] in ["a", "b", "c", "d"]:
            return 0.2
        return 0.0

    if gold_n in ["yes", "no", "unknown"]:
        if pred_n == gold_n:
            return 1.0
        if pred_n in ["yes", "no", "unknown"]:
            return 0.2
        return 0.0

    return 1.0 if pred_n == gold_n else 0.0


def p2_explanation_logicality(obj, p1, json_valid_original):
    if obj is None:
        return 0.0

    exp = str(obj.get("explanation", "") or "")
    fol = str(obj.get("fol", "") or "")
    cot = obj.get("cot", [])
    premises = obj.get("premises", [])

    score = 0.0

    if len(exp.split()) >= 8:
        score += 0.15
    if len(exp.split()) >= 20:
        score += 0.15

    evidence_words = [
        "because", "therefore", "thus", "hence", "formula", "premise",
        "given", "substitute", "calculate", "derive", "support",
        "contradict", "entail", "evidence", "compare", "verify"
    ]

    if any(w in exp.lower() for w in evidence_words):
        score += 0.20

    if isinstance(premises, list) and len(premises) >= 1:
        score += 0.15
    if isinstance(fol, str) and len(fol.strip()) > 0:
        score += 0.15
    if isinstance(cot, list) and len(cot) >= 3:
        score += 0.20

    score = min(score, 1.0)

    if not json_valid_original:
        score *= 0.75

    if p1 == 0:
        score = min(score, 0.55)
    elif p1 < 0.5:
        score = min(score, 0.65)
    elif p1 < 1.0:
        score = min(score, 0.85)

    return round(score, 4)


def p3_reasoning_depth_logicality(obj, p1, json_valid_original):
    if obj is None:
        return 0.0

    cot = obj.get("cot", [])
    premises = obj.get("premises", [])
    fol = str(obj.get("fol", "") or "")

    if not isinstance(cot, list):
        cot = []

    cot_text = " ".join([str(x).lower() for x in cot])

    score = 0.0

    if len(cot) >= 2:
        score += 0.10
    if len(cot) >= 4:
        score += 0.10
    if len(cot) >= 5:
        score += 0.10

    if isinstance(premises, list) and len(premises) >= 1:
        score += 0.10
    if isinstance(premises, list) and len(premises) >= 2:
        score += 0.10

    if len(fol.strip()) > 0:
        score += 0.10

    nexus = {
        "formalization": ["formalization", "identify", "target", "claim", "question"],
        "evidence_generation": ["evidence generation", "given", "extract", "known", "premise"],
        "evidence_evaluation": ["evidence evaluation", "check", "compare", "verify", "support", "contradict", "entail"],
        "calculation_or_inference": ["calculation", "inference", "calculate", "compute", "derive", "infer", "substitute"],
        "conclusion": ["conclusion", "therefore", "thus", "answer", "return"],
    }

    hits = 0
    for _, words in nexus.items():
        if any(w in cot_text for w in words):
            hits += 1

    score += 0.40 * (hits / len(nexus))
    score = min(score, 1.0)

    if not json_valid_original:
        score *= 0.75

    if p1 == 0:
        score = min(score, 0.60)
    elif p1 < 0.5:
        score = min(score, 0.70)
    elif p1 < 1.0:
        score = min(score, 0.90)

    return round(score, 4)


def final_proxy_score(p1, p2, p3):
    return round(0.60 * p1 + 0.20 * p2 + 0.20 * p3, 4)


# ============================================================
# HARD OVERRIDE METRICS PATCH
# ============================================================

def _extract_numbers_clean(x):
    if x is None:
        return []

    s = normalize_superscript(str(x))
    s = s.replace("\\times", "×").replace("x", "×").replace("*", "×")
    s = s.replace("−", "-").replace("{", "").replace("}", "")

    values = []
    spans = []

    sci_pattern = r"(-?\d+(?:\.\d+)?)\s*(?:×|\.)\s*10\s*(?:\^|\*\*)?\s*(-?\d+)"
    for m in re.finditer(sci_pattern, s, flags=re.I):
        try:
            values.append(float(m.group(1)) * (10 ** int(m.group(2))))
            spans.append(m.span())
        except Exception:
            pass

    def inside_span(pos):
        return any(a <= pos < b for a, b in spans)

    for m in re.finditer(r"-?\d+(?:\.\d+)?(?:e[+\-]?\d+)?", s, flags=re.I):
        if inside_span(m.start()):
            continue
        try:
            values.append(float(m.group(0)))
        except Exception:
            pass

    return values


def _score_numeric(p, g):
    rel_err = abs(p - g) / max(1.0, abs(g))
    if rel_err <= 0.01:
        return 1.0
    if rel_err <= 0.03:
        return 0.9
    if rel_err <= 0.05:
        return 0.8
    if rel_err <= 0.10:
        return 0.6
    if rel_err <= 0.20:
        return 0.4
    if rel_err <= 0.50:
        return 0.2
    return 0.0


def p1_correctness_continuous(pred, gold, task_type):
    if is_nan_like(gold):
        return 0.0

    task_type = str(task_type).lower()

    if task_type == "physics":
        ps = _extract_numbers_clean(pred)
        gs = _extract_numbers_clean(gold)

        if not ps or not gs:
            return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

        # multi-answer: each gold number should be matched by a predicted number
        if len(gs) >= 2:
            scores = []
            for g in gs:
                scores.append(max(_score_numeric(p, g) for p in ps))
            return sum(scores) / len(scores)

        return max(_score_numeric(p, gs[0]) for p in ps)

    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)

    label_map = {"true": "yes", "false": "no"}
    pred_n = label_map.get(pred_n, pred_n)
    gold_n = label_map.get(gold_n, gold_n)

    if gold_n in ["a", "b", "c", "d"]:
        if pred_n[:1] == gold_n:
            return 1.0
        if pred_n[:1] in ["a", "b", "c", "d"]:
            return 0.2
        return 0.0

    if gold_n in ["yes", "no", "unknown"]:
        if pred_n == gold_n:
            return 1.0
        if pred_n in ["yes", "no", "unknown"]:
            return 0.2
        return 0.0

    return 1.0 if pred_n == gold_n else 0.0


