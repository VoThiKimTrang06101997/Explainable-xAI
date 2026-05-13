import ast
import json
import math
import re


def parse_obj(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            try:
                return ast.literal_eval(x)
            except Exception:
                return {}
    return {}


def normalize_superscript(s):
    table = str.maketrans({
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁻": "-", "⁺": "+",
    })
    return str(s).translate(table)


def is_nan_like(x):
    if x is None:
        return True
    s = str(x).strip().lower()
    return s in ["", "nan", "none", "null", "na"]


def norm(x):
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = x.replace("μ", "u").replace("µ", "u")
    x = x.replace("−", "-")
    x = re.sub(r"\s+", " ", x)
    return x.strip(" .,:;\"'`")


def extract_json(text):
    if not text:
        return None

    text = str(text).strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        return json.loads(text[start:end])
    except Exception:
        return None


def extract_first_number(x):
    if x is None:
        return None

    s = normalize_superscript(str(x))
    s = s.replace("×", "x")
    s = s.replace("\\times", "x")
    s = s.replace("−", "-")
    s = s.replace("{", "").replace("}", "")

    # Handle a\sqrt{b} × 10^-n
    sqrt_sci = re.search(
        r"(-?\d+(?:\.\d+)?)?\s*\\?sqrt\s*\(?\s*([0-9.]+)\s*\)?\s*(?:x|\*)\s*10\s*\^?\s*(-?\d+)",
        s,
        flags=re.I,
    )
    if sqrt_sci:
        coef = float(sqrt_sci.group(1)) if sqrt_sci.group(1) else 1.0
        root = float(sqrt_sci.group(2))
        exp = int(sqrt_sci.group(3))
        return coef * math.sqrt(root) * (10 ** exp)

    # Handle 24.45 × 10^-3, 1.22 . 10^{-3}
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

    return float(nums[0])


def answer_score(pred, gold, task_type):
    if is_nan_like(gold):
        return 0.0

    task_type = str(task_type).lower()

    if task_type == "physics":
        p = extract_first_number(pred)
        g = extract_first_number(gold)

        if p is not None and g is not None:
            rel_err = abs(p - g) / max(1.0, abs(g))
            if rel_err <= 1e-2:
                return 1.0
            if rel_err <= 5e-2:
                return 0.6
            if rel_err <= 1e-1:
                return 0.3
            return 0.0

        return 1.0 if norm(pred) == norm(gold) else 0.0

    pred_n = norm(pred)
    gold_n = norm(gold)

    if pred_n == gold_n:
        return 1.0

    tokens = re.findall(r"[a-z0-9]+", pred_n)
    if gold_n in tokens and gold_n in ["a", "b", "c", "d", "yes", "no", "unknown", "false", "true"]:
        return 0.5

    return 0.0


def text_quality_score(text):
    if not text:
        return 0.0

    text = str(text)
    score = 0.0

    if len(text.split()) >= 8:
        score += 0.25
    if len(text.split()) >= 20:
        score += 0.20
    if any(k in text.lower() for k in [
        "because", "therefore", "thus", "hence", "formula",
        "premise", "substitute", "apply", "derive", "calculate",
    ]):
        score += 0.30
    if re.search(r"\d", text):
        score += 0.10
    if "{" in text and "}" in text:
        score += 0.15

    return min(score, 1.0)


def clamp01(x):
    return max(0.0, min(1.0, float(x)))
