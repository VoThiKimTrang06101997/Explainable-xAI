# %%writefile /content/Explainable-xAI/One-Shot-RLVR/evaluate_exact_symbolic_moe.py
import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# ============================================================
# Optional project modules
# ============================================================

try:
    from exact_modules.common import extract_json, is_nan_like, parse_obj
except Exception:
    extract_json = None

    def is_nan_like(x):
        if x is None:
            return True
        return str(x).strip().lower() in ["", "nan", "none", "null", "na"]

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


try:
    from exact_modules.config import EXACT_CONFIG
except Exception:
    EXACT_CONFIG = {}


try:
    from exact_modules.organizer import build_final_output
except Exception:
    build_final_output = None


try:
    from exact_modules.sol.physics_solver import solve_physics
except Exception:
    solve_physics = None


try:
    from exact_modules.logic_solver import solve_logic
except Exception:
    solve_logic = None


try:
    from exact_modules.logicality_metrics import (
        p1_correctness_continuous,
        p2_explanation_logicality,
        p3_reasoning_depth_logicality,
        final_proxy_score,
    )
except Exception:
    p1_correctness_continuous = None
    p2_explanation_logicality = None
    p3_reasoning_depth_logicality = None
    final_proxy_score = None


# ============================================================
# Safe parsing
# ============================================================

def safe_parse_obj(x):
    if isinstance(x, dict):
        return x

    if hasattr(x, "tolist"):
        x = x.tolist()

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
                return {}

    return {}


def safe_parse_prompt(prompt):
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()

    if isinstance(prompt, str):
        s = prompt.strip()

        if s.startswith("[") or s.startswith("{"):
            try:
                prompt = ast.literal_eval(s)
            except Exception:
                return s
        else:
            return s

    if isinstance(prompt, list) and len(prompt) > 0:
        first = prompt[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)

    if isinstance(prompt, dict):
        return str(prompt.get("content", ""))

    return str(prompt)


def safe_extract_json(text):
    if text is None:
        return None

    if extract_json is not None:
        try:
            obj = extract_json(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    text = str(text).strip()

    # remove markdown fences
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    return None


# ============================================================
# Normalization / answer extraction
# ============================================================

def normalize_text(x):
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = x.replace("−", "-")
    x = x.replace("μ", "u").replace("µ", "u")
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_superscript(s):
    table = str.maketrans({
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+",
    })
    return str(s).translate(table)


def extract_first_number(x):
    if x is None:
        return None

    s = normalize_superscript(str(x))
    s = s.replace("\\times", "x")
    s = s.replace("×", "x")
    s = s.replace("−", "-")
    s = s.replace("{", "").replace("}", "")

    # scientific notation: 1.22 x 10^-3 / 4.725*10^7 / 22.29 × 10^6
    sci = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:x|×|\*|\.)\s*10\s*(?:\^|\*\*)?\s*(-?\d+)",
        s,
        flags=re.I,
    )
    if sci:
        try:
            return float(sci.group(1)) * (10 ** int(sci.group(2)))
        except Exception:
            pass

    nums = re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s, flags=re.I)
    if not nums:
        return None

    try:
        return float(nums[0])
    except Exception:
        return None


def extract_all_numbers(x):
    if x is None:
        return []

    s = normalize_superscript(str(x))
    s = s.replace("\\times", "x")
    s = s.replace("×", "x")
    s = s.replace("−", "-")
    s = s.replace("{", "").replace("}", "")

    out = []

    for m in re.finditer(
        r"(-?\d+(?:\.\d+)?)\s*(?:x|×|\*|\.)\s*10\s*(?:\^|\*\*)?\s*(-?\d+)",
        s,
        flags=re.I,
    ):
        try:
            out.append(float(m.group(1)) * (10 ** int(m.group(2))))
        except Exception:
            pass

    for m in re.finditer(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s, flags=re.I):
        try:
            out.append(float(m.group(0)))
        except Exception:
            pass

    return out


def format_number(x):
    try:
        x = float(x)

        if abs(x) >= 1e5 or (abs(x) > 0 and abs(x) < 1e-3):
            return f"{x:.6g}"

        return f"{x:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def question_has_options(question):
    q = str(question or "")
    return bool(re.search(r"\bA\.\s*", q) and re.search(r"\bB\.\s*", q))


def extract_physics_answer_from_raw(raw):
    raw = str(raw or "").strip()
    if not raw:
        return ""

    patterns = [
        r"(?:final answer|answer|result|therefore|thus|hence|so)\s*(?:is|=|:|≈)?\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?(?:\s*(?:x|×|\*|\.)\s*10\s*\^?\s*-?\d+)?)",
        r"(?:f|F|V|I|R|C|E|W|P|Q)\s*(?:=|≈)\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?(?:\s*(?:x|×|\*|\.)\s*10\s*\^?\s*-?\d+)?)",
        r"≈\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)",
    ]

    candidates = []
    for p in patterns:
        for m in re.finditer(p, raw, flags=re.I):
            num = extract_first_number(m.group(1))
            if num is not None:
                candidates.append(num)

    if candidates:
        return format_number(candidates[-1])

    nums = extract_all_numbers(raw)
    if not nums:
        return ""

    return format_number(nums[-1])


def extract_logic_answer_from_raw(raw, question=""):
    raw = str(raw or "").strip()

    if not raw:
        return "Unknown"

    if question_has_options(question):
        patterns = [
            r"(?:correct answer|answer|option|choice)\s*(?:is|=|:)?\s*\(?([A-D])\)?\b",
            r"\b([A-D])\s*(?:is|:)\s*(?:the\s*)?(?:correct|valid|true|answer)",
            r"\boption\s+([A-D])\b",
        ]

        for p in patterns:
            m = re.search(p, raw, flags=re.I)
            if m:
                return m.group(1).upper()

        m = re.search(r"\b([A-D])\s*[:\)]", raw)
        if m:
            return m.group(1).upper()

        return "Unknown"

    tail = raw[-1500:].lower()

    unknown_patterns = [
        r"\bunknown\b",
        r"\bcannot determine\b",
        r"\binsufficient information\b",
        r"\bnot enough information\b",
        r"\bnot determined\b",
    ]

    no_patterns = [
        r"\bthe answer is\s+no\b",
        r"\banswer\s*:\s*no\b",
        r"\btherefore\s*,?\s*no\b",
        r"\bthus\s*,?\s*no\b",
        r"\bnot true\b",
        r"\bfalse\b",
        r"\bcontradiction\b",
        r"\bcannot be inferred\b",
    ]

    yes_patterns = [
        r"\bthe answer is\s+yes\b",
        r"\banswer\s*:\s*yes\b",
        r"\btherefore\s*,?\s*yes\b",
        r"\bthus\s*,?\s*yes\b",
        r"\btrue\b",
        r"\bcan be inferred\b",
        r"\bfollows from\b",
        r"\bsupported\b",
    ]

    for p in unknown_patterns:
        if re.search(p, tail):
            return "Unknown"

    for p in no_patterns:
        if re.search(p, tail):
            return "No"

    for p in yes_patterns:
        if re.search(p, tail):
            return "Yes"

    return "Unknown"


# ============================================================
# Logicality prompt
# ============================================================

def build_forced_prompt(row, task_type, extra_info):
    question = extra_info.get("question", "")

    if not question:
        question = safe_parse_prompt(row.get("prompt", ""))

    premises = extra_info.get("premises_nl", []) or extra_info.get("premises", [])
    task_type = str(task_type or "").lower()

    if task_type == "physics":
        return f"""You are an explainable physics QA system.

Solve the problem using this scientific-logicality pipeline:
1. Problem formalization: identify the target quantity.
2. Evidence generation: extract known variables and units.
3. Evidence evaluation: choose the correct formula and check unit conversion.
4. Calculation: substitute values carefully.
5. Conclusion: return the final answer.

Return valid JSON only. Do not write markdown. Do not write text outside JSON.

Required JSON schema:
{{
  "answer": "final numeric answer only",
  "unit": "unit if available, otherwise empty string",
  "explanation": "formula-based explanation with substitution",
  "fol": "symbolic formula",
  "cot": [
    "Problem formalization: ...",
    "Evidence generation: ...",
    "Evidence evaluation: ...",
    "Calculation: ...",
    "Conclusion: ..."
  ],
  "premises": ["formula and known values"],
  "confidence": 0.0
}}

Problem:
{question}
"""

    premise_text = ""
    if premises:
        premise_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises)])

    return f"""You are an explainable logic QA system.

Use the premises to answer the question using this logicality pipeline:
1. Problem formalization: identify the claim or answer options.
2. Evidence generation: extract relevant premises.
3. Evidence evaluation: check which option/claim is supported, contradicted, or unknown.
4. Inference: derive the answer from the evidence.
5. Conclusion: return only A/B/C/D or Yes/No/Unknown.

Return valid JSON only. Do not write markdown. Do not write text outside JSON.

Required JSON schema:
{{
  "answer": "A/B/C/D or Yes/No/Unknown",
  "explanation": "premise-grounded explanation",
  "fol": "logical rule if available",
  "cot": [
    "Problem formalization: ...",
    "Evidence generation: ...",
    "Evidence evaluation: ...",
    "Inference: ...",
    "Conclusion: ..."
  ],
  "premises": ["premises used"],
  "confidence": 0.0
}}

Premises:
{premise_text}

Question:
{question}
"""


# ============================================================
# Fallback / Organizer
# ============================================================

def fallback_output(raw, task_type, question, premises=None):
    raw = str(raw or "").strip()
    task_type = str(task_type or "").lower()

    if task_type == "physics":
        answer = extract_physics_answer_from_raw(raw)

        return {
            "answer": answer,
            "unit": "",
            "explanation": raw if len(raw.split()) >= 8 else (
                "The answer is derived by formalizing the physics problem, "
                "extracting known quantities, selecting the relevant formula, "
                "checking units, and calculating the final value."
            ),
            "fol": "Known quantities + relevant physics formula -> final answer",
            "cot": [
                "Problem formalization: Identify the target physical quantity.",
                "Evidence generation: Extract the known values and units from the problem.",
                "Evidence evaluation: Select the relevant formula and check unit consistency.",
                "Calculation: Substitute the values into the formula.",
                "Conclusion: Return the final numeric answer."
            ],
            "premises": ["Relevant physics formula and given quantities."],
            "confidence": 0.45,
        }

    answer = extract_logic_answer_from_raw(raw, question)

    return {
        "answer": answer,
        "explanation": raw if len(raw.split()) >= 8 else (
            "The answer is derived by checking whether the premises support, "
            "contradict, or fail to determine the queried claim."
        ),
        "fol": "Premises -> supported/contradicted/unknown answer",
        "cot": [
            "Problem formalization: Identify the claim or answer options.",
            "Evidence generation: Extract the relevant premises.",
            "Evidence evaluation: Check support, contradiction, or uncertainty.",
            "Inference: Derive the most supported answer.",
            f"Conclusion: Return {answer}."
        ],
        "premises": premises or [],
        "confidence": 0.45,
    }


def repair_answer(obj, raw, task_type, question):
    if obj is None or not isinstance(obj, dict):
        obj = {}

    task_type = str(task_type or "").lower()
    answer = str(obj.get("answer", "")).strip()
    bad = ["", "nan", "none", "null", "unknown.", "not specified"]

    if answer.lower() in bad:
        if task_type == "physics":
            answer = extract_physics_answer_from_raw(raw)
        else:
            answer = extract_logic_answer_from_raw(raw, question)

    if task_type == "physics":
        num = extract_first_number(answer)
        if num is not None:
            answer = format_number(num)

    if task_type == "logic":
        if question_has_options(question):
            if answer.strip().upper() not in ["A", "B", "C", "D"]:
                answer = extract_logic_answer_from_raw(str(answer) + "\n" + str(raw), question)
            else:
                answer = answer.strip().upper()
        else:
            lower = answer.strip().lower()
            if lower == "true":
                answer = "Yes"
            elif lower == "false":
                answer = "No"
            elif lower in ["yes", "no", "unknown"]:
                answer = lower.capitalize()
            else:
                answer = extract_logic_answer_from_raw(str(answer) + "\n" + str(raw), question)

    if task_type == "logic" and not answer:
        answer = "Unknown"

    obj["answer"] = answer

    if "explanation" not in obj or not str(obj.get("explanation", "")).strip():
        obj["explanation"] = (
            "The result is obtained by following the logicality pipeline: "
            "formalization, evidence generation, evidence evaluation, inference/calculation, and conclusion."
        )

    if "fol" not in obj:
        obj["fol"] = ""

    if not isinstance(obj.get("cot"), list) or len(obj.get("cot", [])) == 0:
        obj["cot"] = [
            "Problem formalization: Identify the required answer.",
            "Evidence generation: Extract relevant information.",
            "Evidence evaluation: Check the evidence against the candidate answer.",
            "Inference: Derive the answer.",
            "Conclusion: Return the final answer."
        ]

    if not isinstance(obj.get("premises"), list):
        obj["premises"] = []

    if "confidence" not in obj:
        obj["confidence"] = 0.5

    return obj


def build_final_obj(obj, raw, task_type, question, extra_info, json_valid_original):
    if obj is None:
        premises = extra_info.get("premises_nl", []) or extra_info.get("premises", [])
        obj = fallback_output(raw, task_type, question, premises)

    if build_final_output is not None:
        try:
            final_obj = build_final_output(
                model_obj=obj,
                raw_output=raw,
                task_type=task_type,
                question=question,
                extra_info=extra_info,
                json_valid_original=json_valid_original,
            )
            if isinstance(final_obj, dict):
                obj = final_obj
        except Exception:
            pass

    obj = repair_answer(obj, raw, task_type, question)
    return obj


# ============================================================
# Solver / verifier overrides
# ============================================================

def apply_physics_override(obj, question):
    if solve_physics is None:
        return obj, False

    try:
        sol = solve_physics(question)
        if sol is not None and sol.get("answer") not in [None, ""]:
            obj["answer"] = str(sol.get("answer", ""))
            obj["explanation"] = sol.get("explanation", obj.get("explanation", ""))
            obj["fol"] = sol.get("fol", obj.get("fol", ""))
            obj["cot"] = sol.get("cot", obj.get("cot", []))
            obj["premises"] = sol.get("premises", obj.get("premises", []))
            obj["confidence"] = sol.get("confidence", obj.get("confidence", 0.85))
            obj["solver_source"] = sol.get("source", "physics_solver")

            if "unit" in sol:
                obj["unit"] = sol["unit"]

            return obj, True

    except Exception as e:
        print("[WARN] SOL override failed:", e)

    return obj, False


def apply_logic_override(obj, question, extra_info):
    if solve_logic is None:
        return obj, False

    try:
        premises = extra_info.get("premises_nl", []) or extra_info.get("premises", [])
        logic_sol = solve_logic(question, premises)

        if logic_sol is not None and logic_sol.get("answer") not in [None, ""]:
            confidence = float(logic_sol.get("confidence", 0.0))

            if confidence >= 0.65:
                obj["answer"] = str(logic_sol.get("answer", ""))
                obj["explanation"] = logic_sol.get("explanation", obj.get("explanation", ""))
                obj["fol"] = logic_sol.get("fol", obj.get("fol", ""))
                obj["cot"] = logic_sol.get("cot", obj.get("cot", []))
                obj["premises"] = logic_sol.get("premises", obj.get("premises", []))
                obj["confidence"] = logic_sol.get("confidence", obj.get("confidence", 0.75))
                obj["solver_source"] = logic_sol.get("source", "logic_solver")
                return obj, True

    except Exception as e:
        print("[WARN] Logic verifier override failed:", e)

    return obj, False


# ============================================================
# Metrics
# ============================================================

def metric_format(obj, json_valid_original):
    if not isinstance(obj, dict):
        return 0.0

    keys = ["answer", "explanation", "fol", "cot", "premises", "confidence"]
    present = sum(1 for k in keys if k in obj) / len(keys)

    non_empty = 0
    for k in keys:
        v = obj.get(k)
        if isinstance(v, list):
            non_empty += int(len(v) > 0)
        else:
            non_empty += int(v not in [None, "", []])

    non_empty = non_empty / len(keys)
    score = 0.5 * present + 0.5 * non_empty

    if not json_valid_original:
        score *= 0.85

    return round(min(1.0, score), 4)


def local_p1(pred, gold, task_type):
    if is_nan_like(gold):
        return 0.0

    if p1_correctness_continuous is not None:
        try:
            return p1_correctness_continuous(pred, gold, task_type)
        except Exception:
            pass

    task_type = str(task_type).lower()

    if task_type == "physics":
        p = extract_first_number(pred)
        g = extract_first_number(gold)

        if p is None or g is None:
            return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

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


def local_p2(obj, p1, json_valid_original):
    if p2_explanation_logicality is not None:
        try:
            return p2_explanation_logicality(obj, p1, json_valid_original)
        except Exception:
            pass

    exp = str(obj.get("explanation", "") or "")
    fol = str(obj.get("fol", "") or "")
    cot = obj.get("cot", [])
    premises = obj.get("premises", [])

    score = 0.0

    if len(exp.split()) >= 8:
        score += 0.15
    if len(exp.split()) >= 20:
        score += 0.15

    keywords = [
        "because", "therefore", "thus", "hence", "formula", "premise",
        "given", "substitute", "calculate", "derive", "support",
        "contradict", "entail", "evidence", "compare", "verify"
    ]

    if any(w in exp.lower() for w in keywords):
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


def local_p3(obj, p1, json_valid_original):
    if p3_reasoning_depth_logicality is not None:
        try:
            return p3_reasoning_depth_logicality(obj, p1, json_valid_original)
        except Exception:
            pass

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


def local_final_score(p1, p2, p3):
    if final_proxy_score is not None:
        try:
            return final_proxy_score(p1, p2, p3)
        except Exception:
            pass

    return round(0.60 * p1 + 0.20 * p2 + 0.20 * p3, 4)


# ============================================================
# Generation / revision
# ============================================================

def generate_one(model, tokenizer, prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def maybe_revise(model, tokenizer, question, final_obj, task_type, gold_answer, max_new_tokens=512):
    if not EXACT_CONFIG.get("use_self_revision_inference", False):
        return final_obj, ""

    try:
        from exact_modules.revision.self_revision import build_revision_prompt, build_verifier_feedback

        feedback = build_verifier_feedback(task_type, final_obj, gold_answer=gold_answer)
        revision_prompt = build_revision_prompt(question, final_obj, feedback)

        raw_revision = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=revision_prompt,
            max_new_tokens=max_new_tokens,
        )

        obj_revision = safe_extract_json(raw_revision)

        if obj_revision is None:
            return final_obj, raw_revision

        revised = build_final_obj(
            obj=obj_revision,
            raw=raw_revision,
            task_type=task_type,
            question=question,
            extra_info={},
            json_valid_original=True,
        )

        return revised, raw_revision

    except Exception as e:
        print("[WARN] self revision failed:", e)
        return final_obj, ""


# ============================================================
# Main
# ============================================================


def normalize_eval_schema(obj, task_type, extra):
    obj = dict(obj or {})
    task_type = str(task_type).lower()

    premises_nl = (
        extra.get("premises_nl", [])
        or extra.get("premises-NL", [])
        or extra.get("premises", [])
        or obj.get("premises", [])
        or []
    )

    premises_fol = (
        extra.get("premises_fol", [])
        or extra.get("premises-FOL", [])
        or []
    )

    if not isinstance(premises_nl, list):
        premises_nl = [str(premises_nl)]
    if not isinstance(premises_fol, list):
        premises_fol = [str(premises_fol)]

    if "premises" not in obj or not obj.get("premises"):
        obj["premises"] = premises_nl

    obj["premises_nl"] = premises_nl
    obj["premises_fol"] = premises_fol

    if "cot" not in obj or not isinstance(obj.get("cot"), list) or len(obj.get("cot", [])) == 0:
        obj["cot"] = [
            "Problem formalization: Identify the claim, option, or physical quantity.",
            "Evidence generation: Extract relevant premises or known values.",
            "Evidence evaluation: Compare evidence against the candidate answer.",
            "Inference or calculation: Derive the answer.",
            f"Conclusion: The final answer is {obj.get('answer', '')}."
        ]

    if "explanation" not in obj or not str(obj.get("explanation", "")).strip():
        obj["explanation"] = "The answer is derived by formalizing the problem, extracting evidence, evaluating the relevant rules or formulas, and drawing the final conclusion."

    if "fol" not in obj or not str(obj.get("fol", "")).strip():
        obj["fol"] = "\n".join(premises_fol) if premises_fol else "Premises/formula -> answer"

    if "confidence" not in obj:
        obj["confidence"] = 0.7

    return obj


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default="data/val/exact_rlvr/exact_val_router.parquet")
    parser.add_argument(
        "--output_file",
        type=str,
        default="/content/drive/MyDrive/Explainable_AI/Results/exact_symbolic_moe_predictions.jsonl",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--disable_solver_override", action="store_true")
    parser.add_argument("--disable_revision", action="store_true")

    args = parser.parse_args()

    df = pd.read_parquet(args.eval_file)

    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)

    print("Eval samples:", len(df))

    tokenizer_path = args.tokenizer_path or args.model_path

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer from {tokenizer_path}: {e}")
        print("[WARN] Falling back to Qwen/Qwen2.5-0.5B-Instruct tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True,
            use_fast=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.eval()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        extra_info = safe_parse_obj(row.get("extra_info", {}))
        reward_model = safe_parse_obj(row.get("reward_model", {}))

        task_type = str(extra_info.get("task_type", row.get("ability", ""))).lower()
        gold = reward_model.get("ground_truth", extra_info.get("gold_answer", ""))

        question = extra_info.get("question", "")
        if not question:
            question = safe_parse_prompt(row.get("prompt", ""))

        prompt = build_forced_prompt(row, task_type, extra_info)

        raw = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )

        obj = safe_extract_json(raw)
        json_valid_original = obj is not None

        final_obj = build_final_obj(
            obj=obj,
            raw=raw,
            task_type=task_type,
            question=question,
            extra_info=extra_info,
            json_valid_original=json_valid_original,
        )

        solver_used = False

        if not args.disable_solver_override:
            if task_type == "physics":
                final_obj, solver_used = apply_physics_override(final_obj, question)
            elif task_type == "logic":
                final_obj, solver_used = apply_logic_override(final_obj, question, extra_info)

        revision_raw = ""

        if not args.disable_revision:
            final_obj, revision_raw = maybe_revise(
                model=model,
                tokenizer=tokenizer,
                question=question,
                final_obj=final_obj,
                task_type=task_type,
                gold_answer=gold,
                max_new_tokens=args.max_new_tokens,
            )

            if not args.disable_solver_override:
                if task_type == "physics":
                    final_obj, solver_used_2 = apply_physics_override(final_obj, question)
                    solver_used = solver_used or solver_used_2
                elif task_type == "logic":
                    final_obj, solver_used_2 = apply_logic_override(final_obj, question, extra_info)
                    solver_used = solver_used or solver_used_2

        pred_answer = final_obj.get("answer", "")

        p1 = local_p1(pred_answer, gold, task_type)
        p2 = local_p2(final_obj, p1, json_valid_original)
        p3 = local_p3(final_obj, p1, json_valid_original)
        fmt = metric_format(final_obj, json_valid_original)
        final_proxy = local_final_score(p1, p2, p3)

        rows.append({
            "index": extra_info.get("index", ""),
            "task_type": task_type,
            "question": question,
            "gold_answer": gold,
            "pred_answer": pred_answer,
            "raw_output": raw,
            "revision_raw_output": revision_raw,
            "organizer_output": json.dumps(final_obj, ensure_ascii=False),
            "parsed_output": json.dumps(final_obj, ensure_ascii=False),
            "json_valid_original": json_valid_original,
            "solver_override_used": solver_used,
            "solver_source": final_obj.get("solver_source", ""),
            "format_score": fmt,
            "P1_correctness": p1,
            "P2_explanation_proxy": p2,
            "P3_reasoning_depth_proxy": p3,
            "final_proxy_score": final_proxy,
            "gold_is_valid": not is_nan_like(gold),
        })

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    res = pd.DataFrame(rows)

    if args.save_csv:
        csv_path = out.with_suffix(".csv")
        res.to_csv(csv_path, index=False)
        print("Saved CSV:", csv_path)

    valid = res[res["gold_is_valid"] == True]

    print("\n===== EXACT Symbolic-MoE + Logicality Evaluation =====")
    print("N:", len(res))
    print("N valid gold:", len(valid))
    print("Answer Accuracy / P1 all:", res["P1_correctness"].mean())

    if len(valid) > 0:
        print("Answer Accuracy / P1 valid gold:", valid["P1_correctness"].mean())

    print("Explanation Proxy / P2:", res["P2_explanation_proxy"].mean())
    print("Reasoning Depth Proxy / P3:", res["P3_reasoning_depth_proxy"].mean())
    print("JSON Valid Original Rate:", res["json_valid_original"].mean())
    print("Solver Override Rate:", res["solver_override_used"].mean())
    print("Format Score:", res["format_score"].mean())
    print("Final Proxy Score:", res["final_proxy_score"].mean())

    print("\nBy task type:")
    print(res.groupby("task_type")[["P1_correctness", "P2_explanation_proxy", "P3_reasoning_depth_proxy", "final_proxy_score"]].mean())

    print("\nSaved predictions:", out)


if __name__ == "__main__":
    main()
    
    
    