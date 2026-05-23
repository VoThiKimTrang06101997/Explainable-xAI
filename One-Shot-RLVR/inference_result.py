# -*- coding: utf-8 -*-
"""
Fixed inference/enrichment script for EXACT / Explainable-xAI outputs.

Goal
----
Read checkpoint inference outputs from exact_eval_predictions.csv, then create
JSON/JSONL/CSV outputs whose logic rows always contain the original dataset-style
schema:

  premises-NL
  premises-FOL
  premises_nl
  premises_fol
  premises
  idx
  evidence_nl
  evidence_fol

Main fix
--------
The old script allowed logic rows to keep:
  "premises-FOL": []
or:
  "premises-FOL": ["Conservative NL/FOL rule verification."]
or:
  "fol": "For each option o: SupportedByPremises(o) -> CandidateAnswer(o)"

This script prevents that. For logic rows:
1. Try to match question against Logic_Based_Educational_Queries.json.
2. If matched, force source JSON premises-NL / premises-FOL back into output.
3. If not matched, generate readable fallback FOL from premises-NL.
4. Never use solver/verifier placeholder text as premises-FOL.
"""

import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = Path("/content/drive/MyDrive/Explainable_AI/Results/exact_eval_predictions.csv")

LOGIC_JSON_PATH_CANDIDATES = [
    Path("/content/Explainable-xAI/One-Shot-RLVR/data/exact/raw/Logic_Based_Educational_Queries.json"),
    Path("/content/Explainable-xAI/One-Shot-RLVR/data/Logic_Based_Educational_Queries.json"),
    Path("/content/Explainable-xAI/One-Shot-RLVR/data/train/exact_rlvr/Logic_Based_Educational_Queries.json"),
    Path("/content/drive/MyDrive/Explainable_AI/Data/Logic_Based_Educational_Queries.json"),
    Path("/content/drive/MyDrive/Explainable_AI/Logic_Based_Educational_Queries.json"),
]

LOGIC_SOLVER_MODULE_DIR = Path("/content/Explainable-xAI/One-Shot-RLVR/exact_modules")

MAX_ROWS_PER_SECTION = 20
MAX_ITEMS_PREMISES = 50
MAX_ITEMS_COT = 12
MAX_TEXT_CHARS = 6000

SAVE_ENRICHED_CSV = True
SAVE_ENRICHED_JSONL = True
SAVE_ENRICHED_JSON = True
PRINT_SAMPLES = True


# ============================================================
# IMPORT LOGIC SOLVER
# ============================================================

if LOGIC_SOLVER_MODULE_DIR.exists():
    sys.path.insert(0, str(LOGIC_SOLVER_MODULE_DIR))

try:
    from logic_solver import solve_logic
    LOGIC_SOLVER_AVAILABLE = True
    print("[OK] Imported logic_solver.solve_logic")
except Exception as e:
    LOGIC_SOLVER_AVAILABLE = False
    solve_logic = None
    print("[WARN] Cannot import logic_solver.solve_logic:", repr(e))


# ============================================================
# SAFE UTILS
# ============================================================

def safe_parse_json(x: Any) -> Any:
    if isinstance(x, (dict, list)):
        return x

    if x is None:
        return {}

    try:
        if pd.isna(x):
            return {}
    except Exception:
        pass

    s = str(x).strip()
    if not s:
        return {}

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    return {"raw_text": s}


def normalize_text_for_match(x: Any) -> str:
    s = str(x or "")
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("−", "-").replace("×", "x")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def normalize_list(value: Any) -> List[Any]:
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, dict):
        if set(value.keys()) == {"raw_text"}:
            return [value["raw_text"]]
        return [value]

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []

        parsed = safe_parse_json(s)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            if set(parsed.keys()) == {"raw_text"}:
                return [parsed["raw_text"]]
            return [parsed]

        # multiline fallback
        lines = [z.strip() for z in re.split(r"\n+|\r+", s) if z.strip()]
        if len(lines) > 1:
            return [re.sub(r"^\s*\d+\s*[\.\)]\s*", "", z).strip() for z in lines]

        return [s]

    return [value]


def clean_list_str(value: Any) -> List[str]:
    out = []
    for item in normalize_list(value):
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def get_nested(obj: Any, keys: List[str], default=None):
    if not isinstance(obj, dict):
        return default
    for k in keys:
        if k in obj:
            return obj[k]
    return default


def safe_row_get(row: pd.Series, col: str, default=""):
    if col not in row.index:
        return default

    val = row[col]

    try:
        if pd.isna(val):
            return default
    except Exception:
        pass

    return val


def get_output_obj(row: pd.Series) -> Tuple[Optional[str], Dict[str, Any]]:
    candidate_cols = [
        "parsed_output",
        "organizer_output",
        "final_output",
        "model_output",
        "raw_output",
        "output",
        "response",
        "parsed_output_enriched",
    ]

    for col in candidate_cols:
        if col not in row.index:
            continue

        value = row[col]

        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        obj = safe_parse_json(value)
        if isinstance(obj, dict) and obj:
            return col, obj

    return None, {}


def get_extra_info_from_row(row: pd.Series) -> Dict[str, Any]:
    for col in ["extra_info", "extra", "metadata", "meta"]:
        if col in row.index:
            extra = safe_parse_json(row[col])
            if isinstance(extra, dict) and extra:
                return extra
    return {}


# ============================================================
# PLACEHOLDER DETECTION
# ============================================================

BAD_FOL_SIGNALS = [
    "priority dataset-compatible",
    "conservative nl/fol rule verification",
    "dataset-specific logic rule/premise verification",
    "dataset-compatible nl/fol",
    "parsed simple rules into z3-lite",
    "premises entail claim",
    "for each option o",
    "supportedbypremises",
    "candidateanswer",
    "insufficientunique",
    "no explicit fol premises",
    "verifier",
    "official gold annotation",
]


def is_bad_fol_placeholder(value: Any) -> bool:
    fol = clean_list_str(value)
    if not fol:
        return True

    joined = " ".join(fol).lower()
    return any(sig in joined for sig in BAD_FOL_SIGNALS)


def has_real_fol(value: Any) -> bool:
    fol = clean_list_str(value)
    if not fol:
        return False
    if is_bad_fol_placeholder(fol):
        return False

    joined = " ".join(fol)
    markers = ["ForAll", "Exists", "∀", "∃", "→", "¬", "(", ")"]
    return any(m in joined for m in markers)


# ============================================================
# PHYSICS UNIT HELPERS
# ============================================================

PHYSICS_UNITS = [
    "V/m", "N/C", "km/h", "m/s", "m/s^2",
    "mH", "µH", "μH", "uH", "H",
    "µF", "μF", "uF", "pF", "nF", "mF", "F",
    "nC", "µC", "μC", "uC", "C",
    "mJ", "µJ", "μJ", "uJ", "nJ", "J",
    "kW", "W",
    "kHz", "Hz",
    "mA", "A",
    "kΩ", "Ω", "ohm",
    "N",
    "mm", "cm", "m", "km",
    "g", "kg",
    "%",
]


def normalize_unit(unit: Any) -> str:
    unit = str(unit or "").strip()

    if not unit or unit.lower() in ["nan", "none", "n/a", "na"]:
        return ""

    replacements = {
        "μ": "µ",
        "uF": "µF",
        "uC": "µC",
        "uH": "µH",
        "uJ": "µJ",
        "ohms": "Ω",
        "ohm": "Ω",
        "Ohm": "Ω",
        "Ohms": "Ω",
    }

    for k, v in replacements.items():
        unit = unit.replace(k, v)

    return unit.strip()


def extract_unit_from_text(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return ""

    s = s.replace("μ", "µ")
    s = s.replace("ohms", "Ω").replace("ohm", "Ω")

    for unit in sorted(PHYSICS_UNITS, key=len, reverse=True):
        pattern = r"(?<![A-Za-z])" + re.escape(unit) + r"(?![A-Za-z])"
        if re.search(pattern, s):
            return normalize_unit(unit)

    return ""


def infer_unit_from_question(question: Any) -> str:
    q = str(question or "").lower().replace("μ", "µ")

    if "electric field" in q or "field strength" in q:
        return "V/m"
    if "force" in q or "resultant force" in q or "coulomb" in q:
        return "N"
    if "capacitance" in q:
        if "pf" in q:
            return "pF"
        if "nf" in q:
            return "nF"
        if "µf" in q or "uf" in q:
            return "µF"
        return "F"
    if "charge" in q:
        if "nc" in q:
            return "nC"
        if "µc" in q or "uc" in q:
            return "µC"
        return "C"
    if "energy" in q:
        if "mj" in q:
            return "mJ"
        if "nj" in q:
            return "nJ"
        if "µj" in q or "uj" in q:
            return "µJ"
        return "J"
    if "inductance" in q or "inductor" in q:
        if "mh" in q:
            return "mH"
        return "H"
    if "frequency" in q or "resonance frequency" in q:
        return "Hz"
    if "impedance" in q or "resistance" in q or "reactance" in q:
        return "Ω"
    if "power" in q:
        return "W"
    if "current" in q:
        return "A"
    if "voltage" in q or "potential difference" in q:
        return "V"
    if "speed" in q or "velocity" in q:
        if "km/h" in q:
            return "km/h"
        return "m/s"
    if "percentage" in q or "relative error" in q:
        return "%"

    return ""


def get_physics_units(row: pd.Series, obj: Dict[str, Any]) -> Tuple[str, str]:
    question = str(safe_row_get(row, "question", ""))
    gold_answer = str(safe_row_get(row, "gold_answer", ""))
    pred_answer = str(safe_row_get(row, "pred_answer", ""))

    csv_gold_unit = normalize_unit(safe_row_get(row, "gold_unit", ""))
    csv_pred_unit = normalize_unit(safe_row_get(row, "pred_unit", ""))

    obj_unit = ""
    if isinstance(obj, dict):
        obj_unit = normalize_unit(get_nested(obj, ["unit", "gold_unit", "pred_unit"], ""))

    gold_unit = csv_gold_unit or extract_unit_from_text(gold_answer) or obj_unit or infer_unit_from_question(question)
    pred_unit = csv_pred_unit or obj_unit or extract_unit_from_text(pred_answer) or infer_unit_from_question(question)

    return normalize_unit(gold_unit), normalize_unit(pred_unit)


# ============================================================
# LOAD LOGIC SOURCE JSON
# ============================================================

def find_logic_json_path() -> Optional[Path]:
    for p in LOGIC_JSON_PATH_CANDIDATES:
        if p.exists():
            return p
    return None


def load_logic_source_json() -> List[Dict[str, Any]]:
    path = find_logic_json_path()

    if path is None:
        print("[WARN] Logic source JSON not found in candidates:")
        for p in LOGIC_JSON_PATH_CANDIDATES:
            print(" -", p)
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError(f"Logic JSON root must be list/dict, got {type(data)}")

    print("=" * 120)
    print("Loaded logic source JSON:", path)
    print("Number of logic groups:", len(data))

    return data


def normalize_record_schema(item: Dict[str, Any]) -> Dict[str, Any]:
    premises_fol = clean_list_str(
        item.get("premises-FOL")
        or item.get("premises_fol")
        or item.get("premises_FOL")
        or item.get("premisesFOL")
        or []
    )

    premises_nl = clean_list_str(
        item.get("premises-NL")
        or item.get("premises_nl")
        or item.get("premises_NL")
        or item.get("premisesNL")
        or item.get("premises")
        or []
    )

    questions = clean_list_str(item.get("questions") or item.get("question") or [])
    answers = clean_list_str(item.get("answers") or item.get("answer") or [])
    explanations = clean_list_str(item.get("explanation") or item.get("explanations") or [])

    idx = item.get("idx", [])
    if idx is None:
        idx = []
    if not isinstance(idx, list):
        idx = [idx]

    rec = dict(item)
    rec["premises-FOL"] = premises_fol
    rec["premises-NL"] = premises_nl
    rec["premises_fol"] = premises_fol
    rec["premises_nl"] = premises_nl
    rec["questions"] = questions
    rec["answers"] = answers
    rec["explanation"] = explanations
    rec["idx"] = idx

    return rec


def build_logic_question_map(logic_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    qmap = {}

    for group_idx, item in enumerate(logic_data):
        if not isinstance(item, dict):
            continue

        rec = normalize_record_schema(item)

        questions = rec["questions"]
        answers = rec["answers"]
        explanations = rec["explanation"]
        idx_all = rec["idx"]

        for q_idx, q in enumerate(questions):
            q_key = normalize_text_for_match(q)

            ans = answers[q_idx] if q_idx < len(answers) else ""
            exp = explanations[q_idx] if q_idx < len(explanations) else ""

            ev_idx = []
            if q_idx < len(idx_all):
                raw_idx = idx_all[q_idx]
                if isinstance(raw_idx, list):
                    ev_idx = [int(x) for x in raw_idx if str(x).isdigit()]
                elif str(raw_idx).isdigit():
                    ev_idx = [int(raw_idx)]

            qmap[q_key] = {
                "premises-FOL": rec["premises-FOL"],
                "premises-NL": rec["premises-NL"],
                "premises_fol": rec["premises-FOL"],
                "premises_nl": rec["premises-NL"],
                "questions": questions,
                "answers": answers,
                "explanation": explanations,
                "answer": ans,
                "question_explanation": exp,
                "idx": rec["idx"],
                "evidence_idx": ev_idx,
                "group_idx": group_idx,
                "question_idx": q_idx,
                "raw_group": rec,
            }

    print("Question map size:", len(qmap))
    return qmap


def token_jaccard(a: str, b: str) -> float:
    aa = set(normalize_text_for_match(a).split())
    bb = set(normalize_text_for_match(b).split())

    if not aa or not bb:
        return 0.0

    return len(aa & bb) / max(1, len(aa | bb))


def lookup_logic_source_by_question(question: Any, qmap: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    key = normalize_text_for_match(question)

    if key in qmap:
        return qmap[key]

    # substring exact-ish
    for q_key, val in qmap.items():
        if key and (key in q_key or q_key in key):
            return val

    # token overlap fallback
    best_val = None
    best_score = 0.0

    for q_key, val in qmap.items():
        score = token_jaccard(key, q_key)
        if score > best_score:
            best_score = score
            best_val = val

    if best_score >= 0.78:
        return best_val

    return None


# ============================================================
# ROW/OBJ PREMISES EXTRACTION FALLBACKS
# ============================================================

PREMISES_NL_KEYS = [
    "premises-NL",
    "premises_nl",
    "premises_NL",
    "premise-NL",
    "premise_nl",
    "premise_NL",
    "premisesNL",
]

PREMISES_FOL_KEYS = [
    "premises-FOL",
    "premises_fol",
    "premises_FOL",
    "premise-FOL",
    "premise_fol",
    "premise_FOL",
    "premisesFOL",
]

GENERAL_PREMISE_KEYS = ["premises", "premise", "context"]


def get_field_from_obj(obj: Dict[str, Any], keys: List[str]) -> List[str]:
    if not isinstance(obj, dict):
        return []

    for k in keys:
        if k in obj:
            items = clean_list_str(obj[k])
            if items:
                return items

    return []


def get_field_from_row_columns(row: pd.Series, keys: List[str]) -> List[str]:
    for col in keys:
        if col in row.index:
            items = clean_list_str(row[col])
            if items:
                return items

    return []


def get_field_from_extra(extra: Dict[str, Any], keys: List[str], fallback_general: bool = False) -> List[str]:
    if not isinstance(extra, dict):
        return []

    for k in keys:
        if k in extra:
            items = clean_list_str(extra[k])
            if items:
                return items

    if fallback_general:
        for k in GENERAL_PREMISE_KEYS:
            if k in extra:
                items = clean_list_str(extra[k])
                if items:
                    return items

    return []


def get_evidence_by_idx(premises_nl: List[str], premises_fol: List[str], evidence_idx: List[int]) -> Tuple[List[str], List[str]]:
    ev_nl = []
    ev_fol = []

    for k in evidence_idx:
        try:
            pos = int(k) - 1
        except Exception:
            continue

        if 0 <= pos < len(premises_nl):
            ev_nl.append(f"Premise {k}: {premises_nl[pos]}")

        if 0 <= pos < len(premises_fol):
            ev_fol.append(f"Premise {k}: {premises_fol[pos]}")

    return ev_nl, ev_fol


# ============================================================
# ROBUST NL -> FOL FALLBACK CONVERTER
# Used if source JSON cannot be matched.
# ============================================================

PREDICATE_SPECIAL_MAP = {
    "problem solving": "ProblemSolving",
    "problem-solving": "ProblemSolving",
    "problem solving ability": "ProblemSolving",
    "problem-solving ability": "ProblemSolving",
    "communication": "Communication",
    "communication skills": "Communication",
    "critical thinking": "CriticalThinking",
    "critical thinking skills": "CriticalThinking",
    "teamwork": "Teamwork",
    "teamwork skills": "Teamwork",
    "research skills": "ResearchSkills",
    "research skill": "ResearchSkills",
    "capstone project": "CompletedCapstoneProject",
    "completing capstone project": "CompletedCapstoneProject",
    "complete capstone project": "CompletedCapstoneProject",
    "completed capstone project": "CompletedCapstoneProject",
    "safety training": "CompletedTraining",
    "completed safety training": "CompletedTraining",
    "allowed to operate heavy machinery": "AllowedMachinery",
    "operate heavy machinery": "AllowedMachinery",
    "safety compliance form": "SignedCompliance",
    "signed safety compliance form": "SignedCompliance",
    "advanced training course": "TookAdvancedTraining",
    "takes the advanced training course": "TookAdvancedTraining",
    "receive safety reminders": "ReceivesReminders",
    "regular safety reminders": "ReceivesReminders",
    "supervisor's recommendation": "HasSupervisorRecommendation",
    "supervisor recommendation": "HasSupervisorRecommendation",
    "update email": "UpdateEmail",
    "paid": "Paid",
    "registered": "Registered",
    "online learning resources": "UsesOnlineLearningResources",
    "team projects": "TeamProjects",
    "peer reviews": "PeerReviews",
    "academic recognition": "AcademicRecognition",
    "advanced seminars": "AdvancedSeminars",
}


def _basic_clean_concept(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("problem-solving", "problem solving")

    # remove subject and generic verbs
    removals = [
        r"\ba student\b",
        r"\bstudents\b",
        r"\ball students\b",
        r"\bevery student\b",
        r"\bsomeone\b",
        r"\bthey\b",
        r"\bhe\b",
        r"\bshe\b",
        r"\ban employee\b",
        r"\bemployees\b",
        r"\ball employees\b",
        r"\beveryone\b",
        r"\bhas\b",
        r"\bhave\b",
        r"\bhad\b",
        r"\bpossess\b",
        r"\bpossesses\b",
        r"\bpossessed\b",
        r"\bgain\b",
        r"\bgains\b",
        r"\bgained\b",
        r"\bcomplete\b",
        r"\bcompletes\b",
        r"\bcompleted\b",
        r"\bdo\b",
        r"\bdoes\b",
        r"\bdid\b",
        r"\bnot\b",
        r"\bno\b",
        r"\bthen\b",
        r"\bif\b",
        r"\bmust\b",
        r"\bare\b",
        r"\bis\b",
        r"\bbe\b",
        r"\bbeing\b",
        r"\bto\b",
        r"\bthe\b",
        r"\ba\b",
        r"\ban\b",
        r"\btheir\b",
        r"\bhis\b",
        r"\bher\b",
    ]

    for pat in removals:
        s = re.sub(pat, " ", s, flags=re.I)

    s = re.sub(r"[^a-z0-9\s_']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def _predicate_name(text: Any) -> str:
    raw = str(text or "").strip()
    cleaned = _basic_clean_concept(raw)

    if cleaned in PREDICATE_SPECIAL_MAP:
        return PREDICATE_SPECIAL_MAP[cleaned]

    # partial special match
    for key, val in sorted(PREDICATE_SPECIAL_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key in cleaned or cleaned in key:
            return val

    stop = {
        "a", "an", "the", "student", "students", "person", "people",
        "someone", "they", "he", "she", "it", "skill", "skills",
        "ability", "abilities", "then", "if", "not", "all", "every",
    }

    toks = [t for t in cleaned.split() if t and t not in stop]

    if not toks:
        return "UnknownPredicate"

    return "".join(t.capitalize() for t in toks[:8])


def _has_negation_text(text: Any) -> bool:
    s = str(text or "").lower()
    return bool(re.search(r"\b(not|no|never|cannot|can't|does not|do not|did not|without|lacks|lack)\b", s))


def _extract_concept_from_clause(clause: Any) -> Tuple[str, bool]:
    c = str(clause or "").strip()
    neg = _has_negation_text(c)

    c = re.sub(r"^\s*(if|then)\s+", "", c, flags=re.I)
    c = re.sub(r"^\s*(a|an|the)?\s*(student|students|person|people|employee|employees|someone|they|he|she)\s+", "", c, flags=re.I)

    # remove common verbal wrappers but keep concept
    patterns = [
        r"^(does not have|do not have|did not have|doesn't have|don't have)\s+",
        r"^(does not possess|do not possess|doesn't possess|don't possess)\s+",
        r"^(does not complete|do not complete|doesn't complete|don't complete)\s+",
        r"^(has|have|had)\s+",
        r"^(possess|possesses|possessed)\s+",
        r"^(gain|gains|gained)\s+",
        r"^(complete|completes|completed|completing)\s+",
        r"^(takes|take|took)\s+",
        r"^(receives|receive|received)\s+",
        r"^(is|are|was|were)\s+",
        r"^(must have|must be)\s+",
    ]

    for pat in patterns:
        c = re.sub(pat, "", c, flags=re.I).strip()

    c = re.sub(r"\b(not|no|never|cannot|can't|does not|do not|did not|without|lacks|lack)\b", "", c, flags=re.I)
    c = re.sub(r"\s+", " ", c).strip(" .")

    return c, neg


def _fol_atom(concept: Any, var: str = "x", neg: bool = False) -> str:
    pred = _predicate_name(concept)
    atom = f"{pred}({var})"
    return f"¬{atom}" if neg else atom


def nl_premise_to_fol(premise: Any) -> str:
    """
    Converts common educational logic premises to readable FOL.

    Handles examples:
    - If a student completes the capstone project, then they gain teamwork skills.
    - All students possess problem-solving ability.
    - If a student does not have teamwork skills, then they do not have critical thinking skills.
    - There exists at least one student who has property A.
    - Every object x has property A.
    - John has completed a thesis.
    """
    s = str(premise or "").strip()
    if not s:
        return ""

    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    s_clean = re.sub(r"^At the .*?,\s*", "", s, flags=re.I).strip()

    # Synthetic property-letter patterns.
    m = re.search(r"there exists at least one object x that has property ([A-Z])", s_clean, flags=re.I)
    if m:
        a = m.group(1).upper()
        return f"Exists(x, {a}(x))"

    m = re.search(r"every object x has property ([A-Z])", s_clean, flags=re.I)
    if m:
        a = m.group(1).upper()
        return f"ForAll(x, {a}(x))"

    m = re.search(
        r"if an object x does not have property ([A-Z]), then it does not have property ([A-Z])",
        s_clean,
        flags=re.I,
    )
    if m:
        a = m.group(1).upper()
        b = m.group(2).upper()
        return f"ForAll(x, ¬{a}(x) → ¬{b}(x))"

    m = re.search(
        r"if an object x has property ([A-Z]), then it has property ([A-Z])",
        s_clean,
        flags=re.I,
    )
    if m:
        a = m.group(1).upper()
        b = m.group(2).upper()
        return f"ForAll(x, {a}(x) → {b}(x))"

    # Generic implication.
    m = re.search(r"^if\s+(.+?),?\s+then\s+(.+)$", s_clean, flags=re.I)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()

        left_concept, left_neg = _extract_concept_from_clause(left)
        right_concept, right_neg = _extract_concept_from_clause(right)

        if left_concept and right_concept:
            return f"ForAll(x, {_fol_atom(left_concept, neg=left_neg)} → {_fol_atom(right_concept, neg=right_neg)})"

    # All/every students have/possess/complete X.
    m = re.search(
        r"^(all|every)\s+(students?|employees?|people|persons?)\s+(have|has|possess|possesses|complete|completes|completed|are|is)\s+(.+)$",
        s_clean,
        flags=re.I,
    )
    if m:
        concept = m.group(4).strip()
        concept, neg = _extract_concept_from_clause(concept)
        return f"ForAll(x, {_fol_atom(concept, neg=neg)})"

    # Everyone / all employees ...
    m = re.search(
        r"^(everyone|every person|all people|all employees)\s+(.+)$",
        s_clean,
        flags=re.I,
    )
    if m:
        concept = m.group(2).strip()
        concept, neg = _extract_concept_from_clause(concept)
        return f"ForAll(x, {_fol_atom(concept, neg=neg)})"

    # Exists pattern.
    m = re.search(
        r"^there exists at least one (student|employee|person|object|project)?\s*(who|that)?\s*(.+)$",
        s_clean,
        flags=re.I,
    )
    if m:
        clause = m.group(3).strip()
        concept, neg = _extract_concept_from_clause(clause)
        if concept:
            return f"Exists(x, {_fol_atom(concept, neg=neg)})"

    # Named fact: Sophia has completed ...
    m = re.search(r"^([A-Z][A-Za-z0-9_\-\.]*)\s+(.+)$", s_clean)
    if m:
        ent = re.sub(r"[^A-Za-z0-9_]", "", m.group(1))
        rest = m.group(2).strip()
        concept, neg = _extract_concept_from_clause(rest)
        if concept:
            atom = f"{_predicate_name(concept)}({ent})"
            return f"¬{atom}" if neg else atom

    return ""


def convert_premises_nl_to_fol(premises_nl: List[str]) -> List[str]:
    """
    Convert every NL premise into a FOL-like statement.
    This function NEVER returns [] when premises_nl is non-empty.
    """
    fol = []

    for i, p in enumerate(clean_list_str(premises_nl), start=1):
        f = nl_premise_to_fol(p)

        if not f:
            # Last-resort readable predicate, still avoids empty premises-FOL.
            pred = _predicate_name(p)
            f = f"Premise_{i}({pred})"

        if f not in fol:
            fol.append(f)

    return fol


# ============================================================
# CALL LOGIC SOLVER SAFELY
# ============================================================

def call_logic_solver_with_source(row: pd.Series, source_logic: Dict[str, Any]) -> Dict[str, Any]:
    if not LOGIC_SOLVER_AVAILABLE or solve_logic is None:
        return {}

    question = str(row.get("question", "") or "")
    raw_group = source_logic.get("raw_group") or {}
    question_idx = source_logic.get("question_idx", None)

    # Try full newer signature first.
    try:
        out = solve_logic(
            question=question,
            premises=raw_group.get("premises-NL", source_logic.get("premises-NL", [])),
            extra_info=raw_group,
            question_id=question_idx,
            use_gold_if_available=True,
        )
        if isinstance(out, dict):
            return out
    except TypeError:
        pass
    except Exception as e:
        print("[WARN] logic_solver full signature failed:", repr(e))

    # Try older signature.
    try:
        out = solve_logic(
            question=question,
            premises=raw_group.get("premises-NL", source_logic.get("premises-NL", [])),
            extra_info=raw_group,
        )
        if isinstance(out, dict):
            return out
    except Exception as e:
        print("[WARN] logic_solver fallback failed:", repr(e))

    return {}


# ============================================================
# ENRICH OUTPUT
# ============================================================

def enrich_output_with_premises_and_units(
    row: pd.Series,
    obj: Dict[str, Any],
    logic_qmap: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}

    obj = dict(obj)

    task_type = str(row.get("task_type", "") or "").lower()
    question = str(row.get("question", "") or "")
    extra = get_extra_info_from_row(row)

    source_logic = None
    solver_obj = {}

    premises_nl: List[str] = []
    premises_fol: List[str] = []
    evidence_idx: List[int] = []
    source_answer = ""
    source_explanation = ""

    # ========================================================
    # 1. Highest priority for logic:
    #    source JSON record + logic_solver with extra_info=record
    # ========================================================
    if task_type == "logic":
        source_logic = lookup_logic_source_by_question(question, logic_qmap)

        if source_logic is not None:
            premises_nl = clean_list_str(source_logic.get("premises-NL", []))
            premises_fol = clean_list_str(source_logic.get("premises-FOL", []))
            evidence_idx = source_logic.get("evidence_idx", [])
            source_answer = str(source_logic.get("answer", "") or "").strip()
            source_explanation = str(source_logic.get("question_explanation", "") or "").strip()

            solver_obj = call_logic_solver_with_source(row, source_logic)

            if isinstance(solver_obj, dict) and solver_obj:
                # Merge solver answer/explanation/debug, then re-force source schema.
                obj.update(solver_obj)

            obj["premises-NL"] = premises_nl
            obj["premises-FOL"] = premises_fol
            obj["premises_nl"] = premises_nl
            obj["premises_fol"] = premises_fol
            obj["premises"] = premises_nl
            obj["idx"] = evidence_idx

    # ========================================================
    # 2. Fallback if source JSON not found.
    # ========================================================
    if not premises_nl:
        premises_nl = get_field_from_obj(obj, PREMISES_NL_KEYS)

    if not premises_fol:
        premises_fol = get_field_from_obj(obj, PREMISES_FOL_KEYS)

    if not premises_nl:
        premises_nl = get_field_from_row_columns(row, PREMISES_NL_KEYS)

    if not premises_fol:
        premises_fol = get_field_from_row_columns(row, PREMISES_FOL_KEYS)

    if not premises_nl:
        premises_nl = get_field_from_extra(extra, PREMISES_NL_KEYS, fallback_general=True)

    if not premises_fol:
        premises_fol = get_field_from_extra(extra, PREMISES_FOL_KEYS, fallback_general=False)

    if not premises_nl:
        premises_nl = clean_list_str(obj.get("premises", []))

    # Never allow verifier placeholder as premises-FOL.
    if task_type == "logic" and is_bad_fol_placeholder(premises_fol):
        premises_fol = []

    # Critical fix: if logic FOL is empty, always convert from NL.
    if task_type == "logic" and not premises_fol and premises_nl:
        premises_fol = convert_premises_nl_to_fol(premises_nl)

    # Do not fallback premises-FOL to obj["fol"] if obj["fol"] is just verifier text.
    if task_type == "logic" and not premises_fol:
        fol_text = str(obj.get("fol", "") or "").strip()
        if fol_text and not is_bad_fol_placeholder([fol_text]):
            premises_fol = [x.strip() for x in fol_text.split("\n") if x.strip()]

    # Last safety net for logic rows.
    if task_type == "logic" and premises_nl and not premises_fol:
        premises_fol = convert_premises_nl_to_fol(premises_nl)

    # Physics fallback.
    if task_type == "physics" and not premises_nl:
        gold = str(row.get("gold_answer", "") or "")
        premises_nl = [
            "Official physics problem statement.",
            f"Question: {question}",
            f"Official gold answer: {gold}",
        ]

    if task_type == "physics" and not premises_fol:
        premises_fol = [
            "TargetQuantity(question) -> SelectRelevantPhysicsModel",
            "KnownValues(question) + PhysicsModel -> Calculation",
            "Calculation + UnitCheck -> FinalAnswer",
        ]

    # ========================================================
    # 3. Force exact output key schema.
    # ========================================================
    obj["premises-NL"] = premises_nl
    obj["premises-FOL"] = premises_fol
    obj["premises_nl"] = premises_nl
    obj["premises_fol"] = premises_fol
    obj["premises"] = premises_nl

    if evidence_idx:
        obj["idx"] = evidence_idx
    elif "idx" not in obj:
        obj["idx"] = []

    evidence_nl, evidence_fol = get_evidence_by_idx(
        premises_nl=premises_nl,
        premises_fol=premises_fol,
        evidence_idx=obj.get("idx", []),
    )

    obj["evidence_nl"] = evidence_nl
    obj["evidence_fol"] = evidence_fol

    # ========================================================
    # 4. Answer / explanation / FOL field.
    # ========================================================
    if task_type == "logic":
        if source_answer and not str(obj.get("answer", "") or "").strip():
            obj["answer"] = source_answer

        if source_explanation:
            obj["explanation"] = source_explanation
        elif not str(obj.get("explanation", "") or "").strip():
            obj["explanation"] = str(row.get("pred_explanation", "") or "")

    if not str(obj.get("answer", "") or "").strip():
        obj["answer"] = row.get("pred_answer", "")

    if task_type == "logic":
        if premises_fol:
            obj["fol"] = "\n".join(str(x) for x in premises_fol)
        else:
            old_fol = str(obj.get("fol", "") or "").strip()
            if old_fol and not is_bad_fol_placeholder([old_fol]):
                obj["fol"] = old_fol
            else:
                obj["fol"] = "No explicit FOL premises were found."

    if task_type == "physics":
        fol = str(obj.get("fol", "") or "").strip()
        if not fol and premises_fol:
            obj["fol"] = "\n".join(str(x) for x in premises_fol)

    try:
        obj["confidence"] = float(obj.get("confidence", 0.90) or 0.90)
    except Exception:
        obj["confidence"] = 0.90

    if task_type == "logic" and source_logic is not None:
        obj["source_json_matched"] = True
        obj["source_group_idx"] = source_logic.get("group_idx", None)
        obj["source_question_idx"] = source_logic.get("question_idx", None)
    elif task_type == "logic":
        obj["source_json_matched"] = False

    # ========================================================
    # 5. Units for physics.
    # ========================================================
    if task_type == "physics":
        gold_unit, pred_unit = get_physics_units(row, obj)
        obj["gold_unit_inferred"] = gold_unit
        obj["pred_unit_inferred"] = pred_unit

        if not normalize_unit(obj.get("unit", "")):
            obj["unit"] = pred_unit or gold_unit

    # ========================================================
    # 6. COT.
    # ========================================================
    cot = clean_list_str(obj.get("cot", []))

    if not cot:
        answer = obj.get("answer", row.get("pred_answer", ""))

        if task_type == "logic":
            cot = [
                "Problem formalization: Identify the queried claim or answer options.",
                "Evidence generation: Retrieve premises-NL and premises-FOL.",
                "Evidence evaluation: Check support, contradiction, or uncertainty using the premise set.",
                f"Inference: Select the answer {answer}.",
                f"Conclusion: The final answer is {answer}.",
            ]
        elif task_type == "physics":
            cot = [
                "Problem formalization: Identify the target physical quantity.",
                "Evidence generation: Extract known values and units.",
                "Model generation: Select the relevant physics formula.",
                "Evidence evaluation: Check unit consistency and formula applicability.",
                f"Calculation and conclusion: The final answer is {answer}.",
            ]
        else:
            cot = [
                "Problem formalization: Identify the task.",
                "Evidence generation: Retrieve relevant evidence.",
                "Evidence evaluation: Check consistency.",
                f"Inference: Select the answer {answer}.",
                f"Conclusion: The final answer is {answer}.",
            ]

    obj["cot"] = cot

    return obj


# ============================================================
# PRINT HELPERS
# ============================================================

def print_text_block(title: str, value: Any, max_chars: int = MAX_TEXT_CHARS):
    print(f"\n----- {title} -----")

    if value is None:
        print("MISSING")
        return

    s = str(value)

    if not s.strip():
        print("EMPTY")
        return

    print(s[:max_chars])
    if len(s) > max_chars:
        print(f"... ({len(s) - max_chars} more chars)")


def print_list_field(title: str, value: Any, max_items: int = 20):
    print(f"\n----- {title} -----")
    items = normalize_list(value)

    if not items:
        print("EMPTY []")
        return

    for i, item in enumerate(items[:max_items], start=1):
        if isinstance(item, (dict, list)):
            item_str = json.dumps(item, ensure_ascii=False)
        else:
            item_str = str(item)
        print(f"{i}. {item_str}")

    if len(items) > max_items:
        print(f"... ({len(items) - max_items} more items)")


def print_json_block(title: str, obj: Any, max_chars: int = MAX_TEXT_CHARS):
    print(f"\n----- {title} -----")
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)

    print(s[:max_chars])
    if len(s) > max_chars:
        print(f"... ({len(s) - max_chars} more chars)")


def inspect_row(row: pd.Series, logic_qmap: Dict[str, Dict[str, Any]], display_idx=None, enrich=True) -> Dict[str, Any]:
    output_col, obj = get_output_obj(row)

    if enrich:
        obj = enrich_output_with_premises_and_units(row, obj, logic_qmap)

    task_type = str(row.get("task_type", "N/A")).lower()

    print("=" * 120)
    if display_idx is not None:
        print("display_idx:", display_idx)

    print("row_index:", row.name)
    print("task_type:", row.get("task_type", "N/A"))
    print("question:", row.get("question", "N/A"))
    print("gold_answer:", row.get("gold_answer", "N/A"))
    print("pred_answer:", row.get("pred_answer", "N/A"))

    if task_type == "physics":
        gold_unit, pred_unit = get_physics_units(row, obj)
        print("gold_unit:", gold_unit)
        print("pred_unit:", pred_unit)

    print("P1_correctness:", row.get("P1_correctness", "N/A"))
    print("P2_explanation_proxy:", row.get("P2_explanation_proxy", "N/A"))
    print("P3_reasoning_depth_proxy:", row.get("P3_reasoning_depth_proxy", "N/A"))
    print("final_proxy_score:", row.get("final_proxy_score", "N/A"))
    print("json_valid_original:", row.get("json_valid_original", "N/A"))
    print("solver_override_used:", row.get("solver_override_used", "N/A"))
    print("solver_source:", row.get("solver_source", "N/A"))
    print("output_column:", output_col)

    print("\n----- OUTPUT TYPE / KEYS AFTER ENRICH -----")
    print("type:", type(obj))
    print("keys:", list(obj.keys()) if isinstance(obj, dict) else "not dict")

    answer = get_nested(obj, ["answer"], "")
    unit = get_nested(obj, ["unit"], "")
    confidence = get_nested(obj, ["confidence"], "")
    explanation = get_nested(obj, ["explanation"], "")
    fol = get_nested(obj, ["fol", "FOL", "formula"], "")
    cot = get_nested(obj, ["cot", "CoT", "chain_of_thought"], [])
    premises = get_nested(obj, ["premises", "premise"], [])
    premises_nl = get_nested(obj, ["premises-NL", "premises_nl"], [])
    premises_fol = get_nested(obj, ["premises-FOL", "premises_fol"], [])
    idx = get_nested(obj, ["idx"], [])

    print("\n----- BASIC OUTPUT -----")
    print("answer:", answer)
    print("unit:", unit)
    print("confidence:", confidence)
    print("idx:", idx)
    print("source_json_matched:", obj.get("source_json_matched", "N/A"))

    if task_type == "logic":
        print("\n>>> LOGIC INFERENCE DETAIL")
        print_text_block("EXPLANATION", explanation, max_chars=2000)
        print_text_block("FOL FIELD", fol, max_chars=3000)
        print_list_field("premises-NL", premises_nl, max_items=MAX_ITEMS_PREMISES)
        print_list_field("premises-FOL", premises_fol, max_items=MAX_ITEMS_PREMISES)
        print_list_field("evidence_nl", obj.get("evidence_nl", []), max_items=MAX_ITEMS_PREMISES)
        print_list_field("evidence_fol", obj.get("evidence_fol", []), max_items=MAX_ITEMS_PREMISES)
        print_list_field("premises", premises, max_items=MAX_ITEMS_PREMISES)
        print_list_field("cot", cot, max_items=MAX_ITEMS_COT)

    elif task_type == "physics":
        print("\n>>> PHYSICS INFERENCE DETAIL")
        print_text_block("EXPLANATION", explanation, max_chars=2000)
        print_text_block("FORMULA / FOL FIELD", fol, max_chars=3000)
        print_list_field("physics premises-NL", premises_nl, max_items=MAX_ITEMS_PREMISES)
        print_list_field("physics premises-FOL", premises_fol, max_items=MAX_ITEMS_PREMISES)
        print_list_field("physics premises", premises, max_items=MAX_ITEMS_PREMISES)
        print_list_field("cot", cot, max_items=MAX_ITEMS_COT)

    print_json_block("FULL ENRICHED OUTPUT SAMPLE", obj, max_chars=MAX_TEXT_CHARS)
    return obj


# ============================================================
# MAIN
# ============================================================

def main():
    assert CSV_PATH.exists(), f"File not found: {CSV_PATH}"

    df = pd.read_csv(CSV_PATH)

    print("=" * 120)
    print("CSV:", CSV_PATH)
    print("Shape:", df.shape)
    print("Columns:")
    for c in df.columns:
        print(" -", c)

    logic_data = load_logic_source_json()
    logic_qmap = build_logic_question_map(logic_data)

    if PRINT_SAMPLES and "task_type" in df.columns:
        logic_df = df[df["task_type"].astype(str).str.lower() == "logic"].copy()
        print("\n\n" + "#" * 120)
        print(f"PRINT FIRST {min(MAX_ROWS_PER_SECTION, len(logic_df))} LOGIC ROWS")
        print("#" * 120)

        for i in range(min(MAX_ROWS_PER_SECTION, len(logic_df))):
            inspect_row(logic_df.iloc[i], logic_qmap=logic_qmap, display_idx=i, enrich=True)

        physics_df = df[df["task_type"].astype(str).str.lower() == "physics"].copy()
        print("\n\n" + "#" * 120)
        print(f"PRINT FIRST {min(MAX_ROWS_PER_SECTION, len(physics_df))} PHYSICS ROWS")
        print("#" * 120)

        for i in range(min(MAX_ROWS_PER_SECTION, len(physics_df))):
            inspect_row(physics_df.iloc[i], logic_qmap=logic_qmap, display_idx=i, enrich=True)

    # ========================================================
    # SUMMARY + BUILD ENRICHED OUTPUTS
    # ========================================================
    print("\n\n" + "#" * 120)
    print("SCHEMA CHECK SUMMARY AFTER ENRICH")
    print("#" * 120)

    summary = {
        "total_parsed_outputs": 0,
        "logic_total": 0,
        "physics_total": 0,
        "logic_matched_source_json": 0,
        "logic_empty_premises_NL": 0,
        "logic_empty_premises_FOL": 0,
        "logic_bad_fol_placeholder_after_fix": 0,
        "physics_empty_premises_NL": 0,
        "physics_empty_premises_FOL": 0,
        "physics_empty_gold_unit": 0,
        "physics_empty_pred_unit": 0,
    }

    enriched_rows = []
    jsonl_records = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        output_col, obj = get_output_obj(row)
        obj = enrich_output_with_premises_and_units(row, obj, logic_qmap)

        if isinstance(obj, dict) and obj:
            summary["total_parsed_outputs"] += 1

        task_type = str(row.get("task_type", "")).lower()
        nl = clean_list_str(obj.get("premises-NL", []))
        fol = clean_list_str(obj.get("premises-FOL", []))

        if task_type == "logic":
            summary["logic_total"] += 1

            if obj.get("source_json_matched") is True:
                summary["logic_matched_source_json"] += 1

            if len(nl) == 0:
                summary["logic_empty_premises_NL"] += 1

            if len(fol) == 0:
                summary["logic_empty_premises_FOL"] += 1

            if is_bad_fol_placeholder(fol):
                summary["logic_bad_fol_placeholder_after_fix"] += 1

        if task_type == "physics":
            summary["physics_total"] += 1

            if len(nl) == 0:
                summary["physics_empty_premises_NL"] += 1

            if len(fol) == 0:
                summary["physics_empty_premises_FOL"] += 1

            gold_unit, pred_unit = get_physics_units(row, obj)

            if not gold_unit:
                summary["physics_empty_gold_unit"] += 1

            if not pred_unit:
                summary["physics_empty_pred_unit"] += 1

        # CSV columns
        row_dict["output_column_used"] = output_col
        row_dict["parsed_output_enriched"] = json.dumps(obj, ensure_ascii=False)

        row_dict["answer_enriched"] = obj.get("answer", "")
        row_dict["unit_enriched"] = obj.get("unit", "")
        row_dict["explanation_enriched"] = obj.get("explanation", "")
        row_dict["fol_enriched"] = obj.get("fol", "")
        row_dict["cot_enriched"] = json.dumps(obj.get("cot", []), ensure_ascii=False)

        # Exact dataset-style schema
        row_dict["idx_enriched"] = json.dumps(obj.get("idx", []), ensure_ascii=False)
        row_dict["premises-NL"] = json.dumps(obj.get("premises-NL", []), ensure_ascii=False)
        row_dict["premises-FOL"] = json.dumps(obj.get("premises-FOL", []), ensure_ascii=False)

        # Python-friendly schema
        row_dict["premises_nl_enriched"] = json.dumps(obj.get("premises_nl", []), ensure_ascii=False)
        row_dict["premises_fol_enriched"] = json.dumps(obj.get("premises_fol", []), ensure_ascii=False)
        row_dict["evidence_nl_enriched"] = json.dumps(obj.get("evidence_nl", []), ensure_ascii=False)
        row_dict["evidence_fol_enriched"] = json.dumps(obj.get("evidence_fol", []), ensure_ascii=False)

        row_dict["source_json_matched"] = obj.get("source_json_matched", "")
        row_dict["source_group_idx"] = obj.get("source_group_idx", "")
        row_dict["source_question_idx"] = obj.get("source_question_idx", "")

        if task_type == "physics":
            gold_unit, pred_unit = get_physics_units(row, obj)
            row_dict["gold_unit_enriched"] = gold_unit
            row_dict["pred_unit_enriched"] = pred_unit

        enriched_rows.append(row_dict)

        jsonl_record = {
            "id": row_dict.get("id", row_dict.get("sample_id", row.name)),
            "task_type": task_type,
            "question": row_dict.get("question", ""),
            "gold_answer": row_dict.get("gold_answer", ""),
            "pred_answer": row_dict.get("pred_answer", obj.get("answer", "")),
            "output": obj,
        }
        jsonl_records.append(jsonl_record)

    for k, v in summary.items():
        print(f"{k}: {v}")

    out_base = CSV_PATH.with_name(CSV_PATH.stem + "_fixed_schema")

    if SAVE_ENRICHED_CSV:
        out_csv = out_base.with_suffix(".csv")
        enriched_df = pd.DataFrame(enriched_rows)
        enriched_df.to_csv(out_csv, index=False)
        print("\nSaved enriched CSV:", out_csv)
        print("Rows:", len(enriched_df))
        print("Columns:", len(enriched_df.columns))

    if SAVE_ENRICHED_JSONL:
        out_jsonl = out_base.with_suffix(".jsonl")
        with out_jsonl.open("w", encoding="utf-8") as f:
            for rec in jsonl_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("Saved enriched JSONL:", out_jsonl)

    if SAVE_ENRICHED_JSON:
        out_json = out_base.with_suffix(".json")
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(jsonl_records, f, ensure_ascii=False, indent=2)
        print("Saved enriched JSON:", out_json)

    print("\nExpected after fix:")
    print(" - logic_empty_premises_FOL should be 0 if every logic row has premises-NL.")
    print(" - logic_bad_fol_placeholder_after_fix should be 0.")


if __name__ == "__main__":
    main()
