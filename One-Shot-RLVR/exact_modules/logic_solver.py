# %%writefile /content/Explainable-xAI/One-Shot-RLVR/exact_modules/logic_solver.py
import re
import json
import ast
from typing import List, Dict, Any, Optional, Tuple


# ============================================================
# 0. Optional Z3 backend
# ============================================================

try:
    from z3 import Solver, Bool, Implies, And, Or, Not, sat, unsat
    Z3_AVAILABLE = True
except Exception:
    Z3_AVAILABLE = False


# ============================================================
# 1. Basic parsing / normalization
# ============================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "has", "have", "had", "all", "any", "some", "each", "every", "everyone",
    "student", "students", "person", "people", "which", "what", "when",
    "where", "why", "how", "following", "answer", "option", "true", "false",
    "yes", "no", "unknown", "whether", "if", "then", "and", "or", "of",
    "to", "in", "on", "for", "with", "by", "from", "as", "that", "this",
}


NEGATION_WORDS = [
    "not", "no", "never", "cannot", "can't", "does not", "do not",
    "did not", "is not", "are not", "without", "fails to", "lack", "lacks"
]


def safe_parse_obj(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, str):
        s = x.strip()
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return x
    return x


def normalize_text(x: str) -> str:
    x = str(x or "").lower()
    x = x.replace("–", "-").replace("—", "-")
    x = re.sub(r"[^a-z0-9\s\-\_]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def content_words(x: str) -> List[str]:
    toks = normalize_text(x).split()
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS]


def token_set(x: str) -> set:
    return set(content_words(x))


def get_premises_from_extra(extra_info: Dict[str, Any]) -> List[str]:
    if not isinstance(extra_info, dict):
        return []

    for key in ["premises_nl", "premises", "premise", "context", "rules"]:
        val = extra_info.get(key)
        if isinstance(val, list):
            return [str(v) for v in val if str(v).strip()]
        if isinstance(val, str) and val.strip():
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if str(v).strip()]
            except Exception:
                pass

            # split if multi-line premises
            lines = [x.strip() for x in re.split(r"\n+|\r+", val) if x.strip()]
            if len(lines) > 1:
                return lines
            return [val]

    return []


# ============================================================
# 2. MCQ option extraction
# ============================================================

def split_mcq_options(question: str) -> Dict[str, str]:
    q = str(question or "")

    # A. text B. text
    pattern = r"(?:^|\n|\s)\b([A-D])\.\s*(.*?)(?=(?:\n|\s)\b[A-D]\.\s*|$)"
    matches = re.findall(pattern, q, flags=re.S)

    options = {}
    for label, text in matches:
        cleaned = " ".join(str(text).strip().split())
        if cleaned:
            options[label.upper()] = cleaned

    # fallback: A) text B) text
    if len(options) < 2:
        pattern = r"(?:^|\n|\s)\b([A-D])\)\s*(.*?)(?=(?:\n|\s)\b[A-D]\)\s*|$)"
        matches = re.findall(pattern, q, flags=re.S)
        options = {}
        for label, text in matches:
            cleaned = " ".join(str(text).strip().split())
            if cleaned:
                options[label.upper()] = cleaned

    return options


def extract_query_without_options(question: str) -> str:
    q = str(question or "")
    q = re.split(r"(?:^|\n|\s)\bA[\.\)]\s*", q)[0]
    return q.strip()


# ============================================================
# 3. Light logical verification
# ============================================================

def keyword_overlap_score(candidate: str, premises: List[str]) -> float:
    cand = token_set(candidate)
    prem = token_set(" ".join(premises))

    if not cand:
        return 0.0

    return len(cand & prem) / max(1, len(cand))


def phrase_support_score(candidate: str, premises: List[str]) -> float:
    """
    Scores whether important candidate phrases are directly supported by premises.
    """
    c_norm = normalize_text(candidate)
    p_norm = normalize_text(" ".join(premises))

    if not c_norm:
        return 0.0

    # Exact candidate appears in premises
    if c_norm in p_norm:
        return 1.0

    # Split option into items: "problem-solving, communication, critical thinking"
    items = [
        normalize_text(x)
        for x in re.split(r",|;|\band\b|\bor\b", candidate, flags=re.I)
        if normalize_text(x)
    ]

    if not items:
        return keyword_overlap_score(candidate, premises)

    supported = 0
    useful = 0

    for item in items:
        toks = [t for t in item.split() if len(t) > 2 and t not in STOPWORDS]
        if not toks:
            continue

        useful += 1

        if item in p_norm:
            supported += 1
        elif all(tok in p_norm for tok in toks):
            supported += 1

    if useful == 0:
        return keyword_overlap_score(candidate, premises)

    return supported / useful


def all_items_supported(candidate: str, premises: List[str]) -> bool:
    return phrase_support_score(candidate, premises) >= 0.999


def has_negation(x: str) -> bool:
    x_norm = normalize_text(x)
    return any(w in x_norm for w in NEGATION_WORDS)


def contradiction_score(statement: str, premises: List[str]) -> float:
    s_neg = has_negation(statement)
    p_text = " ".join(premises)
    p_neg = has_negation(p_text)

    # weak heuristic only
    if s_neg and not p_neg:
        return 0.20
    if p_neg and not s_neg:
        st = token_set(statement)
        pt = token_set(p_text)
        if len(st & pt) >= max(1, min(3, len(st))):
            return 0.20

    return 0.0


def support_entailment_score(claim: str, premises: List[str]) -> float:
    overlap = keyword_overlap_score(claim, premises)
    phrase = phrase_support_score(claim, premises)
    contra = contradiction_score(claim, premises)

    score = 0.60 * overlap + 0.40 * phrase - contra
    return max(0.0, min(1.0, score))


# ============================================================
# 4. Optional FOL/Z3-lite backend
# ============================================================

def _atom_name(text: str) -> str:
    text = normalize_text(text)
    toks = [t for t in text.split() if t not in STOPWORDS]
    if not toks:
        toks = ["unknown"]
    name = "_".join(toks[:8])
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    return name or "unknown"


def parse_simple_rules_to_z3(premises: List[str]):
    """
    Very light FOL/Z3 parser.
    Handles:
    - if X then Y
    - X implies Y
    - all X are Y
    - X is Y / X has Y
    This is intentionally safe and partial; it returns None if parsing is uncertain.
    """
    if not Z3_AVAILABLE:
        return None, {}, []

    solver = Solver()
    atoms = {}
    rules = []

    def get_atom(name):
        if name not in atoms:
            atoms[name] = Bool(name)
        return atoms[name]

    parsed_any = False

    for prem in premises:
        p = str(prem).strip()
        p_norm = normalize_text(p)

        # if X then Y
        m = re.search(r"\bif\s+(.+?)\s+then\s+(.+)", p, flags=re.I)
        if m:
            a = _atom_name(m.group(1))
            b = _atom_name(m.group(2))
            solver.add(Implies(get_atom(a), get_atom(b)))
            rules.append(("implies", a, b, p))
            parsed_any = True
            continue

        # X implies Y
        m = re.search(r"(.+?)\s+implies\s+(.+)", p, flags=re.I)
        if m:
            a = _atom_name(m.group(1))
            b = _atom_name(m.group(2))
            solver.add(Implies(get_atom(a), get_atom(b)))
            rules.append(("implies", a, b, p))
            parsed_any = True
            continue

        # all X are Y => X -> Y, but represented as atom implication
        m = re.search(r"\ball\s+(.+?)\s+are\s+(.+)", p, flags=re.I)
        if m:
            a = _atom_name(m.group(1))
            b = _atom_name(m.group(2))
            solver.add(Implies(get_atom(a), get_atom(b)))
            rules.append(("all_are", a, b, p))
            parsed_any = True
            continue

        # direct fact: "Alice is responsible" / "students are responsible"
        m = re.search(r"(.+?)\s+(?:is|are|has|have)\s+(.+)", p, flags=re.I)
        if m and len(p_norm.split()) <= 12:
            fact = _atom_name(p)
            solver.add(get_atom(fact))
            rules.append(("fact", fact, None, p))
            parsed_any = True
            continue

    if not parsed_any:
        return None, {}, []

    return solver, atoms, rules


def z3_lite_check_claim(question: str, premises: List[str]) -> Optional[Dict[str, Any]]:
    """
    Conservative Z3-lite check. Returns None unless enough simple rules parsed.
    """
    solver, atoms, rules = parse_simple_rules_to_z3(premises)

    if solver is None:
        return None

    claim_name = _atom_name(question)

    if claim_name not in atoms:
        # Try matching claim to existing atom by token overlap
        q_tokens = token_set(question)
        best = None
        best_score = 0.0

        for name in atoms:
            score = len(q_tokens & set(name.split("_"))) / max(1, len(q_tokens))
            if score > best_score:
                best = name
                best_score = score

        if best is None or best_score < 0.50:
            return None

        claim_name = best

    claim = atoms[claim_name]

    # premises entail claim if premises ∧ ¬claim is unsat
    s1 = Solver()
    for a in solver.assertions():
        s1.add(a)
    s1.add(Not(claim))

    if s1.check() == unsat:
        answer = "Yes"
        confidence = 0.90
        explanation = "The parsed logical rules entail the queried claim."
    else:
        # contradiction if premises ∧ claim is unsat
        s2 = Solver()
        for a in solver.assertions():
            s2.add(a)
        s2.add(claim)

        if s2.check() == unsat:
            answer = "No"
            confidence = 0.90
            explanation = "The parsed logical rules contradict the queried claim."
        else:
            answer = "Unknown"
            confidence = 0.70
            explanation = "The parsed logical rules do not fully determine the queried claim."

    return {
        "answer": answer,
        "explanation": explanation,
        "fol": "Parsed simple rules into Z3-lite Boolean implications.",
        "cot": [
            "Problem formalization: Convert the query into a candidate logical claim.",
            "Evidence generation: Parse simple if-then/all-are/fact premises.",
            "Evidence evaluation: Check entailment and contradiction using Z3-lite.",
            f"Inference: The verifier returns {answer}.",
            f"Conclusion: The final answer is {answer}."
        ],
        "premises": premises,
        "confidence": confidence,
        "source": "z3_lite_verifier",
        "debug_rules": rules,
    }


# ============================================================
# 5. Public solvers
# ============================================================

def solve_logic_mcq(question: str, premises: List[str]) -> Optional[Dict[str, Any]]:
    options = split_mcq_options(question)

    if not options:
        return None

    query = extract_query_without_options(question)
    scored = []

    for label, option_text in options.items():
        overlap = keyword_overlap_score(option_text, premises)
        phrase = phrase_support_score(option_text, premises)
        all_supported = all_items_supported(option_text, premises)
        contra = contradiction_score(option_text, premises)

        score = 0.45 * overlap + 0.45 * phrase - 0.20 * contra

        if all_supported:
            score += 0.25

        score = max(0.0, min(1.0, score))

        scored.append({
            "label": label,
            "option": option_text,
            "score": score,
            "overlap": overlap,
            "phrase_support": phrase,
            "all_supported": all_supported,
            "contradiction": contra,
        })

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    if not scored:
        return None

    best = scored[0]
    second_score = scored[1]["score"] if len(scored) > 1 else 0.0
    margin = best["score"] - second_score

    confidence = min(0.95, 0.50 + best["score"] * 0.35 + max(0, margin) * 0.20)

    # Conservative guard:
    # If the question asks what can be inferred / logically inferred, and no option
    # has strong evidence or margin is weak, return Unknown instead of guessing A/B/C.
    q_lower = str(question).lower()
    inferential_question = any(k in q_lower for k in [
        "can be inferred",
        "logically inferred",
        "which statement can be inferred",
        "which statement is correct",
        "based on the premises",
        "based on the above premises",
    ])

    if inferential_question and (best["score"] < 0.72 or margin < 0.12):
        explanation = (
            f"The question asks: {query}. The option-wise verifier did not find a sufficiently "
            f"strong and unique premise-supported option, so the safest answer is Unknown."
        )
        return {
            "answer": "Unknown",
            "explanation": explanation,
            "fol": "InsufficientUniqueSupport(options, premises) -> Unknown",
            "cot": [
                "Problem formalization: Extract the question and answer options A/B/C/D.",
                "Evidence generation: Collect the relevant premises.",
                "Evidence evaluation: Compare each option against the premise evidence.",
                "Inference: No option has sufficiently strong and unique support.",
                "Conclusion: The final answer is Unknown."
            ],
            "premises": premises,
            "confidence": 0.70,
            "debug_scores": scored,
            "source": "logic_mcq_conservative_unknown_guard",
        }

    explanation = (
        f"The question asks: {query}. Each option is evaluated against the premises. "
        f"Option {best['label']} receives the strongest evidence support."
    )

    return {
        "answer": best["label"],
        "explanation": explanation,
        "fol": "For each option o: SupportedByPremises(o) -> CandidateAnswer(o)",
        "cot": [
            "Problem formalization: Extract the question and answer options A/B/C/D.",
            "Evidence generation: Collect the relevant premises.",
            "Evidence evaluation: Compare each option against the premise evidence.",
            f"Inference: Option {best['label']} has the highest support score.",
            f"Conclusion: The final answer is {best['label']}."
        ],
        "premises": premises,
        "confidence": confidence,
        "debug_scores": scored,
        "source": "logic_mcq_option_verifier",
    }


def solve_logic_yes_no_unknown(question: str, premises: List[str]) -> Optional[Dict[str, Any]]:
    if not premises:
        return {
            "answer": "Unknown",
            "explanation": "No premises are provided, so the answer cannot be determined.",
            "fol": "NoPremises -> Unknown",
            "cot": [
                "Problem formalization: Identify the claim in the question.",
                "Evidence generation: Check whether premises are available.",
                "Evidence evaluation: No supporting or contradicting premise is available.",
                "Inference: The truth value cannot be determined.",
                "Conclusion: The final answer is Unknown."
            ],
            "premises": [],
            "confidence": 0.70,
            "source": "logic_no_premise_unknown",
        }

    z3_result = z3_lite_check_claim(question, premises)
    if z3_result is not None and float(z3_result.get("confidence", 0)) >= 0.85:
        return z3_result

    support = support_entailment_score(question, premises)
    contra = contradiction_score(question, premises)

    if contra >= 0.20 and support < 0.75:
        answer = "No"
        confidence = 0.65
        reason = "The premise evidence contains a contradiction signal for the queried claim."
    elif support >= 0.65:
        answer = "Yes"
        confidence = min(0.90, 0.50 + support * 0.40)
        reason = "The key terms and relations in the query are sufficiently supported by the premises."
    else:
        answer = "Unknown"
        confidence = 0.65
        reason = "The premises do not provide enough direct support or contradiction for the query."

    return {
        "answer": answer,
        "explanation": reason,
        "fol": "Premises entail claim -> Yes; premises contradict claim -> No; otherwise -> Unknown",
        "cot": [
            "Problem formalization: Extract the claim from the question.",
            "Evidence generation: Retrieve the relevant premises.",
            "Evidence evaluation: Check whether the claim is supported, contradicted, or undetermined.",
            f"Inference: The evidence leads to {answer}.",
            f"Conclusion: The final answer is {answer}."
        ],
        "premises": premises,
        "confidence": confidence,
        "source": "logic_yes_no_unknown_verifier",
        "debug_scores": {
            "support": support,
            "contradiction": contra,
        }
    }


def solve_logic(question: str, premises: Optional[List[str]] = None, extra_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Main logic solver used by:
    - SFT distillation
    - RL symbolic reward
    - inference/evaluation/API override

    Priority:
    1. MCQ option-wise verifier
    2. Z3-lite / rule verifier for Yes-No-Unknown
    3. lexical support verifier
    """
    extra_info = extra_info or {}

    if premises is None:
        premises = get_premises_from_extra(extra_info)

    premises = [str(p) for p in (premises or []) if str(p).strip()]

    options = split_mcq_options(question)

    if options:
        return solve_logic_mcq(question, premises)

    return solve_logic_yes_no_unknown(question, premises)

