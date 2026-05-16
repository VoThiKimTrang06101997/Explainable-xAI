# %%writefile /content/Explainable-xAI/One-Shot-RLVR/exact_modules/logic_solver.py
import re
from typing import List, Dict, Any, Optional


def normalize_text(x: str) -> str:
    x = str(x or "").lower()
    x = re.sub(r"[^a-z0-9\s\-]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def split_mcq_options(question: str) -> Dict[str, str]:
    q = str(question or "")
    pattern = r"\b([A-D])\.\s*(.*?)(?=\n?[A-D]\.\s*|$)"
    matches = re.findall(pattern, q, flags=re.S)

    options = {}
    for label, text in matches:
        cleaned = " ".join(str(text).strip().split())
        if cleaned:
            options[label.upper()] = cleaned

    return options


def extract_query_without_options(question: str) -> str:
    q = str(question or "")
    q = re.split(r"\bA\.\s*", q)[0]
    return q.strip()


def keyword_overlap_score(option: str, premises: List[str]) -> float:
    option_tokens = set(normalize_text(option).split())
    option_tokens = {t for t in option_tokens if len(t) > 2}

    premise_text = normalize_text(" ".join(premises))
    premise_tokens = set(premise_text.split())

    if not option_tokens:
        return 0.0

    return len(option_tokens & premise_tokens) / max(1, len(option_tokens))


def all_items_supported(option: str, premises: List[str]) -> bool:
    premise_text = normalize_text(" ".join(premises))

    items = re.split(r",| and ", option)
    items = [normalize_text(x) for x in items if normalize_text(x)]

    if not items:
        return False

    supported = 0
    for item in items:
        item_tokens = [t for t in item.split() if len(t) > 2]
        if not item_tokens:
            continue

        if all(tok in premise_text for tok in item_tokens):
            supported += 1

    return supported == len(items)


def contradiction_score(statement: str, premises: List[str]) -> float:
    s = normalize_text(statement)
    p = normalize_text(" ".join(premises))

    neg_words = ["not", "no", "never", "cannot", "does not", "do not", "without"]

    if any(w in s for w in neg_words) and not any(w in p for w in neg_words):
        return 0.2

    if any(w in p for w in neg_words) and not any(w in s for w in neg_words):
        return 0.2

    return 0.0


def solve_logic_mcq(question: str, premises: List[str]) -> Optional[Dict[str, Any]]:
    options = split_mcq_options(question)

    if not options:
        return None

    query = extract_query_without_options(question)
    scored = []

    for label, option_text in options.items():
        overlap = keyword_overlap_score(option_text, premises)
        all_supported = all_items_supported(option_text, premises)

        score = overlap
        if all_supported:
            score += 0.5

        scored.append((label, option_text, score, overlap, all_supported))

    scored = sorted(scored, key=lambda x: x[2], reverse=True)
    best_label, best_text, best_score, best_overlap, best_all_supported = scored[0]

    confidence = min(0.95, 0.45 + best_score)

    explanation = (
        f"The question asks: {query}. "
        f"Each option is evaluated against the premises. "
        f"Option {best_label} has the strongest evidence support."
    )

    return {
        "answer": best_label,
        "explanation": explanation,
        "fol": "For each option o: SupportedByPremises(o) -> CandidateAnswer(o)",
        "cot": [
            "Problem formalization: Extract the question and answer options A/B/C/D.",
            "Evidence generation: Collect the relevant premises.",
            "Evidence evaluation: Compare each option against the premises.",
            f"Inference: Option {best_label} receives the strongest support.",
            f"Conclusion: The final answer is {best_label}."
        ],
        "premises": premises,
        "confidence": confidence,
        "debug_scores": [
            {
                "label": label,
                "option": text,
                "score": score,
                "overlap": overlap,
                "all_supported": all_supported,
            }
            for label, text, score, overlap, all_supported in scored
        ],
        "source": "logic_mcq_option_verifier",
    }


def solve_logic_yes_no_unknown(question: str, premises: List[str]) -> Optional[Dict[str, Any]]:
    q = normalize_text(question)
    premise_text = normalize_text(" ".join(premises))

    if not premises:
        return {
            "answer": "Unknown",
            "explanation": "No premises are provided, so the answer cannot be determined.",
            "fol": "NoPremises -> Unknown",
            "cot": [
                "Problem formalization: Identify the claim in the question.",
                "Evidence generation: Check available premises.",
                "Evidence evaluation: No premise is available for support or contradiction.",
                "Inference: The claim cannot be determined.",
                "Conclusion: The final answer is Unknown."
            ],
            "premises": [],
            "confidence": 0.70,
            "source": "logic_no_premise_unknown",
        }

    q_tokens = set(q.split())
    q_tokens = {
        t for t in q_tokens
        if len(t) > 2 and t not in [
            "does", "all", "any", "which", "what", "when", "where",
            "student", "students", "answer", "true", "false", "following"
        ]
    }

    if not q_tokens:
        return None

    premise_tokens = set(premise_text.split())
    overlap = len(q_tokens & premise_tokens) / max(1, len(q_tokens))
    contra = contradiction_score(question, premises)

    if contra > 0:
        answer = "No"
        confidence = 0.65
        reason = "The premises contain a contradiction signal for the queried claim."
    elif overlap >= 0.65:
        answer = "Yes"
        confidence = min(0.90, 0.50 + overlap * 0.40)
        reason = "The key terms in the query are sufficiently supported by the premises."
    else:
        answer = "Unknown"
        confidence = 0.65
        reason = "The premises do not provide enough direct support for the query."

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
    }


def solve_logic(question: str, premises: List[str]) -> Optional[Dict[str, Any]]:
    options = split_mcq_options(question)

    if options:
        return solve_logic_mcq(question, premises)

    return solve_logic_yes_no_unknown(question, premises)

