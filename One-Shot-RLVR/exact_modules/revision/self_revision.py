import json


def build_revision_prompt(question, draft_output, verifier_feedback):
    return f"""You are revising an explainable educational QA answer.

Original question:
{question}

Draft answer JSON:
{json.dumps(draft_output, ensure_ascii=False, indent=2)}

Verifier feedback:
{verifier_feedback}

Revise the answer. Return valid JSON only with:
answer, explanation, fol, cot, premises, confidence.
"""


def build_verifier_feedback(task_type, final_obj, gold_answer=None):
    feedback = []

    if not final_obj.get("answer"):
        feedback.append("The answer field is empty.")

    if not final_obj.get("explanation"):
        feedback.append("The explanation field is missing or too short.")

    if not final_obj.get("cot"):
        feedback.append("The chain-of-thought steps are missing.")

    if not final_obj.get("premises"):
        feedback.append("The supporting premises are missing.")

    if task_type == "physics" and "formula" not in str(final_obj.get("explanation", "")).lower():
        feedback.append("For physics, the explanation should mention the relevant formula and substitution.")

    if task_type == "logic" and not final_obj.get("fol"):
        feedback.append("For logic, the FOL/rule representation should be provided.")

    if gold_answer is not None:
        feedback.append(f"Gold answer for training/evaluation reference is: {gold_answer}")

    if not feedback:
        return "The draft answer is structurally valid. Improve clarity and confidence if necessary."

    return " ".join(feedback)
