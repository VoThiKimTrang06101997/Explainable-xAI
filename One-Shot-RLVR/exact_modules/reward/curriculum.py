# %%writefile /content/Explainable-xAI/One-Shot-RLVR/exact_modules/reward/curriculum.py
from exact_modules.config import EXACT_CONFIG

_CALL_COUNTER = 0


def get_curriculum_stage():
    global _CALL_COUNTER
    _CALL_COUNTER += 1

    manual_stage = EXACT_CONFIG.get("reward_stage", "auto")
    if manual_stage != "auto":
        return manual_stage

    if not EXACT_CONFIG.get("use_curriculum_rl", True):
        return "phase3_correctness_symbolic"

    if _CALL_COUNTER < EXACT_CONFIG.get("curriculum_phase1_calls", 800):
        return "phase1_format"

    if _CALL_COUNTER < EXACT_CONFIG.get("curriculum_phase2_calls", 2200):
        return "phase2_reasoning"

    return "phase3_correctness_symbolic"


def get_reward_weights(task_type):
    stage = get_curriculum_stage()
    task_type = str(task_type).lower()

    if stage == "phase1_format":
        return {
            "answer": 0.25,
            "sol": 0.10 if task_type == "physics" else 0.00,
            "format": 0.20,
            "explanation": 0.15,
            "reasoning": 0.15,
            "symbolic": 0.10 if task_type == "logic" else 0.00,
            "revision": 0.02,
            "floor": 0.03,
        }

    if stage == "phase2_reasoning":
        return {
            "answer": 0.45,
            "sol": 0.20 if task_type == "physics" else 0.00,
            "format": 0.08,
            "explanation": 0.08,
            "reasoning": 0.10,
            "symbolic": 0.15 if task_type == "logic" else 0.00,
            "revision": 0.04,
            "floor": 0.02,
        }

    if task_type == "physics":
        return {
            "answer": 0.65,
            "sol": 0.20,
            "format": 0.02,
            "explanation": 0.04,
            "reasoning": 0.04,
            "symbolic": 0.00,
            "revision": 0.03,
            "floor": 0.02,
        }

    return {
        "answer": 0.65,
        "sol": 0.00,
        "format": 0.02,
        "explanation": 0.04,
        "reasoning": 0.04,
        "symbolic": 0.20,
        "revision": 0.03,
        "floor": 0.02,
    }
    