from exact_modules.common import answer_score, clamp01
from exact_modules.config import EXACT_CONFIG
from exact_modules.organizer import build_final_output
from exact_modules.verifier import explanation_score, format_score, reasoning_score


def score_organizer_output(final_obj, raw_text, gold, task_type, question, extra_info):
    pred_answer = final_obj.get("answer", "")

    r_answer = answer_score(pred_answer, gold, task_type)
    r_format = format_score(final_obj, raw_text)
    r_expl = explanation_score(final_obj, raw_text)
    r_reason = reasoning_score(final_obj)

    return clamp01(
        0.45 * r_answer
        + 0.15 * r_format
        + 0.20 * r_expl
        + 0.20 * r_reason
    )


def self_revision_improvement_reward(
    obj,
    raw_text,
    base_reward,
    gold,
    task_type,
    question,
    extra_info,
    json_valid_original=True,
):
    """
    Train-time self-revision proxy:
    y0 = model output
    y1 = verifier-repaired organizer output
    reward_revision = R(y1) + lambda * max(0, R(y1) - R(y0))

    It does not call the model again during training.
    This avoids slow training/OOM while still giving a self-revision signal.
    """
    if not EXACT_CONFIG.get("use_self_revision_rl", True):
        return 0.0

    final_obj = build_final_output(
        model_obj=obj or {},
        raw_output=raw_text,
        task_type=task_type,
        question=question,
        extra_info=extra_info,
        json_valid_original=json_valid_original,
    )

    revised_reward = score_organizer_output(
        final_obj=final_obj,
        raw_text=raw_text,
        gold=gold,
        task_type=task_type,
        question=question,
        extra_info=extra_info,
    )

    lam = EXACT_CONFIG.get("revision_lambda", 0.25)
    improvement = max(0.0, revised_reward - base_reward)

    return clamp01(revised_reward + lam * improvement)
