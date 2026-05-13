from exact_modules.common import clamp01, extract_json, parse_obj
from exact_modules.reward.curriculum import get_reward_weights
from exact_modules.reward.self_revision_reward import self_revision_improvement_reward
from exact_modules.verifier import exact_reward_components


def compute_advanced_exact_reward(solution_str, ground_truth, extra_info):
    extra_info = parse_obj(extra_info)
    task_type = extra_info.get("task_type", "")
    question = extra_info.get("question", "")

    raw_text = str(solution_str or "")
    obj = extract_json(raw_text)
    json_valid = obj is not None

    if obj is None:
        pred_answer = raw_text.strip().splitlines()[-1] if raw_text.strip() else ""
    else:
        pred_answer = obj.get("answer", "")

    comps = exact_reward_components(
        obj=obj,
        raw_text=raw_text,
        pred_answer=pred_answer,
        gold=ground_truth,
        task_type=task_type,
        question=question,
        extra_info=extra_info,
    )

    weights = get_reward_weights(task_type)

    base_reward = (
        weights.get("answer", 0.0) * comps["answer"]
        + weights.get("sol", 0.0) * comps["sol"]
        + weights.get("format", 0.0) * comps["format"]
        + weights.get("explanation", 0.0) * comps["explanation"]
        + weights.get("reasoning", 0.0) * comps["reasoning"]
        + weights.get("symbolic", 0.0) * comps["symbolic"]
        + weights.get("floor", 0.0) * comps["floor"]
    )

    base_reward = clamp01(base_reward)

    revision_reward = self_revision_improvement_reward(
        obj=obj,
        raw_text=raw_text,
        base_reward=base_reward,
        gold=ground_truth,
        task_type=task_type,
        question=question,
        extra_info=extra_info,
        json_valid_original=json_valid,
    )

    final_reward = base_reward + weights.get("revision", 0.0) * revision_reward

    return float(clamp01(final_reward))
