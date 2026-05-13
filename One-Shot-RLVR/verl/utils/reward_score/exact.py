import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exact_modules.reward.advanced_reward import compute_advanced_exact_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None, use_think=False):
    """
    EXACT advanced reward:
    - curriculum reward
    - dense reward
    - task-adaptive reward
    - SOL physics reward
    - light Z3 train reward
    - self-revision RL proxy reward
    """
    return compute_advanced_exact_reward(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    