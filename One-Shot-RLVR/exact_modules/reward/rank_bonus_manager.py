import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exact_modules.config import EXACT_CONFIG

try:
    from verl.workers.reward_manager.naive import NaiveRewardManager
except Exception:
    NaiveRewardManager = None


def _parse_extra(x):
    if isinstance(x, dict):
        return x

    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}

    return {}


def _get_group_keys(data, batch_size):
    """
    Group candidates from same prompt.
    Preferred key: extra_info.index or question.
    Fallback: i // rollout_n.
    """
    rollout_n = int(os.environ.get("EXACT_ROLLOUT_N", "2"))
    keys = [i // rollout_n for i in range(batch_size)]

    try:
        nb = data.non_tensor_batch
        extra_infos = nb.get("extra_info", None)

        if extra_infos is not None:
            new_keys = []

            for i in range(batch_size):
                extra = _parse_extra(extra_infos[i])
                idx = extra.get("index", None)
                question = extra.get("question", None)

                if idx is not None:
                    new_keys.append(str(idx))
                elif question is not None:
                    new_keys.append(str(question)[:200])
                else:
                    new_keys.append(keys[i])

            keys = new_keys

    except Exception:
        pass

    return keys


def _last_reward_position(reward_tensor, row_idx):
    nz = torch.nonzero(reward_tensor[row_idx] != 0, as_tuple=False)

    if nz.numel() > 0:
        return int(nz[-1].item())

    return reward_tensor.shape[-1] - 1


class ExactRankBonusRewardManager(NaiveRewardManager):
    """
    Adds verifier-ranked GRPO bonus after normal reward computation.

    For candidates in the same prompt group:
        best candidate gets +alpha reward bonus.
    This strengthens contrastive learning signal.
    """

    def __call__(self, data):
        reward_tensor = super().__call__(data)

        if not EXACT_CONFIG.get("use_contrastive_rank_bonus", True):
            return reward_tensor

        alpha = float(EXACT_CONFIG.get("rank_bonus_alpha", 0.12))

        if alpha <= 0:
            return reward_tensor

        try:
            batch_size = reward_tensor.shape[0]
            row_rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()
            keys = _get_group_keys(data, batch_size)

            groups = defaultdict(list)
            for i, k in enumerate(keys):
                groups[k].append(i)

            for _, idxs in groups.items():
                if len(idxs) < 2:
                    continue

                best_reward = max(row_rewards[i] for i in idxs)
                best_idxs = [i for i in idxs if row_rewards[i] == best_reward]

                for i in best_idxs:
                    pos = _last_reward_position(reward_tensor, i)
                    reward_tensor[i, pos] += alpha

        except Exception as e:
            print(f"[WARN] ExactRankBonusRewardManager failed: {e}")

        return reward_tensor
    