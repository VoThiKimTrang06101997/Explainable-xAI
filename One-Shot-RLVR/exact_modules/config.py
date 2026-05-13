import os


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v).lower() in ["1", "true", "yes", "y"]


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


EXACT_CONFIG = {
    # Base training modules
    "use_dense_reward": env_bool("EXACT_USE_DENSE_REWARD", True),
    "use_task_adaptive_reward": env_bool("EXACT_USE_TASK_ADAPTIVE_REWARD", True),
    "use_light_z3_train": env_bool("EXACT_USE_LIGHT_Z3_TRAIN", True),
    "use_sol_reward": env_bool("EXACT_USE_SOL_REWARD", True),
    "use_moe_router": env_bool("EXACT_USE_MOE_ROUTER", True),

    # Advanced training modules
    "use_self_revision_rl": env_bool("EXACT_USE_SELF_REVISION_RL", True),
    "use_contrastive_rank_bonus": env_bool("EXACT_USE_RANK_BONUS", True),
    "use_curriculum_rl": env_bool("EXACT_USE_CURRICULUM_RL", True),

    # Inference/evaluation modules
    "use_heavy_z3_inference": env_bool("EXACT_USE_HEAVY_Z3_INFERENCE", True),
    "use_self_revision_inference": env_bool("EXACT_USE_SELF_REVISION_INFERENCE", True),

    # Curriculum
    "reward_stage": os.environ.get("EXACT_REWARD_STAGE", "auto"),
    "curriculum_phase1_calls": env_int("EXACT_CURRICULUM_PHASE1_CALLS", 800),
    "curriculum_phase2_calls": env_int("EXACT_CURRICULUM_PHASE2_CALLS", 2200),

    # Contrastive rank bonus
    "rank_bonus_alpha": env_float("EXACT_RANK_BONUS_ALPHA", 0.12),

    # Self-revision reward
    "revision_lambda": env_float("EXACT_REVISION_LAMBDA", 0.25),
}
