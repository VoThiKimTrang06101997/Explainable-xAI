#!/bin/bash
set -x

ray stop -f || true

python3 - <<'PY'
import gc
try:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("CUDA cache cleared.")
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Skip CUDA clean:", e)
PY

nvidia-smi || true

# ===== Memory / attention =====
export VLLM_ATTENTION_BACKEND=XFORMERS
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export HF_USE_FLASH_ATTENTION_2=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export RAY_TMPDIR="/tmp/ray"
mkdir -p $RAY_TMPDIR

# ===== EXACT advanced flags =====
export EXACT_USE_DENSE_REWARD=1
export EXACT_USE_TASK_ADAPTIVE_REWARD=1
export EXACT_USE_LIGHT_Z3_TRAIN=1
export EXACT_USE_SOL_REWARD=1
export EXACT_USE_MOE_ROUTER=1

# Advanced training enabled
export EXACT_USE_SELF_REVISION_RL=1
export EXACT_USE_RANK_BONUS=1
export EXACT_USE_CURRICULUM_RL=1

# Curriculum auto
export EXACT_REWARD_STAGE=auto
export EXACT_CURRICULUM_PHASE1_CALLS=800
export EXACT_CURRICULUM_PHASE2_CALLS=2200

# Rank bonus
export EXACT_RANK_BONUS_ALPHA=0.12
export EXACT_ROLLOUT_N=2

# Self-revision reward
export EXACT_REVISION_LAMBDA=0.25

# Heavy inference modules off during training
export EXACT_USE_HEAVY_Z3_INFERENCE=0
export EXACT_USE_SELF_REVISION_INFERENCE=0

export CHECKPOINTS_DIR="/content/drive/MyDrive/Explainable_AI/Checkpoint"
export WANDB_PROJECT="exact_rlvr_advanced"
export WANDB_NAME="exact-qwen0.5b-advanced-rlvr"
export HYDRA_FULL_ERROR=1

mkdir -p $CHECKPOINTS_DIR
cd /content/One-Shot-RLVR

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=data/train/exact_rlvr/exact_train_router.parquet \
  data.val_files=data/val/exact_rlvr/exact_val_router.parquet \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  reward_model.reward_manager='exact_rank' \
  actor_rollout_ref.model.path='Qwen/Qwen2.5-0.5B-Instruct' \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.7 \
  +actor_rollout_ref.rollout.val_temperature=0.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.n=2 \
  +actor_rollout_ref.rollout.n_val=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='exact_rlvr_advanced' \
  trainer.experiment_name='exact-qwen0.5b-advanced-rlvr' \
  trainer.checkpoints_dir=$CHECKPOINTS_DIR \
  +trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=100000 \
  trainer.default_hdfs_dir=null \
  trainer.total_epochs=1 2>&1 | tee /content/drive/MyDrive/Explainable_AI/Checkpoint/exact_advanced_rlvr_train.log

  