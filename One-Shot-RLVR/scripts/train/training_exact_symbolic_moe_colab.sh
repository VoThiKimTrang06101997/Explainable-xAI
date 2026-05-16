#!/bin/bash
set -x

# ============================================================
# 0. Basic paths
# ============================================================
export REPO_DIR="/content/Explainable-xAI/One-Shot-RLVR"
export CHECKPOINTS_DIR="/content/drive/MyDrive/Explainable_AI/Checkpoint"
export RESULTS_DIR="/content/drive/MyDrive/Explainable_AI/Results"

mkdir -p "$CHECKPOINTS_DIR"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR" || exit 1

# ============================================================
# 1. Cleanup function: run before train and on exit/error
# ============================================================
cleanup_memory() {
  echo "================ CLEANUP MEMORY ================"

  ray stop -f || true

  # Kill leftover Ray / vLLM / python workers from previous failed runs
  pkill -f "ray::" || true
  pkill -f "raylet" || true
  pkill -f "plasma_store" || true
  pkill -f "vllm" || true

  # Clean Ray temp/object store
  rm -rf /tmp/ray || true
  rm -rf /tmp/runtime_env_* || true
  rm -rf /tmp/tmp*ray* || true

  # Python CUDA cleanup
  python3 - <<'PY'
import gc
import os

try:
    import torch
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        print("CUDA cache cleared.")
        print("GPU:", torch.cuda.get_device_name(0))
        print("Allocated GB:", round(torch.cuda.memory_allocated() / 1024**3, 4))
        print("Reserved GB:", round(torch.cuda.memory_reserved() / 1024**3, 4))
    else:
        print("CUDA is not available.")
except Exception as e:
    print("Skip CUDA clean:", e)
PY

  nvidia-smi || true
  echo "============== END CLEANUP MEMORY =============="
}

# Run cleanup when script exits, even if error
trap cleanup_memory EXIT

# Initial cleanup before training
cleanup_memory

# ============================================================
# 2. Environment variables for memory stability
# ============================================================

# Attention / FlashAttention disabled
export VLLM_ATTENTION_BACKEND=XFORMERS
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export HF_USE_FLASH_ATTENTION_2=0
export FLASH_ATTENTION_FORCE_DISABLE=1

# CUDA memory fragmentation control
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Reduce CPU thread overload
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

# Ray temp dir on local disk, not Drive
export RAY_TMPDIR="/tmp/ray"
mkdir -p "$RAY_TMPDIR"

# Optional: reduce NCCL weirdness on Colab single GPU
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# HuggingFace cache on local disk to avoid Drive latency
export HF_HOME="/content/hf_cache"
export TRANSFORMERS_CACHE="/content/hf_cache"
mkdir -p "$HF_HOME"

# ============================================================
# 3. EXACT advanced flags
# ============================================================
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

# WandB / Hydra
export WANDB_PROJECT="exact_rlvr_advanced"
export WANDB_NAME="exact-qwen0.5b-advanced-rlvr"
export HYDRA_FULL_ERROR=1

# ============================================================
# 4. Show system status before train
# ============================================================
echo "================ SYSTEM STATUS BEFORE TRAIN ================"
pwd
python3 --version
python3 - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi || true
echo "============================================================="

# ============================================================
# 5. Train
# ============================================================

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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.n=2 \
  +actor_rollout_ref.rollout.n_val=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='exact_rlvr_advanced' \
  trainer.experiment_name='exact-qwen0.5b-advanced-rlvr' \
  trainer.checkpoints_dir="$CHECKPOINTS_DIR" \
  +trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=100000 \
  trainer.default_hdfs_dir=null \
  trainer.total_epochs=1 2>&1 | tee "$CHECKPOINTS_DIR/exact_advanced_rlvr_train.log"
  


# #!/bin/bash
# set -x

# # ============================================================
# # 0. Basic paths
# # ============================================================
# export REPO_DIR="/content/Explainable-xAI/One-Shot-RLVR"
# export CHECKPOINTS_DIR="/content/drive/MyDrive/Explainable_AI/Checkpoint"
# export RESULTS_DIR="/content/drive/MyDrive/Explainable_AI/Results"

# mkdir -p "$CHECKPOINTS_DIR"
# mkdir -p "$RESULTS_DIR"

# cd "$REPO_DIR" || exit 1

# # ============================================================
# # 1. Cleanup function: run before train and on exit/error
# # ============================================================
# cleanup_memory() {
#   echo "================ CLEANUP MEMORY ================"

#   ray stop -f || true

#   # Kill leftover Ray / vLLM / python workers from previous failed runs
#   pkill -f "ray::" || true
#   pkill -f "raylet" || true
#   pkill -f "plasma_store" || true
#   pkill -f "vllm" || true

#   # Clean Ray temp/object store
#   rm -rf /tmp/ray || true
#   rm -rf /tmp/runtime_env_* || true
#   rm -rf /tmp/tmp*ray* || true

#   # Python CUDA cleanup
#   python3 - <<'PY'
# import gc
# import os

# try:
#     import torch
#     gc.collect()

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()
#         torch.cuda.synchronize()
#         print("CUDA cache cleared.")
#         print("GPU:", torch.cuda.get_device_name(0))
#         print("Allocated GB:", round(torch.cuda.memory_allocated() / 1024**3, 4))
#         print("Reserved GB:", round(torch.cuda.memory_reserved() / 1024**3, 4))
#     else:
#         print("CUDA is not available.")
# except Exception as e:
#     print("Skip CUDA clean:", e)
# PY

#   nvidia-smi || true
#   echo "============== END CLEANUP MEMORY =============="
# }

# # Run cleanup when script exits, even if error
# trap cleanup_memory EXIT

# # Initial cleanup before training
# cleanup_memory

# # ============================================================
# # 2. Environment variables for memory stability
# # ============================================================

# # Attention / FlashAttention disabled
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export TRANSFORMERS_NO_FLASH_ATTENTION=1
# export HF_USE_FLASH_ATTENTION_2=0
# export FLASH_ATTENTION_FORCE_DISABLE=1

# # CUDA memory fragmentation control
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# # Reduce CPU thread overload
# export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=2
# export MKL_NUM_THREADS=2
# export NUMEXPR_NUM_THREADS=2

# # Ray temp dir on local disk, not Drive
# export RAY_TMPDIR="/tmp/ray"
# mkdir -p "$RAY_TMPDIR"

# # Optional: reduce NCCL weirdness on Colab single GPU
# export NCCL_DEBUG=WARN
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# # HuggingFace cache on local disk to avoid Drive latency
# export HF_HOME="/content/hf_cache"
# export TRANSFORMERS_CACHE="/content/hf_cache"
# mkdir -p "$HF_HOME"

# # ============================================================
# # 3. EXACT advanced flags
# # ============================================================
# export EXACT_USE_DENSE_REWARD=1
# export EXACT_USE_TASK_ADAPTIVE_REWARD=1
# export EXACT_USE_LIGHT_Z3_TRAIN=1
# export EXACT_USE_SOL_REWARD=1
# export EXACT_USE_MOE_ROUTER=1

# # Advanced training enabled
# export EXACT_USE_SELF_REVISION_RL=1
# export EXACT_USE_RANK_BONUS=1
# export EXACT_USE_CURRICULUM_RL=1

# # Curriculum auto
# export EXACT_REWARD_STAGE=auto
# export EXACT_CURRICULUM_PHASE1_CALLS=800
# export EXACT_CURRICULUM_PHASE2_CALLS=2200

# # Rank bonus
# export EXACT_RANK_BONUS_ALPHA=0.12
# export EXACT_ROLLOUT_N=2

# # Self-revision reward
# export EXACT_REVISION_LAMBDA=0.25

# # Heavy inference modules off during training
# export EXACT_USE_HEAVY_Z3_INFERENCE=0
# export EXACT_USE_SELF_REVISION_INFERENCE=0

# # WandB / Hydra
# export WANDB_PROJECT="exact_rlvr_advanced"
# export WANDB_NAME="exact-qwen0.5b-advanced-rlvr"
# export HYDRA_FULL_ERROR=1

# # ============================================================
# # 4. Show system status before train
# # ============================================================
# echo "================ SYSTEM STATUS BEFORE TRAIN ================"
# pwd
# python3 --version
# python3 - <<'PY'
# import torch
# print("Torch:", torch.__version__)
# print("CUDA:", torch.version.cuda)
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))
# PY
# nvidia-smi || true
# echo "============================================================="

# # ============================================================
# # 5. Train
# # ============================================================

# python3 -m verl.trainer.main_ppo \
#   algorithm.adv_estimator=grpo \
#   data.train_files=data/train/exact_rlvr/exact_train_router.parquet \
#   data.val_files=data/val/exact_rlvr/exact_val_router.parquet \
#   data.train_batch_size=1 \
#   data.val_batch_size=1 \
#   data.max_prompt_length=1024 \
#   data.max_response_length=512 \
#   reward_model.reward_manager='exact_rank' \
#   actor_rollout_ref.model.path='Qwen/Qwen2.5-0.5B-Instruct' \
#   actor_rollout_ref.model.use_remove_padding=False \
#   +actor_rollout_ref.model.attn_implementation=sdpa \
#   actor_rollout_ref.actor.optim.lr=5e-7 \
#   actor_rollout_ref.actor.ppo_mini_batch_size=1 \
#   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
#   actor_rollout_ref.actor.use_dynamic_bsz=False \
#   actor_rollout_ref.actor.use_kl_loss=True \
#   actor_rollout_ref.actor.kl_loss_coef=0.001 \
#   actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#   actor_rollout_ref.model.enable_gradient_checkpointing=True \
#   actor_rollout_ref.actor.fsdp_config.param_offload=True \
#   +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
#   actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
#   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
#   actor_rollout_ref.ref.fsdp_config.param_offload=True \
#   actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#   actor_rollout_ref.rollout.name=vllm \
#   actor_rollout_ref.rollout.temperature=0.7 \
#   +actor_rollout_ref.rollout.val_temperature=0.0 \
#   actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
#   actor_rollout_ref.rollout.n=2 \
#   +actor_rollout_ref.rollout.n_val=1 \
#   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
#   algorithm.kl_ctrl.kl_coef=0.001 \
#   trainer.critic_warmup=0 \
#   trainer.logger=['console','wandb'] \
#   trainer.project_name='exact_rlvr_advanced' \
#   trainer.experiment_name='exact-qwen0.5b-advanced-rlvr' \
#   trainer.checkpoints_dir="$CHECKPOINTS_DIR" \
#   +trainer.val_before_train=False \
#   trainer.n_gpus_per_node=1 \
#   trainer.nnodes=1 \
#   trainer.save_freq=20 \
#   trainer.test_freq=100000 \
#   trainer.default_hdfs_dir=null \
#   trainer.total_epochs=1 2>&1 | tee "$CHECKPOINTS_DIR/exact_advanced_rlvr_train.log"











