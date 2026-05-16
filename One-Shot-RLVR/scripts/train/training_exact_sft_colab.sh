# %%writefile /content/Explainable-xAI/One-Shot-RLVR/scripts/train/training_exact_sft_colab.sh
#!/bin/bash
set -x

cd /content/Explainable-xAI/One-Shot-RLVR || exit 1

export TRANSFORMERS_NO_FLASH_ATTENTION=1
export HF_USE_FLASH_ATTENTION_2=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

SFT_ROOT="/content/drive/MyDrive/Explainable_AI/SFT"
mkdir -p "$SFT_ROOT"

python train_sft_exact.py \
  --base_model "Qwen/Qwen2.5-0.5B-Instruct" \
  --train_jsonl "data/train/exact_rlvr/exact_sft_logicality.jsonl" \
  --output_dir "$SFT_ROOT/exact_qwen0p5b_lora" \
  --merged_output_dir "$SFT_ROOT/exact_qwen0p5b_logicality_merged" \
  --max_length 2048 \
  --epochs 1 \
  --lr 2e-5 \
  --batch_size 1 \
  --grad_accum 8



# Chạy SFT:

# %cd /content/Explainable-xAI/One-Shot-RLVR

# !chmod +x scripts/train/training_exact_sft_colab.sh
# !bash scripts/train/training_exact_sft_colab.sh

# SFT model sẽ nằm ở:

# /content/drive/MyDrive/Explainable_AI/SFT/exact_qwen0p5b_logicality_merged