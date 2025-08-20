#!/bin/bash

#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=output/ds_lora_zsre_1000.out
#SBATCH --error=output/ds_lora_zsre_1000.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=dslrz1000

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit

# =============================================================================
# LLAMA 3.1 8B INSTRUCT FINE-TUNING WITH YOUR EXACT PARAMETERS
# =============================================================================

echo "🦙 Starting Deepseek Fine-tuning"
echo "================================================"

echo ""
echo "🔄 Starting LoRA fine-tuning..."
# LoRA fine-tuning for DeepSeek (original model)
python run_finetune_peft.py \
    --model_folder_path "edited_models/DeepSeek-R1-Distill-Llama-8B_MEMIT_zsre_1000edits_layers1-2-3-4-5" \
    --model_name 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' \
    --data_path 'dataset/commonsense_170k.json' --adapter_name 'lora' \
    --batch_size 8 --micro_batch_size 8 --num_epochs 3 --learning_rate 2e-4 --weight_decay 0.0 \
    --use_gradient_checkpointing True --val_set_size 120 --eval_step 80 --save_step 80 \
    --cutoff_len 256 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --dora_simple True \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' --train_on_inputs True

if [ $? -eq 0 ]; then
    echo "✅ LoRA fine-tuning completed successfully!"
else
    echo "❌ LoRA fine-tuning failed!"
    exit 1
fi

echo ""
echo "🎉 ALL FINE-TUNING COMPLETED!"
echo "=============================="
echo ""
