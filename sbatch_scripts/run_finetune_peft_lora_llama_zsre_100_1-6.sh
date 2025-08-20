#!/bin/bash

#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=lm_lora_zsre_100.out
#SBATCH --error=lm_lora_zsre_100.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=lmlrz100

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit

# =============================================================================
# LLAMA 3.1 8B INSTRUCT FINE-TUNING WITH YOUR EXACT PARAMETERS
# =============================================================================

echo "🦙 Starting Llama 3.1 8B Instruct Fine-tuning"
echo "================================================"

# LoRA fine-tuning for Llama 3.1 (original model)
python run_finetune_peft.py \
    --model_folder_path "edited_models_final/Llama-3.1-8B-Instruct_MEMIT_zsre_100edits_layers1-2-3-4-5-6" \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --data_path 'dataset/commonsense_170k.json' --adapter_name 'lora' \
    --batch_size 16 --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 --weight_decay 0.0 \
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
echo "📁 Generated Output Directories:"
echo "   - meta-llama_Llama-3.1-8B-Instruct_dora/"
echo "   - meta-llama_Llama-3.1-8B-Instruct_lora/"
echo ""
echo "💾 Adapter files saved in each directory:"
echo "   - adapter_config.json"
echo "   - adapter_model.bin" 
echo "   - README.md"
echo ""
echo "⏱️  Total estimated training time: ~12-16 hours"
echo "🎯 Ready for downstream evaluation!"
