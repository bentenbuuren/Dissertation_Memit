#!/bin/bash

#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=lora_deepseek.out
#SBATCH --error=lora_deepseek.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=lora

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

echo ""
echo "🔄 Starting LoRA fine-tuning..."
# LoRA fine-tuning for DeepSeek (original model)
python run_finetune_peft.py \
    --model_folder_path "original_models/deepseek-ai_DeepSeek-R1-Distill-Llama-8B" \
    --model_name 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' \
    --data_path 'dataset/commonsense_170k.json' --adapter_name 'lora' \
    --output_dir 'deepseek-ai_DeepSeek-R1-Distill-Llama-8B_lora' \
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
