#!/bin/bash

#SBATCH --mail-user=ycheng80@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=dora_run.out
#SBATCH --error=dora_run.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=dora

# Load modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate memit

# Dora gpt2-xl
python run_dora.py \
    --model_folder_path 'edited_gpt2_xl_models/gpt2-xl_MEMIT_mcf_10000' \
    --model_name 'gpt2-xl' --data_path 'commonsense_170k.json' \
    --adapter_name 'dora' --output_dir 'gpt2-xl_MEMIT_mcf_10000_dora' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 \
    --weight_decay 0.0 --use_gradient_checkpointing True --val_set_size 120 \
    --eval_step 80 --save_step 80 --cutoff_len 256 --lora_r 32 --lora_alpha 64 \
    --lora_dropout 0.05 --dora_simple True --Wdecompose_target_modules None --train_on_inputs True

# Lora gpt2-xl
python run_dora.py \
    --model_folder_path 'edited_gpt2_xl_models/gpt2-xl_ROME_zsre_1000' \
    --model_name 'gpt2-xl' --data_path 'commonsense_170k.json' \
    --adapter_name 'lora' --output_dir 'gpt2-xl_ROME_zsre_1000_lora' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 \
    --weight_decay 0.0 --use_gradient_checkpointing True --val_set_size 120 \
    --eval_step 80 --save_step 80 --cutoff_len 256 --lora_r 32 --lora_alpha 64 \
    --lora_dropout 0.05 --dora_simple True --Wdecompose_target_modules None --train_on_inputs True

python run_dora.py --model_folder_path 'EleutherAI_gpt-j-6B' --model_name 'EleutherAI/gpt-j-6B' --data_path 'zwhe99/commonsense_170k' --adapter_name 'dora' --output_dir 'gptj-6b_dora' --batch_size 16  --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 --weight_decay 0.0 --use_gradient_checkpointing True --val_set_size 120 --eval_step 80 --save_step 80 --cutoff_len 256 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --dora_simple True --Wdecompose_target_modules None --train_on_inputs True

# Dora llama2-7b, adjusted
python run_dora.py \
    --model_folder_path "Llama-2-7b-hf-MEMIT_zsre_100" --model_name 'meta-llama/Llama-2-7b-hf' \
    --data_path 'commonsense_170k.json' --adapter_name 'dora' --output_dir 'Llama-2-7b-hf-MEMIT_zsre_100_dora' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 --weight_decay 0.0 \
    --use_gradient_checkpointing True --val_set_size 120 --eval_step 80 --save_step 80 \
    --cutoff_len 256 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --dora_simple True \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' --train_on_inputs True


# Lora llama2-7b, unadjusted
python run_dora.py \
    --model_folder_path 'meta-llama_Llama-2-7b-hf' --model_name 'meta-llama/Llama-2-7b-hf' \
    --data_path 'commonsense_170k.json' --adapter_name 'lora' --output_dir 'meta-llama_Llama-2-7b-hf_lora' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 --weight_decay 0.0 \
    --use_gradient_checkpointing True --val_set_size 120 --eval_step 80 --save_step 80 \
    --cutoff_len 256 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --dora_simple True \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' --train_on_inputs True
