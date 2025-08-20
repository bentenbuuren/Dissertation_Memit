#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=output_final/lm_downstream.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=lm_down

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit

# Constants
DATASETS=("boolq" "openbookqa" "winogrande" "piqa" "ARC-Challenge" "ARC-Easy" "hellaswag" "social_i_qa")
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_NAME="Initial"
MODEL_PATH="original_models/meta-llama_Llama-3.1-8B-Instruct"
SAVE_FOLDER_PATH="Llama-3.1-8B-Instruct_run-downstream_results"

# running model editing script 
for i in "${!DATASETS[@]}"
do
    task="${DATASETS[$i]}"
    CUDA_LAUNCH_BLOCKING=1 python run_downstream_tasks.py \
        --dataset $task \
        --model_name $MODEL_NAME \
        --adapter_name $ADAPTER_NAME \
        --model_path $MODEL_PATH \
        --save_folder_path $SAVE_FOLDER_PATH \
        --batch_size 1
done