#!/bin/bash
set -e

# Constants
DATASETS=("boolq" "openbookqa" "winogrande" "piqa" "ARC-Challenge" "ARC-Easy" "hellaswag" "social_i_qa")
MODEL_NAME="meta-llama/Llama-2-7b-hf"
ADAPTER_NAME="LoRA"
MODEL_PATH="meta-llama_Llama-2-7b-hf"
ADAPTER_PATH="meta-llama_Llama-2-7b-hf_lora"
SAVE_FOLDER_PATH="Llama2_unedited-dora_run-downstream_results"

# running model editing script 
for i in "${!DATASETS[@]}"
do
    task="${DATASETS[$i]}"
    CUDA_LAUNCH_BLOCKING=1 python run_downstream_tasks.py \
        --dataset $task \
        --model_name $MODEL_NAME \
        --adapter_name $ADAPTER_NAME \
        --model_path $MODEL_PATH \
        --adapter_path $ADAPTER_PATH \
        --save_folder_path $SAVE_FOLDER_PATH \
        --batch_size 1
done