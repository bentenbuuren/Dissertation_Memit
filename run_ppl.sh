#!/bin/bash

#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=llama3_ppl.out
#SBATCH --error=llama3_ppl.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=ppl_i

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit

# running model editing script
python run_ppl.py  \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --model_path '' \
    --adapter_name ' ' \
    --adapter_path ' '
