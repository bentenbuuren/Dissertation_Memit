#!/bin/bash

#SBATCH --mail-user=ycheng80@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=lora_run.out
#SBATCH --error=lora_run.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=ppl_i

# Load modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate memit

# running model editing script
python run_ppl.py  \
    --model_name 'gpt2-xl' \
    --model_path ' ' \
    --adapter_name ' ' \
    --adapter_path ' '
