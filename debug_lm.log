#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=output/debug_tokenization_llama3.out
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=debug_llama3

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

# Activate environment
source activate memit

# Run the debugging script for Llama
echo "Starting debugging for Llama-3.1-8B..."
python debug_tokenization_logits.py --model_name "meta-llama/Llama-3.1-8B-Instruct"

echo "Debug complete!"