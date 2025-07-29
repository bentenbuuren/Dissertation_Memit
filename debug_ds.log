#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=output/debug_tokenization_deepseek.out
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=debug_deepseek

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

# Activate environment
source activate memit

# Run the debugging script for DeepSeek
echo "Starting debugging for DeepSeek-R1-Distill-Llama-8B..."
python debug_tokenization_logits.py --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

echo "Debug complete!"