#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=output/deepseek_causal_trace.out
#SBATCH --error=output/deepseek_causal_trace.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=deepseek_causal

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit

python -m experiments.causal_trace_deepseek \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --noise_level "s3"