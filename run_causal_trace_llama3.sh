#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=output/llama_causal_trace.out
#SBATCH --error=output/llama_causal_trace.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=llama_causal

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit

python -m experiments.causal_trace_llama3 \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --noise_level "s3"