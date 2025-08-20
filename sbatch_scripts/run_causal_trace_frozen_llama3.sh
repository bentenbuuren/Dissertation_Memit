#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=output/llama3_frozen_causal_trace.out
#SBATCH --error=output/llama3_frozen_causal_trace.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=llama3_frozen_causal

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit

python -m experiments.causal_trace_frozen_llama3 \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --noise_level "s3" \
    --num_samples 1209