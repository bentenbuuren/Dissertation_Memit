#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=output/llama3_memit_zsre_100edits.out
#SBATCH --error=output/llama3_memit_zsre_100edits.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=llama3_memit

source activate memit

python -m experiments.summarize --dir_name MEMIT --runs run_076 2>&1 | tee output/llama3_memit_zsre_10_13-17.txt