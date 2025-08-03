#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=output/average_causal_analysis.out
#SBATCH --error=output/average_causal_analysis.err
#SBATCH --partition=cpu  # This doesn't need GPU
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=causal_analysis

# Load modules
module load Anaconda3/2024.02-1

source activate memit

# Run the analysis
python average_causal_effects.py \
    --arch "ns3_r0_meta-llama_Llama-3.1-8B-Instruct" \
    --archname "Llama-3.1-Instruct" \
    --count 1209 \
    --output_dir "analysis_results" \
    --results_dir "results"

echo "Analysis complete! Check analysis_results/ for outputs."