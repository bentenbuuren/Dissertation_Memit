#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=output/figure3_causal_analysis.out
#SBATCH --error=output/figure3_causal_analysis.err
#SBATCH --partition=cpu  # This doesn't need GPU
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=figure3_analysis

# Load modules
module load Anaconda3/2024.02-1

source activate memit

echo "Creating Figure 3 causal effects chart..."

# Run the Figure 3 analysis for Llama model
python create_causal_chart.py \
    --arch "ns3_r0_meta-llama_Llama-3.1-8B-Instruct" \
    --archname "Llama-3.1-Instruct" \
    --count 1209 \
    --output_dir "analysis_results" \
    --results_dir "results" \
    --max_layers 25

echo "Figure 3 analysis complete! Check analysis_results/ for outputs."