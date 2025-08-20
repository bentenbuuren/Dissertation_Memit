#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=output/llama3_eval_only_baseline.out
#SBATCH --error=output/llama3_eval_only_baseline.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=llama3_eval_only

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit

echo "=========================================="
echo "EVALUATING UNEDITED (BASELINE) MODEL"
echo "This will test what the model knows BEFORE editing"
echo "=========================================="

# Model parameters
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" # meta-llama/Llama-2-7b-hf
MODEL_PATH=""
ADAPTER_NAME=""
ADAPTER_PATH=""
DS_NAME="zsre" # [cf, mcf, zsre] - Set to zsre to test knowledge injection

# Evaluation parameters - KEY CHANGES FOR EVAL-ONLY
N_EDITS="10"  # Number of test cases to evaluate
ALG_NAMES=("MEMIT")  # Algorithm name (required even though we're not editing)
HPARAMS_FNAMES=("meta-llama_Llama-3.1-8B-Instruct.json") # Hyperparameters file
EVAL_ONLY=1  # *** KEY: Only evaluate, don't edit ***
MODEL_SAVE=0 # *** Don't save since we're not editing ***

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DS_NAME"
echo "  Test cases: $N_EDITS"
echo "  Eval only: $EVAL_ONLY"
echo "  Save model: $MODEL_SAVE"
echo ""

# Execute
for i in ${!ALG_NAMES[@]}
do
    alg_name=${ALG_NAMES[$i]}
    hparams_fname=${HPARAMS_FNAMES[$i]}

    echo "Testing baseline model performance with $alg_name hyperparams..."
    echo "This will show what the model knows BEFORE any editing"
    echo ""

    python3 -m run_edit \
        --alg_name=$alg_name --model_name=$MODEL_NAME --model_path=$MODEL_PATH \
        --adapter_name=$ADAPTER_NAME --adapter_path=$ADAPTER_PATH \
        --hparams_fname=$hparams_fname --num_edits=$N_EDITS --use_cache \
        --dataset_size_limit=$N_EDITS --ds_name=$DS_NAME --eval_only=$EVAL_ONLY \
        --model_save=$MODEL_SAVE 

    echo ""
    echo "Baseline evaluation completed!"
    echo "Results saved to: results/MEMIT/run_XXX/"
    echo ""
    echo "Expected results for zsRE baseline:"
    echo "  - post_rewrite_acc: Should be very low (0-10%) - model doesn't know these facts"
    echo "  - post_paraphrase_acc: Should be very low (0-10%) - model doesn't know paraphrases"
    echo "  - post_neighborhood_acc: Should be reasonable (30-50%) - existing knowledge intact"
    echo ""
done

echo "=========================================="
echo "EVAL-ONLY BASELINE TEST COMPLETE"
echo "Check the output files for detailed results"
echo "=========================================="

exit 0