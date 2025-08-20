#!/bin/bash
#SBATCH --mail-user=btenbuuren1@sheffield.ac.uk

#SBATCH --mail-type=ALL
#SBATCH --output=output_final/ds_mm_mcf_10_1-5.log
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --job-name=ds_mm_mcf_10_1-5

# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit

# Model parameters
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" # meta-llama/Llama-2-7b-hf
MODEL_PATH=""
ADAPTER_NAME=""
ADAPTER_PATH=""
DS_NAME="mcf" # [cf, mcf, zsre]

# Edit parameters 
N_EDITS="10"
ALG_NAMES=("MEMIT")
HPARAMS_FNAMES=("deepseek-ai_DeepSeek-R1-Distill-Llama-8B-1-5.json") # meta-llama_Llama-2-7b-hf.json
EVAL_ONLY=0
MODEL_SAVE=1

# Execute
for i in ${!ALG_NAMES[@]}
do
    alg_name=${ALG_NAMES[$i]}
    hparams_fname=${HPARAMS_FNAMES[$i]}

    echo "Running evals for $alg_name..."

    python3 -m run_edit \
        --alg_name=$alg_name --model_name=$MODEL_NAME --model_path=$MODEL_PATH \
        --adapter_name=$ADAPTER_NAME --adapter_path=$ADAPTER_PATH \
        --hparams_fname=$HPARAMS_FNAMES --num_edits=$N_EDITS --use_cache \
        --dataset_size_limit=$N_EDITS --ds_name=$DS_NAME --eval_only=$EVAL_ONLY \
        --model_save=$MODEL_SAVE 
done
exit 0
 