# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit

python -m experiments.causal_trace_llama3 \
    --model_name "meta-llama--Llama-3.1-8B-Instruct" \
    --noise_level "s3"