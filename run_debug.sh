#!/bin/bash
# Load modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

source activate memit
# Debug Runner Script for Tokenization and Logit Analysis
# This script runs both CounterFact and ZSRE debugging for Llama3 and DeepSeek models

echo "🚀 Starting Comprehensive Tokenization and Logit Debugging"
echo "=========================================================="

# Model configurations
LLAMA_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Create output directory
mkdir -p debug_outputs

echo ""
echo "🦙 DEBUGGING LLAMA MODEL"
echo "========================"

echo ""
echo "📊 Running CounterFact debugging for Llama..."
python debug_counterfact_tokenization.py --model_name "$LLAMA_MODEL" > debug_outputs/llama_counterfact_debug.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ CounterFact debugging completed for Llama"
else
    echo "❌ CounterFact debugging failed for Llama"
fi

echo ""
echo "📊 Running ZSRE debugging for Llama..."
python debug_zsre_tokenization.py --model_name "$LLAMA_MODEL" > debug_outputs/llama_zsre_debug.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ ZSRE debugging completed for Llama"
else
    echo "❌ ZSRE debugging failed for Llama"
fi

echo ""
echo "🔧 DEBUGGING DEEPSEEK MODEL"
echo "==========================="

echo ""
echo "📊 Running CounterFact debugging for DeepSeek..."
python debug_counterfact_tokenization.py --model_name "$DEEPSEEK_MODEL" > debug_outputs/deepseek_counterfact_debug.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ CounterFact debugging completed for DeepSeek"
else
    echo "❌ CounterFact debugging failed for DeepSeek"
fi

echo ""
echo "📊 Running ZSRE debugging for DeepSeek..."
python debug_zsre_tokenization.py --model_name "$DEEPSEEK_MODEL" > debug_outputs/deepseek_zsre_debug.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ ZSRE debugging completed for DeepSeek"
else
    echo "❌ ZSRE debugging failed for DeepSeek"
fi

echo ""
echo "🎯 DEBUGGING COMPLETE"
echo "===================="
echo ""
echo "Output files generated:"
echo "  - debug_outputs/llama_counterfact_debug.log"
echo "  - debug_outputs/llama_zsre_debug.log"
echo "  - debug_outputs/deepseek_counterfact_debug.log"
echo "  - debug_outputs/deepseek_zsre_debug.log"
echo ""
echo "📋 Analysis Summary:"
echo "1. Check each log file to see which RUN (1, 2, or 3) gives the best results"
echo "2. Look for:"
echo "   - Highest target probabilities"
echo "   - Most matches between predicted and target tokens"
echo "   - Reasonable model predictions"
echo "   - Consistent behavior across prompts"
echo ""
echo "💡 Next Steps:"
echo "1. Review the logs to identify the correct approach for each model"
echo "2. Update your eval_utils_counterfact.py and eval_utils_zsre.py accordingly"
echo "3. Test with a small batch of real edits to confirm the fixes work"