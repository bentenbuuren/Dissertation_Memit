# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for neural model editing using MEMIT (Mass-Editing Memory in a Transformer), ROME (Rank-One Model Editing), and AlphaEdit methods. The project focuses on editing factual knowledge in transformer models and evaluating the results.

## Key Algorithms

- **MEMIT**: Mass-editing memory approach for editing multiple facts simultaneously
- **ROME**: Rank-one model editing for individual fact updates
- **AlphaEdit**: Alternative editing method

## Core Architecture

### Main Components

- `memit/`: MEMIT algorithm implementation and hyperparameters
- `rome/`: ROME algorithm implementation and utilities
- `alphaedit/`: AlphaEdit algorithm implementation
- `dsets/`: Dataset handling (CounterFact, zsRE, etc.)
- `experiments/`: Evaluation utilities and causal tracing
- `util/`: Shared utilities including model loading and globals
- `baselines/`: Baseline methods including fine-tuning (ft) and MEND

### Key Files

- `run_edit.py`: Main script for model editing and evaluation
- `util/edit_inherit.py`: Contains `model_load()` function for loading models with optional adapters
- `globals.yml`: Configuration paths for results, data, and hyperparameters
- `hparams/`: JSON configuration files for each algorithm and model combination

## Common Commands

### Environment Setup
```bash
# Auto-setup using conda
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh

# Manual setup
conda create --name memit python=3.9 -y
conda activate memit
conda install --name memit --file requirements.txt -y
```

### Model Editing
```bash
# Basic editing and evaluation
python run_edit.py

# Using shell scripts (configure parameters inside script)
bash run_edit.sh
bash run_edit_llama3_memit.sh
bash run_edit_deepseek_rome.sh
```

### Model Fine-tuning
```bash
# LoRA/DoRA fine-tuning
python run_dora.py
bash run_dora.sh
```

### Evaluation
```bash
# Downstream tasks evaluation
python run_downstream_tasks.py
bash run_downstream_tasks.sh

# Perplexity evaluation
python run_ppl.py

# Manual testing
python run_mannual_test.py
```

### Causal Tracing
```bash
# Run causal trace analysis
bash run_causal_trace_llama3.sh
bash run_average_causal.sh
```

## Model Loading

The `model_load()` function in `util/edit_inherit.py` handles model loading:
```python
def model_load(model_path: str, model_name: str = " ", adapter_path: str = " ", adapter_name: str = " ")
```

Parameters:
- `model_name`: Model identifier (e.g., "meta-llama/Llama-2-7b-hf")
- `model_path`: Local path to model (if empty, downloads from HuggingFace)
- `adapter_name`: Type of adapter (e.g., "LoRA", "DoRA")
- `adapter_path`: Path to adapter weights

## Supported Models

- gpt2-xl
- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-3.1-8B-Instruct
- EleutherAI/gpt-j-6B
- deepseek-ai/DeepSeek-R1-Distill-Llama-8B

## Datasets

- **CounterFact**: Factual knowledge for editing
- **zsRE**: Zero-shot relation extraction
- **Commonsense 170k**: Fine-tuning dataset
- **Downstream tasks**: ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, PIQA, etc.

## Results Structure

- `results/MEMIT/`: MEMIT experiment results
- `results/ROME/`: ROME experiment results
- Each run directory contains `params.json` and evaluation JSON files

## HuggingFace Authentication

For Llama models, authentication is required:
```bash
huggingface-cli login --token $YOUR_HF_TOKEN
```

## Important Notes

- The codebase is designed for HPC environments (SLURM scripts included)
- Model editing modifies factual knowledge while preserving model capabilities
- Evaluation includes both editing quality and downstream task performance
- Results are automatically saved with structured naming conventions