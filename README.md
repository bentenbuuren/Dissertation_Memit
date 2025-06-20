# Edit Inherit

## **Table of Contents**
- [Settings](#settings)
  - [Environment](#1-environment)
  - [Datasets](#2-datasets)
  - [Models](#3-models)
- [Examples](#examples)
  - [Basic Points](#basic-points)
  - [Model Editing](#example-1-model-editing)
  - [Model Finetuning](#example-2-model-finetuning)
  - [Downstream tasks evaluation](#example-3-downstream-tasks-evaluation)
  - [Perplexity Index](#example-4-perplexity-index)
  - [Manual Conversation](#example-5-manual-conversation)
- [Code Reference](#code-reference)
  - [Model Editing](#model-editing)
  - [Model Fine-tuning](#model-fine-tuning)


## **Settings**
### **1. Environment**
#### 1.1. Auto-setup
We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run the following command in the terminal:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```
`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`. The path can be obtained by inputing:
```bash
conda info --base
```
#### 1.2. Manual-setup
For manual setup, we build `conda` environment by executing the following command:
by inputing:
```bash
conda create --name $ENV_NAME python=3.9 -y
conda activate $ENV_NAME
conda install --name $ENV_NAME --file requirements.txt -y
```
`$ENV_NAME` is conda environemnt name, e.g., `memit`.


### **2. Datasets**
#### 2.1. Datasets for finetuning and evaluation
To finetune our models, we use [Commonsense 170k](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json) as our finetuning dataset, which is also used in paper [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353). After downloading, put the downloaded file `commonsense_170k.json` under root directory.

To evaluate the finetuned model, we also use downstream tasks introduced in paper [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353), they can be downloaded [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset). After downloading, organise them as follows:
```
# Folder `dataset` put under root directory
dataset/
├── ARC-Challenge/
├── ARC-Easy/
└── etc./
```
#### 2.2 Datasets for model editing
We use `zsRE` and `CounterFact` datasets for model editing, as introduced in paper [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229). For this stage, we do not need to download the datasets in advance and they are downloaded automatically (refer to `./dsets/counterfact.py` and `./dsets/zsre.py`).

### **3. Models**
For out experiments, we adopt the following models:
 - [gpt2-xl](https://huggingface.co/openai-community/gpt2-xl)
 - [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
 - [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
 
To be notice, for Llama series, there are extra authentication steps:
 - Apply for [model accesses](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/142)
 - Prepare a [Huggingface token](https://huggingface.co/docs/hub/security-tokens)
 - Add login command in bash scripts or input it in the terminal, as follows:
    ```bash
    # Add in bash scripts
    huggingface-cli login --token $YOUR_HF_TOKEN

    # input in terminal and then enter token as it instructs
    huggingface-cli login
    ```

  
## **Examples**
### **Basic points**
#### 1.1. HPC Parameters
The following code was intended for HPC, you can either submit to HPC or just run on your own devices. The first couple of rows in every `.sh` scriptes are settings for HPC, you can edit or just delete it.
```bash
#!/bin/bash
#SBATCH --mail-user=your own email address
#SBATCH --mail-type=ALL
#SBATCH --output=memit_edit.out  
#SBATCH --error=memit_edit.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=64:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=job name
```
#### 1.2. Load model with adapters
As for code, please refere to function `model_load` in `./util/edit_inherit.py`. 
```python
def model_load(model_path: str, model_name: str = " ", adapter_path: str = " ", adapter_name: str = " ")
```
It needs **four** parameters: `model_name`, `model_path`, `adapter_name` and `adapter_path`. `model_name` is the name of the model used. If `model_path` not specified (" " or null), model will be downloaded from Huggingface. If `adapter_name` and `adapter_path` not specofoed, no finetuning adapter shall be loaded, which means the original model will be provided.


### **Example 1. Model Editing**
`run_edit.py` and `run_edit.sh` are scripts we need, it has two functions: **editing** and **evaluating**. 

Users can edit a model and evaluate it or evaluate a saved model without doing editions. The outputs are evaluation of edited model and edited model (optional). Evaluation results are stored under `./results/edit_method(e.g. MEMIT or ROME)`. Input the following commands in the terminal to run editing:
```bash
# Submit to HPC
sbatch run_edit.sh
# Run with nodes or local devices
bash run_edit.sh
```
Below code block demonstrates the parameters for model editing, `MODEL_NAME`, `MODEL_PATH`, `ADAPTER_NAME` and `ADAPTER_PATH` are input parameters for function model_load introduced in [Load model with adapters](#12-load-model-with-adapters). 

`DS_NAME` is the dataset used for editing (e.g. zsRE, CounterFactfor); `N_EDITS` means the number of editings; `ALG_NAMES` represents the algorithm used for model editing (e.g. MEMIT, ROME); `HPARAMS_FNAMES` is the editing parameters with respective to the model. 

`EVAL_ONLY` controls whether the model needs editing: if it is 0, model will be edited first and evaluated then, if it is 1 then it will only be evaluated. `MODEL_SAVE` is whether the edited model will be saved: if it is 0 then the model will not be saved, vice versa.

```bash
# Model parameters
MODEL_NAME="meta-llama/Llama-2-7b-hf"
MODEL_PATH=" "
ADAPTER_NAME=" "
ADAPTER_PATH=" "
DS_NAME="zsre" # [cf, mcf, zsre]

# Edit parameters
N_EDITS="100"
ALG_NAMES=("AlphaEdit")
HPARAMS_FNAMES=("meta-llama_Llama-2-7b-hf.json") # meta-llama_Llama-2-7b-hf.json
EVAL_ONLY=0
MODEL_SAVE=1
```

### **Example 2. Model Finetuning**
`run_dora.py` and `run_dora.sh` are scripts we need, it finetunes the model in LoRA or DoRA. The outputs are evaluation of finetuning adapter. Input the following commands in the terminal to run editing:
```bash
# Submit to HPC
sbatch run_dora.sh
# Run with nodes or local devices
bash run_dora.sh
```
Below code block demonstrates the parameters for model editing, `MODEL_NAME`, `MODEL_PATH`, `ADAPTER_NAME` and `ADAPTER_PATH` are input parameters for function model_load introduced in [Load model with adapters](#12-load-model-with-adapters). The rest parameters are used for finetuning config, users can refer to [DoRA github](https://github.com/NVlabs/DoRA/blob/main/commonsense_reasoning/README.md). Below is an example of dora scipt parameters:
```bash
python run_dora.py \
    --model_folder_path "Llama-2-7b-hf-MEMIT_zsre_100" --model_name 'meta-llama/Llama-2-7b-hf' \
    --data_path 'commonsense_170k.json' --adapter_name 'dora' --output_dir 'Llama-2-7b-hf-MEMIT_zsre_100_dora' \
    --batch_size 16  --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 --weight_decay 0.0 \
    --use_gradient_checkpointing True --val_set_size 120 --eval_step 80 --save_step 80 \
    --cutoff_len 256 --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --dora_simple True \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' --train_on_inputs True
```

### **Example 3. Downstream Tasks Evaluation**
Downstream tasks are used to evaluate the effectiveness of the model after editing/finetuning. There are 8 tasks ("boolq", "piqa", "siqa_ca", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa") evaluating the effectiveness of the model. It is put forward in the paper [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/pdf/2402.09353). `run_downstream_tasks.py` and `run_downstream_tasks.sh` are scripts related. Input the following commands in the terminal to run editing:
```bash
# Submit to HPC
sbatch run_downstream_tasks.sh
# Run with nodes or local devices
bash run_downstream_tasks.sh
```
Below is an example of downstream-task script parameters. `SAVE_FOLDER_PATH` is the output path of the adapter. `DATASETS` is the tasks needed for downstream task evaluation.
```bash
DATASETS=("boolq" "openbookqa" "winogrande" "piqa" "ARC-Challenge" "ARC-Easy" "hellaswag" "social_i_qa")
MODEL_NAME="meta-llama/Llama-2-7b-hf"
ADAPTER_NAME="LoRA"
MODEL_PATH="meta-llama_Llama-2-7b-hf"
ADAPTER_PATH="meta-llama_Llama-2-7b-hf_lora"
SAVE_FOLDER_PATH="Llama2_unedited-dora_run-downstream_results"
```

### **Example 4. Perplexity Index**
Perplexity(PPL) is an index used to evaluate a model's language modeling capabilities. `run_ppl.py` and `run_ppl.sh` are scripts related. 
```bash
python run_ppl.py  \
    --model_name "gpt2-xl" \
    --model_path " " \
    --adapter_name " " \
    --adapter_path " "
```

### **Example 5. Manual Conversation**
`run_mannual_test.py` is the file for mannual check. You can check model's performance by providing a specific question (usually is the concept being edited.) For example, one of the concept is 'The capital city of USA is Washington', it has been updated as 'The capital city of USA is London', then for this part, we can ask 'The capital city of USA is'. The code in `run_mannual_test.py` can be written as:
```python
model_name = "gpt2-xl"
model_folder = "new_model_gpt2-xl_250316"
prompt = "The capital city of USA is"
adapter_folder = "new_model_lora_gpt2-xl_250316"
```


## **Code Reference**
### **Model Editing**
For demos and more information about model editing using MEMIT or ROME, please refer to [MEMIT](https://github.com/kmeng01/memit/blob/main) and [EasyEdit](https://github.com/zjunlp/EasyEdit/tree/main).

### **Model Fine-tuning**
For LoRA, refer to [LoRA](https://github.com/microsoft/LoRA); for DoRA, refer to [DoRA](https://github.com/NVlabs/DoRA).
