import gc
import sys
import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset


from peft_local.src.peft import (
    LoraConfig,
    DoraConfig,
    prepare_model_for_int8_training,
    get_peft_model,
    get_peft_model_state_dict)
from util.edit_inherit import Prompt4Lora, model_load


def run_finetune(
    model_folder_path: str,
    model_name: str,
    data_path: str = "commonsense_170k.json",
    adapter_name: str = "lora",
    # training parameters
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    use_gradient_checkpointing: bool = False,
    val_set_size: int = 2000,
    eval_step: int = 200,
    save_step: int = 200,
    cutoff_len: int = 256,
    # lora parameters
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None,
    # dora parameters
    dora_simple: bool = True,
    Wdecompose_target_modules: List[str] = None,
    # llm parameters
    train_on_inputs: bool = True,
    resume_from_checkpoint: str = None
    ):
    # Set up model and tokenizer
    model, tokenizer = model_load(model_folder_path, model_name)
    model = prepare_model_for_int8_training(
         model, use_gradient_checkpointing=use_gradient_checkpointing)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    # Initiate params
    gradient_accumulation_steps = batch_size // micro_batch_size
    prompt_cfg = Prompt4Lora(tokenizer, cutoff_len, model_name, train_on_inputs)
    output_dir = f"{model_folder_path}_{adapter_name}"

    if adapter_name.lower() == "lora":
        config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
    else:
        config = DoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            dora_simple=dora_simple,
            Wdecompose_target_modules=Wdecompose_target_modules
            )

    model = get_peft_model(model, config)
    data = load_dataset("json", data_files=os.path.join(os.getcwd(), data_path))
    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(prompt_cfg.generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(prompt_cfg.generate_and_tokenize_prompt)
        )
        del train_val
    else:
        train_data = data["train"].shuffle().map(prompt_cfg.generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=None,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    fire.Fire(run_finetune)