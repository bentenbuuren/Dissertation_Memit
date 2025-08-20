import ast
import json
import os
import re
from abc import ABC, abstractmethod
from datetime import date, datetime
from os import walk
from os.path import join
from typing import List, Dict

import pandas as pd
import torch
from datasets import load_dataset
from peft_local import PeftModel as PeftModel_Local
from transformers import AutoModelForCausalLM, AutoTokenizer


class Prompt4Lora():
    def __init__(self, tokenizer, cutoff_len, model_name, train_on_inputs):
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.model_name = model_name
        self.train_on_inputs = train_on_inputs

    def tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            if "chatglm" not in self.model_name:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in self.model_name:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    @staticmethod
    def generate_prompt(data_point):
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {data_point["instruction"]}
                    
                    ### Input:
                    {data_point["input"]}
                    
                    ### Response:
                    {data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                    ### Instruction:
                    {data_point["instruction"]}
                    
                    ### Response:
                    {data_point["output"]}"""
    
    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.generate_prompt(data_point)
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = self.generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt
    

def model_load(model_path: str, model_name: str = " ", adapter_path: str = " ", adapter_name: str = " "): 
    def keyword_detector(keyword, folder_path):
        if (not folder_path) or (not os.path.exists(folder_path)): 
            return False
        for filename in os.listdir(folder_path):
            if keyword.lower() in filename.lower():  # case-insensitive match
                return True
        return False
    
    def parse_args():
        final_model_path, final_tokenizer_path = model_name, model_name
        load_adapter_flag = False
        if model_path and os.path.exists(model_path):
            final_model_path = model_path
            if keyword_detector("tokenizer.json", model_path):
                final_tokenizer_path = model_path
                load_adapter_flag = False
        if adapter_path and adapter_name:
            if os.path.exists(adapter_path) and adapter_name.lower() in ["dora", "lora"]:
                load_adapter_flag = True
        return final_model_path, final_tokenizer_path, load_adapter_flag

    # Load model and tokenizer
    model_path_fnl, token_path_fnl, adapter_flag = parse_args()
    model = AutoModelForCausalLM.from_pretrained(model_path_fnl, cache_dir="original_models", torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(token_path_fnl, cache_dir="original_models", trust_remote_code=True)

    # Load adapter if adapter is DoRA and adapter_path is specified
    if adapter_flag:
        model = PeftModel_Local.from_pretrained(
            model, 
            adapter_path,
            torch_dtype=torch.float16)
        key_list = [(key, module) for key, module in model.model.named_modules()]
        for key, module in key_list:
            if isinstance(model.peft_config.target_modules, str):
                target_module_found = re.fullmatch(model.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.target_modules)

            if adapter_name == "dora":
                if model.peft_config.Wdecompose_target_modules != None:
                    if isinstance(model.peft_config.Wdecompose_target_modules, str):
                        wdecompose_target_module_found = re.fullmatch(model.peft_config.Wdecompose_target_modules, key)
                    else:
                        wdecompose_target_module_found = any(key.endswith(target_key) for target_key in model.peft_config.Wdecompose_target_modules)
                else: 
                    wdecompose_target_module_found = False
            else:
                wdecompose_target_module_found = False

            if target_module_found:
                module.merge_weights = True
                module.train(mode=False)
            elif wdecompose_target_module_found:
                module.merge_weights = True
                module.train(mode=False)
    return model, tokenizer
    

def generate_date_string(date_format: str="%y%m%d"):
    now = datetime.now()
    return now.strftime(date_format)


class DownloadCFRequests(ABC):   
    """
    Basic class about CounterFact dataset, contains method to download dataset from HuggingFace
    """
    def __init__(self, row_num: int=0):
        self.row_num = row_num

    def download_data(self):
        splits = {
            'train': 'data/train-00000-of-00001-05d11247db7abce8.parquet', 
            'test': 'data/test-00000-of-00001-bacb83500fca49a9.parquet'
            }
        df = pd.read_parquet("hf://datasets/azhx/counterfact/" + splits["train"])
        if self.row_num:
            if self.row_num >= df.shape[0]:
                return df
            df = df.sample(n=self.row_num, replace=False) 
        return df
        
    @staticmethod
    def gen_request(json_dict: Dict) -> Dict:
        request = dict()
        request["prompt"] = json_dict.get("prompt")
        request["subject"] = json_dict.get("subject")
        request["target_new"] = {"str": json_dict.get("target_new").get("str")}
        return request
    
    @abstractmethod
    def proc(self):
        pass


class GenerateCFRequests(DownloadCFRequests):
    """
    Generate edit requests for CounterFact dataset and record adopted concepts to a csv file
    """
    def __init__(self, row_num: int=None, model_name: str=""):
        self.row_num = row_num
        self.model_name = model_name
        self.sample_folder = "sample_records"

    def record_edit_copcepts(self, df: pd.DataFrame, folder_name: str="sample_records"):
        os.makedirs(folder_name, exist_ok=True)
        full_path = join(folder_name, f"sampled_edits_{self.model_name}_{date.today()}.csv")
        print("full_path: ", full_path)
        df["case_id"].to_csv(full_path, index=False)

    def proc(self):
        requests = list()
        df = self.download_data()
        self.record_edit_copcepts(df)
        for json_dict in df["requested_rewrite"].to_list():
            requests.append(self.gen_request(json_dict))
        return requests
    

class RetrieveCFRequest(DownloadCFRequests):
    """
    Generate edit requests according to the saved records
    """
    def __init__(self, record_path: str):
        self.record_path = record_path
        self.row_num = 0
        self.dataset_name = "azhx/counterfact"
    
    def download_data(self):
        return load_dataset(self.dataset_name)["train"]

    @staticmethod
    def retrieve_data(df: any, idx_list: list):
        return df.filter(lambda x: x['case_id'] in idx_list)
    
    def proc(self):
        idx_list = pd.read_csv(self.record_path)["case_id"].to_list()
        df = self.retrieve_data(self.download_data(), idx_list)
        return df

