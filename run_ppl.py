import fire
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.edit_inherit import model_load, generate_date_string


def run_ppl_calculation(model_name: str, model_path: str, adapter_name: str, adapter_path: str = None):
    # Model name and model_folder will be loaded everytime but not adapter_path

    def tokenize_function(example):
        encoded =  tokenizer(
            example["text"],
            padding=True,
            truncation=True, 
            max_length=256,
            return_tensors="pt")
        return encoded
    
    model, tokenizer = model_load(model_path, model_name, adapter_path, adapter_name)
    model.eval()

    # Load test dataset
    text = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenized_dataset = text.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    data_loader = DataLoader(tokenized_dataset, batch_size=1)

    # Evaluate perplexity
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = batch["input_ids"].squeeze(0)
            if inputs.size(0) < 2:
                continue

            outputs = model(inputs, labels=inputs)
            loss = outputs.loss.item()
            total_loss += loss * (inputs.size(0) - 1)
            total_tokens += inputs.size(0) - 1
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    # Output
    model_prefix = model_name.split("/")[-1]
    adapter_prefix = adapter_name if (adapter_name) and adapter_name != " " else "noadapter"
    with open(f"result_ppl_{model_prefix}_{adapter_prefix}_{generate_date_string()}.txt", "w") as f:
        f.write(str(perplexity))


if __name__ == "__main__":
    fire.Fire(run_ppl_calculation)
