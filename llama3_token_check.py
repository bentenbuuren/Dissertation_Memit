from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")  # Or your exact 3.1 model name

s = " Bob"
ids = tok(s)["input_ids"]
print(ids)
print([tok.decode([i]) for i in ids])