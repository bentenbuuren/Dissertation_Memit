from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")  # Or your exact 3.1 model name

s = " Bob"
ids = tok(s)["input_ids"]
print(ids)
print([tok.decode([i]) for i in ids])