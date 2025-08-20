from os import walk
from os.path import join

from util.edit_inherit import model_load


def chat_with_model(model, tokenizer, prompt, max_length=1000):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to("cuda")
    output = model.generate(
        input_ids=input_ids, 
        max_length=max_length, 
        pad_token_id=tokenizer.eos_token_id)
    # response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Param setting
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_folder = ""
prompt = "What is the tallest building in the world?"
adapter_folder = None

# Run question
model, tokenizer = model_load(model_folder, model_name, adapter_folder)
model = model.to("cuda")
model.eval()
res = chat_with_model(model, tokenizer, prompt)
print(res)
