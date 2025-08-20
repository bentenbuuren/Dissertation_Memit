from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", cache_dir="original_models")

print("=== MLP Component Shapes ===")
print(f"gate_proj: {model.model.layers[3].mlp.gate_proj.weight.shape}")
print(f"up_proj: {model.model.layers[3].mlp.up_proj.weight.shape}")
print(f"down_proj: {model.model.layers[3].mlp.down_proj.weight.shape}")