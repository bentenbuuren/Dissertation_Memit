from transformers import AutoTokenizer

def test_tokenizer(model_name, label):
    print(f"\n{'='*60}")
    print(f"TESTING {label}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir="original_models", 
            trust_remote_code=True
        )
        
        # Test the same tokens from your debug
        sega_tokens = tok(" Sega")["input_ids"]
        bbc_tokens = tok(" BBC")["input_ids"]
        
        print(f"' Sega' tokens: {sega_tokens}")
        print(f"' BBC' tokens: {bbc_tokens}")
        print(f"Decoded Sega: {[tok.decode([t]) for t in sega_tokens]}")
        print(f"Decoded BBC: {[tok.decode([t]) for t in bbc_tokens]}")
        
        # Test special tokens
        print(f"\nSpecial tokens:")
        print(f"BOS token: {tok.bos_token} (ID: {tok.bos_token_id})")
        print(f"EOS token: {tok.eos_token} (ID: {tok.eos_token_id})")
        print(f"PAD token: {tok.pad_token} (ID: {tok.pad_token_id})")
        
        # Test the exact tokens from debug output
        print(f"\nTesting specific token IDs:")
        if len(sega_tokens) > 0:
            print(f"First Sega token ({sega_tokens[0]}): '{tok.decode([sega_tokens[0]])}'")
        if len(sega_tokens) > 1:
            print(f"Second Sega token ({sega_tokens[1]}): '{tok.decode([sega_tokens[1]])}'")
        if len(bbc_tokens) > 0:
            print(f"First BBC token ({bbc_tokens[0]}): '{tok.decode([bbc_tokens[0]])}'")
        if len(bbc_tokens) > 1:
            print(f"Second BBC token ({bbc_tokens[1]}): '{tok.decode([bbc_tokens[1]])}'")
            
        # Test if model path contains 'llama'
        print(f"\nModel path analysis:")
        print(f"Contains 'llama': {'llama' in model_name.lower()}")
        print(f"Contains 'deepseek': {'deepseek' in model_name.lower()}")
        
        return True
        
    except Exception as e:
        print(f"ERROR loading {model_name}: {e}")
        return False

# Test both models
models_to_test = [
    ("meta-llama/Llama-3.1-8B-Instruct", "LLAMA 3.1"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DEEPSEEK")
]

print("TOKENIZER COMPARISON TEST")
print("=" * 80)

results = {}
for model_name, label in models_to_test:
    success = test_tokenizer(model_name, label)
    results[label] = success

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
for label, success in results.items():
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"{label}: {status}")

print(f"\n🔍 Key differences to look for:")
print("1. Different token IDs for same text")
print("2. Different BOS/EOS token handling") 
print("3. Different number of tokens for same input")
print("4. Special token differences")