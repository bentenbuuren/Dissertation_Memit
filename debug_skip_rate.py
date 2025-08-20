#!/usr/bin/env python3
"""
Debug script to understand why cases are being skipped
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dsets import KnownsDataset
from util.globals import DATA_DIR
import json

def debug_predictions():
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir="original_models",
        torch_dtype=torch.float16
    ).cuda().eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir="original_models"
    )
    
    # Load dataset
    knowns = KnownsDataset(DATA_DIR)
    print(f"Loaded {len(knowns)} facts")
    
    # Test first 10 facts
    correct_count = 0
    for i, knowledge in enumerate(knowns[:20]):
        prompt = knowledge["prompt"]
        expected = knowledge["attribute"]
        
        # Tokenize and predict
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_token_id = outputs.logits[0, -1, :].argmax().item()
            predicted_text = tokenizer.decode(predicted_token_id).strip()
        
        # Check if prediction matches expectation
        is_correct = predicted_text.lower() == expected.lower()
        if is_correct:
            correct_count += 1
            
        print(f"[{i:2d}] Prompt: '{prompt}'")
        print(f"     Expected: '{expected}'")
        print(f"     Predicted: '{predicted_text}' | {'✓' if is_correct else '✗'}")
        print()
        
        if i >= 9:  # Only test first 10
            break
    
    accuracy = correct_count / 10
    print(f"\nAccuracy on first 10 facts: {accuracy:.1%}")
    print(f"Expected skip rate: {(1-accuracy)*100:.1f}%")
    
    if accuracy < 0.3:
        print("\n🚨 LOW ACCURACY DETECTED!")
        print("Possible issues:")
        print("1. Model doesn't know these specific facts")
        print("2. Prompts are ambiguous or poorly formatted") 
        print("3. Expected answers don't match model's vocabulary")
        print("4. Model needs more context or different prompting")

if __name__ == "__main__":
    debug_predictions()