#!/usr/bin/env python3
"""
DEBUG TOKENIZATION AND LOGITS

This script provides comprehensive debugging for tokenization and logit calculations
in model editing evaluation. It helps identify issues with position calculations,
token alignments, and model-specific handling.

Usage:
python debug_tokenization_logits.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
python debug_tokenization_logits.py --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import json


class TokenizationLogitsDebugger:
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize the debugger with a specific model."""
        print(f"🔧 Loading model: {model_name}")
        self.model_name = model_name
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", cache_dir = "original_models"
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✅ Model loaded successfully")
        print(f"   Vocab size: {len(self.tokenizer)}")
        print(f"   BOS token: {self.tokenizer.bos_token} (ID: {self.tokenizer.bos_token_id})")
        print(f"   EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        print(f"   PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
    
    def debug_basic_tokenization(self):
        """Debug basic tokenization behavior."""
        print("\n" + "="*80)
        print("🔤 BASIC TOKENIZATION DEBUG")
        print("="*80)
        
        test_cases = [
            "English",
            " English", 
            "French",
            " French",
            "The mother tongue of Danielle Darrieux is",
            "The mother tongue of Léon Blum is",
            "Danielle Darrieux, a native",
        ]
        
        for text in test_cases:
            tokens = self.tokenizer(text)["input_ids"]
            decoded = self.tokenizer.decode(tokens)
            
            print(f"\nText: '{text}'")
            print(f"  Tokens: {tokens}")
            print(f"  Length: {len(tokens)}")
            print(f"  Decoded: '{decoded}'")
            
            # Show individual token breakdowns
            print("  Token breakdown:")
            for i, token_id in enumerate(tokens):
                token_text = self.tokenizer.decode([token_id])
                print(f"    [{i}] {token_id} → '{token_text}'")
    
    def debug_batch_tokenization(self, prefixes: List[str], targets: List[str]):
        """Debug batch tokenization with padding."""
        print("\n" + "="*80)
        print("🔤 BATCH TOKENIZATION DEBUG")
        print("="*80)
        
        # Create full prompts (prefix + target combinations)
        full_prompts = []
        for prefix in prefixes:
            for target in targets:
                full_prompts.append(f"{prefix} {target}")
        
        print(f"Number of prompts: {len(full_prompts)}")
        print(f"Sample prompts:")
        for i, prompt in enumerate(full_prompts[:4]):
            print(f"  [{i}] '{prompt}'")
        
        # Tokenize with padding
        batch_tokens = self.tokenizer(
            full_prompts, 
            padding=True, 
            return_tensors="pt"
        )
        
        print(f"\nBatch shape: {batch_tokens['input_ids'].shape}")
        print(f"Attention mask shape: {batch_tokens['attention_mask'].shape}")
        
        # Analyze each sequence in the batch
        for i in range(min(4, len(full_prompts))):
            print(f"\n--- Sequence {i}: '{full_prompts[i]}' ---")
            
            input_ids = batch_tokens['input_ids'][i]
            attention_mask = batch_tokens['attention_mask'][i]
            
            # Count padding
            padding_count = (input_ids == self.tokenizer.pad_token_id).sum().item()
            content_length = len(input_ids) - padding_count
            
            print(f"  Total length: {len(input_ids)}")
            print(f"  Padding tokens: {padding_count}")
            print(f"  Content length: {content_length}")
            
            # Find where content starts (skip padding)
            content_start = 0
            while (content_start < len(input_ids) and 
                   input_ids[content_start] == self.tokenizer.pad_token_id):
                content_start += 1
            
            print(f"  Content starts at: {content_start}")
            
            # Show actual tokens
            actual_tokens = input_ids[content_start:]
            print(f"  Actual tokens: {actual_tokens.tolist()}")
            print(f"  Decoded: '{self.tokenizer.decode(actual_tokens)}'")
            
            # Token-by-token breakdown
            print("  Token breakdown:")
            for j, token_id in enumerate(actual_tokens):
                if j >= 10:  # Limit output
                    print("    ...")
                    break
                token_text = self.tokenizer.decode([token_id])
                global_pos = content_start + j
                print(f"    [{global_pos:2d}] {token_id} → '{token_text}'")
        
        return batch_tokens
    
    def calculate_prefix_lengths(self, prefixes: List[str]):
        """Calculate prefix lengths with model-specific handling."""
        print("\n" + "="*80)
        print("📏 PREFIX LENGTH CALCULATION")
        print("="*80)
        
        model_path = self.model_name.lower()
        
        prefix_lengths = []
        
        for i, prefix in enumerate(prefixes):
            # Basic tokenization
            tokens = self.tokenizer(prefix)["input_ids"]
            basic_length = len(tokens)
            
            print(f"\nPrefix {i}: '{prefix}'")
            print(f"  Basic tokens: {tokens}")
            print(f"  Basic length: {basic_length}")
            
            # Model-specific adjustments
            if 'deepseek' in model_path:
                print("  🔧 DeepSeek-specific handling:")
                
                # Check for BOS token
                has_bos = tokens[0] == self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else False
                print(f"    Has BOS token: {has_bos}")
                
                if has_bos:
                    adjusted_length = basic_length - 1
                    print(f"    Adjusted length (removed BOS): {adjusted_length}")
                else:
                    adjusted_length = basic_length
                    print(f"    Adjusted length (no BOS): {adjusted_length}")
                    
            elif any(x in model_path for x in ['llama-3.1', 'llama-3', 'llama-2']):
                print("  🦙 Llama-specific handling:")
                adjusted_length = basic_length - 1  # Always remove BOS for Llama
                print(f"    Adjusted length (removed BOS): {adjusted_length}")
                
            else:
                print("  ❓ Unknown model type:")
                adjusted_length = basic_length
                print(f"    Using basic length: {adjusted_length}")
            
            prefix_lengths.append(adjusted_length)
            print(f"  Final prefix length: {adjusted_length}")
        
        return prefix_lengths
    
    def debug_logit_positions(self, prefixes: List[str], targets: List[str]):
        """Debug logit position calculations."""
        print("\n" + "="*80)
        print("🎯 LOGIT POSITION CALCULATION DEBUG")
        print("="*80)
        
        # Get prefix lengths
        prefix_lengths = self.calculate_prefix_lengths(prefixes)
        
        # Create batch
        batch_tokens = self.debug_batch_tokenization(prefixes, targets)
        
        # Get model logits
        print("\n🧠 RUNNING MODEL FORWARD PASS...")
        with torch.no_grad():
            batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
            outputs = self.model(**batch_tokens)
            logits = outputs.logits
        
        print(f"Raw logits shape: {logits.shape}")
        
        # Model-specific logits adjustment
        model_path = self.model_name.lower()
        if any(x in model_path for x in ['llama-3.1', 'llama-3', 'llama-2']):
            print("🦙 Applying Llama logits adjustment: removing first position")
            logits = logits[:, 1:, :]
            print(f"Adjusted logits shape: {logits.shape}")
        elif 'deepseek' in model_path:
            print("🔧 DeepSeek: No logits adjustment")
        
        # Analyze position calculations for each sequence
        for i in range(min(4, len(prefixes) * len(targets))):
            prefix_idx = i // len(targets)
            target_idx = i % len(targets)
            
            print(f"\n--- Sequence {i}: Prefix {prefix_idx}, Target {target_idx} ---")
            print(f"  Prefix: '{prefixes[prefix_idx]}'")
            print(f"  Target: '{targets[target_idx]}'")
            print(f"  Prefix length: {prefix_lengths[prefix_idx]}")
            
            # Calculate position
            j = 0  # First token of target
            
            # Find padding offset
            input_ids = batch_tokens['input_ids'][i]
            padding_offset = 0
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            while (padding_offset < len(input_ids) and 
                   input_ids[padding_offset] == pad_token_id):
                padding_offset += 1
            
            print(f"  Padding offset: {padding_offset}")
            
            # Calculate logit index
            if 'deepseek' in model_path:
                logit_idx = padding_offset + prefix_lengths[prefix_idx] + j
            else:  # Llama
                logit_idx = padding_offset + prefix_lengths[prefix_idx] + j
            
            print(f"  Calculated logit index: {logit_idx}")
            
            # Verify this position makes sense
            if logit_idx < logits.shape[1]:
                # Get the actual token at this position in input
                actual_pos_in_input = padding_offset + prefix_lengths[prefix_idx] + j
                if actual_pos_in_input < len(input_ids):
                    actual_token_id = input_ids[actual_pos_in_input].item()
                    actual_token_text = self.tokenizer.decode([actual_token_id])
                    print(f"  Token at input position {actual_pos_in_input}: '{actual_token_text}' (ID: {actual_token_id})")
                
                # Get model's prediction at this logit position
                predicted_token_id = logits[i, logit_idx, :].argmax().item()
                predicted_token_text = self.tokenizer.decode([predicted_token_id])
                predicted_prob = torch.softmax(logits[i, logit_idx, :], dim=0)[predicted_token_id].item()
                
                print(f"  Model prediction at logit[{i}, {logit_idx}]: '{predicted_token_text}' (ID: {predicted_token_id}, prob: {predicted_prob:.4f})")
                
                # Check target probabilities
                target_token_ids = self.tokenizer(f" {targets[target_idx]}")["input_ids"]
                if self.tokenizer.bos_token_id and target_token_ids[0] == self.tokenizer.bos_token_id:
                    target_token_ids = target_token_ids[1:]  # Remove BOS
                
                if target_token_ids:
                    target_token_id = target_token_ids[0]
                    target_prob = torch.softmax(logits[i, logit_idx, :], dim=0)[target_token_id].item()
                    target_nll = -torch.log(torch.softmax(logits[i, logit_idx, :], dim=0)[target_token_id]).item()
                    
                    print(f"  Target '{targets[target_idx]}' probability: {target_prob:.4f} (NLL: {target_nll:.4f})")
                    
                    # Check if prediction matches target
                    if predicted_token_id == target_token_id:
                        print(f"  ✅ MATCH: Model predicts target!")
                    else:
                        print(f"  ❌ MISMATCH: Model predicts different token")
            else:
                print(f"  ❌ ERROR: logit_idx {logit_idx} exceeds logits shape {logits.shape}")
    
    def run_comprehensive_debug(self):
        """Run all debugging functions with sample data."""
        print("🚀 STARTING COMPREHENSIVE TOKENIZATION & LOGITS DEBUG")
        print(f"Model: {self.model_name}")
        
        # Sample data similar to CounterFact evaluation
        prefixes = [
            "The mother tongue of Danielle Darrieux is",
            "The mother tongue of Léon Blum is", 
            "Danielle Darrieux, a native",
            "The native language of Montesquieu is",
        ]
        
        targets = ["English", "French"]
        
        # Run all debug functions
        self.debug_basic_tokenization()
        self.debug_batch_tokenization(prefixes, targets)
        self.debug_logit_positions(prefixes, targets)
        
        print("\n" + "="*80)
        print("🎯 DEBUG COMPLETE")
        print("="*80)
        print("\nKey things to check:")
        print("1. Are prefix lengths calculated correctly?")
        print("2. Do logit indices point to the right positions?")
        print("3. Are model predictions reasonable?")
        print("4. Do target probabilities make sense?")


def main():
    parser = argparse.ArgumentParser(description="Debug tokenization and logits for model editing evaluation")
    parser.add_argument("--model_name", required=True, help="Model name or path")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create debugger and run
    debugger = TokenizationLogitsDebugger(args.model_name, args.device)
    debugger.run_comprehensive_debug()


if __name__ == "__main__":
    main()