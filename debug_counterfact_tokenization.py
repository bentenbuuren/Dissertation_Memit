#!/usr/bin/env python3
"""
DEBUG COUNTERFACT TOKENIZATION AND LOGITS

This script provides comprehensive debugging for CounterFact evaluation
with three different logit positioning strategies to identify the correct approach.

Usage:
python debug_counterfact_tokenization.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
python debug_counterfact_tokenization.py --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import json


class CounterFactDebugger:
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize the debugger with a specific model."""
        print(f"🔧 Loading model: {model_name}")
        self.model_name = model_name
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto",
            cache_dir="original_models"
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

    def calculate_prefix_lengths(self, prefixes: List[str]) -> List[int]:
        """Calculate prefix lengths with model-specific adjustments."""
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

    def create_counterfact_batch(self, prefixes: List[str], target_new: str, target_true: str):
        """Create a batch similar to CounterFact evaluation."""
        # Create all prompts (2 per prefix: target_new and target_true)
        all_prompts = []
        for prefix in prefixes:
            all_prompts.append(f"{prefix} {target_new}")
            all_prompts.append(f"{prefix} {target_true}")
        
        print(f"\n📦 CREATING BATCH")
        print(f"Created {len(all_prompts)} prompts from {len(prefixes)} prefixes")
        for i, prompt in enumerate(all_prompts):
            target_type = "NEW" if i % 2 == 0 else "TRUE"
            print(f"  [{i}] {target_type}: '{prompt}'")
        
        # Tokenize the batch
        batch_tokens = self.tokenizer(
            all_prompts,
            padding=True,
            return_tensors="pt"
        )
        
        print(f"Batch shape: {batch_tokens['input_ids'].shape}")
        return batch_tokens, all_prompts

    def debug_three_runs(self, prefixes: List[str], target_new: str, target_true: str):
        """Run three different logit positioning strategies."""
        print("\n" + "="*100)
        print("🎯 RUNNING THREE DIFFERENT LOGIT POSITIONING STRATEGIES")
        print("="*100)
        
        # Get prefix lengths
        prefix_lengths = self.calculate_prefix_lengths(prefixes)
        
        # Create batch and get logits
        batch_tokens, all_prompts = self.create_counterfact_batch(prefixes, target_new, target_true)
        
        # Move to device and get logits
        batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
        
        with torch.no_grad():
            outputs = self.model(**batch_tokens)
            original_logits = outputs.logits
        
        print(f"\n🧠 Original logits shape: {original_logits.shape}")
        
        # Tokenize targets
        target_new_tokens = self.tokenizer(f" {target_new}")["input_ids"]
        target_true_tokens = self.tokenizer(f" {target_true}")["input_ids"]
        
        if self.tokenizer.bos_token_id and target_new_tokens[0] == self.tokenizer.bos_token_id:
            target_new_tokens = target_new_tokens[1:]
        if self.tokenizer.bos_token_id and target_true_tokens[0] == self.tokenizer.bos_token_id:
            target_true_tokens = target_true_tokens[1:]
        
        print(f"Target NEW tokens: {target_new_tokens}")
        print(f"Target TRUE tokens: {target_true_tokens}")
        
        # RUN 1: DeepSeek NO logits adjustment - keeping original alignment
        print(f"\n{'='*60}")
        print(f"🔧 RUN 1: DeepSeek NO logits adjustment - keeping original alignment")
        print(f"{'='*60}")
        self.run_logit_analysis(
            original_logits, 
            batch_tokens, 
            prefix_lengths, 
            target_new_tokens, 
            target_true_tokens,
            all_prompts,
            run_type="RUN1",
            logit_formula="logit_idx = padding_offset + prefix_lens[prefix_idx] + j",
            logits_shifted=False
        )
        
        # RUN 2: DeepSeek NO logits adjustment - keeping original alignment (j-1)
        print(f"\n{'='*60}")
        print(f"🔧 RUN 2: DeepSeek NO logits adjustment - keeping original alignment (j-1)")
        print(f"{'='*60}")
        self.run_logit_analysis(
            original_logits, 
            batch_tokens, 
            prefix_lengths, 
            target_new_tokens, 
            target_true_tokens,
            all_prompts,
            run_type="RUN2",
            logit_formula="logit_idx = padding_offset + prefix_lens[prefix_idx] + j - 1",
            logits_shifted=False
        )
        
        # RUN 3: logits[:, 1:, :] + j-1
        print(f"\n{'='*60}")
        print(f"🔧 RUN 3: logits[:, 1:, :] + j-1")
        print(f"{'='*60}")
        shifted_logits = original_logits[:, 1:, :]
        print(f"Shifted logits shape: {shifted_logits.shape}")
        self.run_logit_analysis(
            shifted_logits, 
            batch_tokens, 
            prefix_lengths, 
            target_new_tokens, 
            target_true_tokens,
            all_prompts,
            run_type="RUN3",
            logit_formula="logits = logits[:, 1:, :]; logit_idx = padding_offset + prefix_lens[prefix_idx] + j - 1",
            logits_shifted=True
        )

    def run_logit_analysis(self, logits, batch_tokens, prefix_lengths, target_new_tokens, target_true_tokens, all_prompts, run_type, logit_formula, logits_shifted):
        """Analyze logit positions for a specific configuration."""
        print(f"\n{run_type} ANALYSIS:")
        print(f"Formula: {logit_formula}")
        print(f"Logits shape: {logits.shape}")
        
        results = []
        
        for i in range(len(all_prompts)):
            prefix_idx = i // 2
            is_target_new = (i % 2 == 0)
            current_target_tokens = target_new_tokens if is_target_new else target_true_tokens
            target_type = "NEW" if is_target_new else "TRUE"
            
            print(f"\n  [{i}] {target_type} - Prefix {prefix_idx}: '{all_prompts[i]}'")
            
            # Find padding offset
            input_ids = batch_tokens['input_ids'][i]
            padding_offset = 0
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            while (padding_offset < len(input_ids) and 
                   input_ids[padding_offset] == pad_token_id):
                padding_offset += 1
            
            print(f"    Padding offset: {padding_offset}")
            print(f"    Prefix length: {prefix_lengths[prefix_idx]}")
            
            # Analyze first token of target
            j = 0
            if run_type == "RUN1":
                logit_idx = padding_offset + prefix_lengths[prefix_idx] + j
            elif run_type == "RUN2":
                logit_idx = padding_offset + prefix_lengths[prefix_idx] + j - 1
            elif run_type == "RUN3":
                logit_idx = padding_offset + prefix_lengths[prefix_idx] + j - 1
            
            print(f"    Calculated logit index: {logit_idx}")
            
            if logit_idx >= 0 and logit_idx < logits.shape[1]:
                # Get model prediction
                predicted_token_id = logits[i, logit_idx, :].argmax().item()
                predicted_token_text = self.tokenizer.decode([predicted_token_id])
                predicted_prob = torch.softmax(logits[i, logit_idx, :], dim=0)[predicted_token_id].item()
                
                print(f"    Model prediction: '{predicted_token_text}' (ID: {predicted_token_id}, prob: {predicted_prob:.4f})")
                
                # Check target probability
                if current_target_tokens:
                    target_token_id = current_target_tokens[0]
                    target_prob = torch.softmax(logits[i, logit_idx, :], dim=0)[target_token_id].item()
                    target_nll = -torch.log(torch.softmax(logits[i, logit_idx, :], dim=0)[target_token_id]).item()
                    target_text = self.tokenizer.decode([target_token_id])
                    
                    print(f"    Target '{target_text}' (ID: {target_token_id}) prob: {target_prob:.4f} (NLL: {target_nll:.4f})")
                    
                    # Check if prediction matches target
                    if predicted_token_id == target_token_id:
                        print(f"    ✅ MATCH: Model predicts target!")
                        match_status = "MATCH"
                    else:
                        print(f"    ❌ MISMATCH: Model predicts different token")
                        match_status = "MISMATCH"
                    
                    results.append({
                        'sequence': i,
                        'target_type': target_type,
                        'predicted': predicted_token_text,
                        'target': target_text,
                        'match': match_status,
                        'target_prob': target_prob,
                        'target_nll': target_nll
                    })
                
                # Show what token is actually at the input position for verification
                actual_input_pos = padding_offset + prefix_lengths[prefix_idx] + j
                if actual_input_pos < len(input_ids):
                    actual_token_id = input_ids[actual_input_pos].item()
                    actual_token_text = self.tokenizer.decode([actual_token_id])
                    print(f"    Token at input pos {actual_input_pos}: '{actual_token_text}' (ID: {actual_token_id})")
                    
            else:
                print(f"    ❌ ERROR: logit_idx {logit_idx} exceeds logits shape")
                results.append({
                    'sequence': i,
                    'target_type': target_type,
                    'error': 'index_out_of_bounds'
                })
        
        # Summary
        if results:
            matches = [r for r in results if r.get('match') == 'MATCH']
            mismatches = [r for r in results if r.get('match') == 'MISMATCH']
            errors = [r for r in results if 'error' in r]
            
            print(f"\n  📊 {run_type} SUMMARY:")
            print(f"    Matches: {len(matches)}/{len(results) - len(errors)}")
            print(f"    Mismatches: {len(mismatches)}/{len(results) - len(errors)}")
            print(f"    Errors: {len(errors)}")
            
            if matches:
                avg_target_prob = np.mean([r['target_prob'] for r in matches])
                avg_target_nll = np.mean([r['target_nll'] for r in matches])
                print(f"    Avg target prob (matches): {avg_target_prob:.4f}")
                print(f"    Avg target NLL (matches): {avg_target_nll:.4f}")

    def run_comprehensive_debug(self):
        """Run the comprehensive debugging with real-world data."""
        print("🚀 STARTING COMPREHENSIVE COUNTERFACT TOKENIZATION & LOGITS DEBUG")
        print(f"Model: {self.model_name}")
        
        # Real-world CounterFact-style data
        prefixes = [
            "The mother tongue of Danielle Darrieux is",
            "The native language of Léon Blum is", 
            "Danielle Darrieux, a native speaker of",
            "The official language of Marie Curie's birthplace is",
        ]
        
        target_new = "English"  # What we want the model to say after editing
        target_true = "French"  # What the model should originally say
        
        print(f"\nUsing target_new: '{target_new}'")
        print(f"Using target_true: '{target_true}'")
        
        # Run all three debugging approaches
        self.debug_three_runs(prefixes, target_new, target_true)
        
        print("\n" + "="*100)
        print("🎯 COMPREHENSIVE DEBUG COMPLETE")
        print("="*100)
        print("\nKey insights to check:")
        print("1. Which run gives the most reasonable predictions?")
        print("2. Which run shows highest target probabilities?")
        print("3. Which run aligns logit positions with actual target tokens?")
        print("4. Are there consistent patterns between NEW and TRUE targets?")


def main():
    parser = argparse.ArgumentParser(description="Debug CounterFact tokenization and logits")
    parser.add_argument("--model_name", type=str, required=True, 
                      help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda", 
                      help="Device to use")
    
    args = parser.parse_args()
    
    debugger = CounterFactDebugger(args.model_name, args.device)
    debugger.run_comprehensive_debug()


if __name__ == "__main__":
    main()