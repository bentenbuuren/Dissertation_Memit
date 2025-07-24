#!/usr/bin/env python3
"""
DEBUG ZSRE TOKENIZATION AND LOGITS

This script provides comprehensive debugging for ZSRE evaluation
with three different logit positioning strategies to identify the correct approach.

Usage:
python debug_zsre_tokenization.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
python debug_zsre_tokenization.py --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import json


class ZSREDebugger:
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

    def debug_target_tokenization(self, targets: List[str]):
        """Debug how targets are tokenized for different models."""
        print("\n" + "="*80)
        print("🎯 TARGET TOKENIZATION ANALYSIS")
        print("="*80)
        
        model_path = self.model_name.lower()
        
        for i, target in enumerate(targets):
            print(f"\nTarget {i}: '{target}'")
            
            # Basic tokenization
            basic_tokens = self.tokenizer(target, padding=True, return_tensors="pt")["input_ids"]
            print(f"  Basic tokenization: {basic_tokens}")
            print(f"  Basic tokens decoded: {[self.tokenizer.decode([t]) for t in basic_tokens[0]]}")
            
            # Model-specific target handling
            if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path:
                print("  🦙 Llama target handling:")
                if basic_tokens.size(1) > 1:
                    target_id = basic_tokens[0, 1].item()  # Take token after BOS
                    print(f"    Using token after BOS: {target_id} ('{self.tokenizer.decode([target_id])}')")
                else:
                    target_id = basic_tokens[0, 0].item()
                    print(f"    Using first token: {target_id} ('{self.tokenizer.decode([target_id])}')")
                    
            elif 'deepseek' in model_path:
                print("  🔧 DeepSeek target handling:")
                if basic_tokens.size(1) > 1 and basic_tokens[0, 0].item() == self.tokenizer.bos_token_id:
                    target_id = basic_tokens[0, 1].item()  # Take token after BOS
                    print(f"    Using token after BOS: {target_id} ('{self.tokenizer.decode([target_id])}')")
                else:
                    target_id = basic_tokens[0, 0].item()
                    print(f"    Using first token: {target_id} ('{self.tokenizer.decode([target_id])}')")
            else:
                print("  ❓ Default target handling:")
                target_id = basic_tokens[0, 0].item()
                print(f"    Using first token: {target_id} ('{self.tokenizer.decode([target_id])}')")
            
            print(f"  Final target ID: {target_id}")

    def create_zsre_batch(self, prompts: List[str]):
        """Create a batch similar to ZSRE evaluation."""
        print(f"\n📦 CREATING ZSRE BATCH")
        print(f"Input prompts ({len(prompts)}):")
        for i, prompt in enumerate(prompts):
            print(f"  [{i}] '{prompt}'")
        
        # Tokenize the batch
        batch_tokens = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        )
        
        print(f"Batch shape: {batch_tokens['input_ids'].shape}")
        print(f"Attention mask shape: {batch_tokens['attention_mask'].shape}")
        
        # Calculate last non-masked positions
        last_non_masked = batch_tokens["attention_mask"].sum(1) - 1
        print(f"Last non-masked positions: {last_non_masked.tolist()}")
        
        return batch_tokens, last_non_masked

    def debug_three_runs_zsre(self, prompts: List[str], targets: List[str]):
        """Run three different logit positioning strategies for ZSRE."""
        print("\n" + "="*100)
        print("🎯 RUNNING THREE DIFFERENT ZSRE LOGIT POSITIONING STRATEGIES")
        print("="*100)
        
        # Debug target tokenization first
        self.debug_target_tokenization(targets)
        
        # Create batch and get logits
        batch_tokens, last_non_masked = self.create_zsre_batch(prompts)
        
        # Move to device and get logits
        batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
        last_non_masked = last_non_masked.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**batch_tokens)
            original_logits = outputs.logits
        
        print(f"\n🧠 Original logits shape: {original_logits.shape}")
        
        # RUN 1: No logits adjustment - using last_non_masked directly
        print(f"\n{'='*60}")
        print(f"🔧 RUN 1: No logits adjustment - using last_non_masked directly")
        print(f"{'='*60}")
        self.run_zsre_analysis(
            original_logits, 
            last_non_masked,
            targets,
            prompts,
            run_type="RUN1",
            description="No logits adjustment, last_non_masked as-is",
            logits_shifted=False,
            position_adjustment=0
        )
        
        # RUN 2: No logits adjustment - using last_non_masked - 1
        print(f"\n{'='*60}")
        print(f"🔧 RUN 2: No logits adjustment - using last_non_masked - 1")
        print(f"{'='*60}")
        self.run_zsre_analysis(
            original_logits, 
            last_non_masked,
            targets,
            prompts,
            run_type="RUN2",
            description="No logits adjustment, last_non_masked - 1",
            logits_shifted=False,
            position_adjustment=-1
        )
        
        # RUN 3: logits[:, 1:, :] + last_non_masked - 1
        print(f"\n{'='*60}")
        print(f"🔧 RUN 3: logits[:, 1:, :] + last_non_masked - 1")
        print(f"{'='*60}")
        shifted_logits = original_logits[:, 1:, :]
        print(f"Shifted logits shape: {shifted_logits.shape}")
        self.run_zsre_analysis(
            shifted_logits, 
            last_non_masked,
            targets,
            prompts,
            run_type="RUN3",
            description="logits[:, 1:, :], last_non_masked - 1",
            logits_shifted=True,
            position_adjustment=-1
        )

    def run_zsre_analysis(self, logits, last_non_masked, targets, prompts, run_type, description, logits_shifted, position_adjustment):
        """Analyze ZSRE logit positions for a specific configuration."""
        print(f"\n{run_type} ANALYSIS:")
        print(f"Description: {description}")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits shifted: {logits_shifted}")
        print(f"Position adjustment: {position_adjustment}")
        
        # Calculate final positions
        if logits_shifted:
            final_positions = last_non_masked - 1 + position_adjustment  # Already account for logit shift
        else:
            final_positions = last_non_masked + position_adjustment
        
        print(f"Final logit positions: {final_positions.tolist()}")
        
        results = []
        
        for i in range(len(prompts)):
            print(f"\n  [{i}] Prompt: '{prompts[i]}'")
            
            logit_pos = final_positions[i].item()
            print(f"    Logit position: {logit_pos}")
            
            if logit_pos >= 0 and logit_pos < logits.shape[1]:
                # Get model prediction at this position
                prediction_logits = logits[i, logit_pos, :]
                predicted_token_id = prediction_logits.argmax().item()
                predicted_token_text = self.tokenizer.decode([predicted_token_id]).strip()
                predicted_prob = torch.softmax(prediction_logits, dim=0)[predicted_token_id].item()
                
                print(f"    Model prediction: '{predicted_token_text}' (ID: {predicted_token_id}, prob: {predicted_prob:.4f})")
                
                # Check all targets
                target_results = []
                for j, target in enumerate(targets):
                    # Get target token ID with model-specific handling
                    target_tokens = self.tokenizer(target, padding=True, return_tensors="pt")["input_ids"]
                    
                    model_path = self.model_name.lower()
                    if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path:
                        target_id = target_tokens[0, 1].item() if target_tokens.size(1) > 1 else target_tokens[0, 0].item()
                    elif 'deepseek' in model_path:
                        if target_tokens.size(1) > 1 and target_tokens[0, 0].item() == self.tokenizer.bos_token_id:
                            target_id = target_tokens[0, 1].item()
                        else:
                            target_id = target_tokens[0, 0].item()
                    else:
                        target_id = target_tokens[0, 0].item()
                    
                    target_prob = torch.softmax(prediction_logits, dim=0)[target_id].item()
                    target_nll = -torch.log(torch.softmax(prediction_logits, dim=0)[target_id]).item()
                    
                    is_match = (predicted_token_id == target_id)
                    match_symbol = "✅" if is_match else "❌"
                    
                    print(f"    Target '{target}' (ID: {target_id}): prob={target_prob:.4f}, NLL={target_nll:.4f} {match_symbol}")
                    
                    target_results.append({
                        'target': target,
                        'target_id': target_id,
                        'prob': target_prob,
                        'nll': target_nll,
                        'is_match': is_match
                    })
                
                # Find best target
                best_target = max(target_results, key=lambda x: x['prob'])
                print(f"    Best target: '{best_target['target']}' (prob: {best_target['prob']:.4f})")
                
                # Check text-based comparison
                predicted_clean = predicted_token_text.lower().strip()
                text_matches = []
                for target in targets:
                    target_clean = target.lower().strip()
                    text_match = predicted_clean == target_clean
                    text_matches.append(text_match)
                    if text_match:
                        print(f"    📝 Text match with '{target}'")
                
                results.append({
                    'prompt_idx': i,
                    'predicted_text': predicted_token_text,
                    'predicted_prob': predicted_prob,
                    'target_results': target_results,
                    'best_target': best_target,
                    'text_matches': text_matches
                })
                
            else:
                print(f"    ❌ ERROR: logit position {logit_pos} out of bounds (logits shape: {logits.shape})")
                results.append({
                    'prompt_idx': i,
                    'error': 'position_out_of_bounds',
                    'logit_pos': logit_pos
                })
        
        # Summary statistics
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            print(f"\n  📊 {run_type} SUMMARY:")
            print(f"    Valid predictions: {len(valid_results)}/{len(results)}")
            
            # Count matches by target
            for target_idx, target in enumerate(targets):
                token_matches = sum(1 for r in valid_results if any(tr['is_match'] and tr['target'] == target for tr in r['target_results']))
                text_matches = sum(1 for r in valid_results if r['text_matches'][target_idx])
                print(f"    '{target}' - Token matches: {token_matches}, Text matches: {text_matches}")
            
            # Average probabilities for each target
            for target in targets:
                target_probs = [tr['prob'] for r in valid_results for tr in r['target_results'] if tr['target'] == target]
                if target_probs:
                    avg_prob = np.mean(target_probs)
                    print(f"    '{target}' avg probability: {avg_prob:.4f}")
            
            # Best target distribution
            best_targets = [r['best_target']['target'] for r in valid_results]
            for target in targets:
                count = best_targets.count(target)
                print(f"    '{target}' was best target: {count}/{len(valid_results)} times")

    def run_comprehensive_debug(self):
        """Run the comprehensive debugging with real-world data."""
        print("🚀 STARTING COMPREHENSIVE ZSRE TOKENIZATION & LOGITS DEBUG")
        print(f"Model: {self.model_name}")
        
        # Real-world ZSRE-style data
        # These are knowledge-based prompts where we expect specific factual answers
        prompts = [
            "The capital of France is",
            "The CEO of Apple is", 
            "The author of Harry Potter is",
            "The currency of Japan is",
            "The largest planet in our solar system is"
        ]
        
        # Expected answers (targets)
        targets = ["Paris", "Tim Cook", "J.K. Rowling", "Yen", "Jupiter"]
        
        print(f"\nUsing {len(prompts)} prompts with {len(targets)} possible targets")
        print("Prompts:")
        for i, prompt in enumerate(prompts):
            print(f"  {i}: '{prompt}'")
        print("Targets:")
        for i, target in enumerate(targets):
            print(f"  {i}: '{target}'")
        
        # Run all three debugging approaches
        self.debug_three_runs_zsre(prompts, targets)
        
        print("\n" + "="*100)
        print("🎯 COMPREHENSIVE ZSRE DEBUG COMPLETE")
        print("="*100)
        print("\nKey insights to check:")
        print("1. Which run gives the most reasonable predictions?")
        print("2. Which run shows highest target probabilities?")
        print("3. Which run aligns logit positions with expected factual answers?")
        print("4. Are the text-based matches consistent with token-based matches?")
        print("5. Does the model show knowledge of the facts being tested?")


def main():
    parser = argparse.ArgumentParser(description="Debug ZSRE tokenization and logits")
    parser.add_argument("--model_name", type=str, required=True, 
                      help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda", 
                      help="Device to use")
    
    args = parser.parse_args()
    
    debugger = ZSREDebugger(args.model_name, args.device)
    debugger.run_comprehensive_debug()


if __name__ == "__main__":
    main()