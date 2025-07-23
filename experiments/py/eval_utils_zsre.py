"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.

FIXED VERSION: Includes proper tokenization handling for DeepSeek and Llama models.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    print(f"\n=== ZSRE EVALUATION DEBUG ===")
    print(f"Subject: {subject}")
    print(f"Target New: {target_new['str']}")
    print(f"Target True: {target_true['str']}")
    print(f"Rewrite prompts: {len(rewrite_prompts)}")
    print(f"Paraphrase prompts: {len(paraphrase_prompts)}")
    print(f"Neighborhood prompts: {len(neighborhood_prompts)}")

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    
    # Flatten all the evaluated prefixes into one list.
    target_tok = tok(" " + target_new["str"])["input_ids"]
    
    # FIXED: Apply model-specific token adjustments
    model_path = model.config._name_or_path.lower()
    if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path:
        print(f"🦙 LLAMA MODEL DETECTED - Removing BOS token from target")
        target_tok = target_tok[1:]
    elif 'deepseek' in model_path:
        print(f"🔧 DEEPSEEK MODEL DETECTED - Removing BOS token from target")
        target_tok = target_tok[1:]
    else:
        print(f"❓ UNKNOWN MODEL TYPE - No target token adjustments")
    
    print(f"Target tokens after adjustment: {target_tok}")
    print(f"Target tokens decoded: {[tok.decode([t]) for t in target_tok]}")

    inp_prompts_og = list(chain(*prob_prompts))
    
    # FIXED: Apply model-specific prompt construction
    if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path:
        inp_prompts = [
            el + tok.decode(target_tok[:i]) if i == 0 else el + ' ' + tok.decode(target_tok[:i])
            for el in inp_prompts_og
            for i in range(len(target_tok))
        ]
    elif 'deepseek' in model_path:
        print(f"🔧 DEEPSEEK MODEL DETECTED - Using SAME prompt construction as Llama for consistency")
        inp_prompts = [
            el + tok.decode(target_tok[:i]) if i == 0 else el + ' ' + tok.decode(target_tok[:i])
            for el in inp_prompts_og
            for i in range(len(target_tok))
        ]
    else:
        # Default behavior for unknown models
        inp_prompts = [
            el + tok.decode(target_tok[:i])
            for el in inp_prompts_og
            for i in range(len(target_tok))
        ]
    
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    print(f"Number of input prompts: {len(inp_prompts)}")
    print(f"Number of targets: {len(inp_targets)}")
    print(f"Sample input prompt: '{inp_prompts[0] if inp_prompts else 'None'}'")
    print(f"Sample target: '{inp_targets[0] if inp_targets else 'None'}'")

    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets, debug_label="REWRITE/PARAPHRASE")

    # Predict for neighborhood prompts (dictionary format).
    neighborhood_prompts_formatted = [
        el["prompt"].format(record["requested_rewrite"])
        for el in neighborhood_prompts
    ]
    neighborhood_targets = [el["target"] for el in neighborhood_prompts]
    
    neighborhood_correct = test_batch_prediction_acc(
        model,
        tok,
        neighborhood_prompts_formatted,
        neighborhood_targets,
        debug_label="NEIGHBORHOOD"
    )

    probs = stuff_probs + neighborhood_correct

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the results as a dictionary.
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct

    print(f"=== EVALUATION RESULTS ===")
    print(f"Rewrite prompts correct: {ret['rewrite_prompts_correct']}")
    print(f"Paraphrase prompts correct: {ret['paraphrase_prompts_correct']}")
    print(f"Neighborhood prompts correct: {ret['neighborhood_prompts_correct']}")
    print(f"================================\n")

    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target, debug_label=""):
    """
    FIXED VERSION: Proper handling for DeepSeek and Llama models with padding detection.
    """
    print(f"\n--- {debug_label} BATCH PREDICTION DEBUG ---")
    print(f"Model: {model.config._name_or_path}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Number of targets: {len(target) if isinstance(target, list) else 1}")
    
    # Get model path for model-specific handling
    model_path = model.config._name_or_path.lower()
    
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    print(f"Prompt batch shape: {prompt_tok['input_ids'].shape}")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        
        # FIXED: Use attention mask to find the last non-padded position
        # This properly handles padding for both DeepSeek and Llama
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        
        print(f"Logits shape: {logits.shape}")
        print(f"Last non-masked positions: {last_non_masked[:5].tolist()}{'...' if len(last_non_masked) > 5 else ''}")
        
        # FIXED: Apply model-specific logits adjustments
        if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path:
            print(f"🦙 Applying Llama logits adjustment: logits[:, 1:, :]")
            logits = logits[:, 1:, :]
            last_non_masked = last_non_masked - 1  # Adjust indices for shifted logits
            print(f"Adjusted logits shape: {logits.shape}")
            print(f"Adjusted last non-masked positions: {last_non_masked[:5].tolist()}{'...' if len(last_non_masked) > 5 else ''}")
        elif 'deepseek' in model_path:
            print(f"🔧 DeepSeek: Applying SAME logits adjustment as Llama for consistency")
            logits = logits[:, 1:, :]  # SAME as Llama - for consistency
            last_non_masked = last_non_masked - 1  # SAME position adjustment as Llama
            print(f"Adjusted logits shape: {logits.shape}")
            print(f"Adjusted last non-masked positions: {last_non_masked[:5].tolist()}{'...' if len(last_non_masked) > 5 else ''}")
        else:
            print(f"❓ Unknown model type, no logits adjustment")
        
        # Gather logits from the last non-masked position for each sequence
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        # FIXED: Handle target tokenization with model-specific adjustments
        if isinstance(target, list):
            # Multiple targets case
            correct_ids = []
            for t in target:
                target_tokens = tok(t, padding=True, return_tensors="pt").to("cuda")["input_ids"]
                
                if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path:
                    # For Llama, take the token after BOS
                    if target_tokens.size(1) > 1:
                        correct_ids.append(target_tokens[0, 1].item())
                    else:
                        correct_ids.append(target_tokens[0, 0].item())
                elif 'deepseek' in model_path:
                    # For DeepSeek, take the first non-BOS token
                    if target_tokens.size(1) > 1 and target_tokens[0, 0].item() == tok.bos_token_id:
                        correct_ids.append(target_tokens[0, 1].item())
                    else:
                        correct_ids.append(target_tokens[0, 0].item())
                else:
                    # Default behavior
                    correct_ids.append(target_tokens[0, 0].item())
            
            correct_id = torch.tensor(correct_ids).to("cuda")
        else:
            # Single target case
            correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")["input_ids"]
            
            if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path:
                correct_id = correct_id[:, 1].squeeze() if correct_id.size(1) > 1 else correct_id[:, 0].squeeze()
            elif 'deepseek' in model_path:
                if correct_id.size(1) > 1 and correct_id[0, 0].item() == tok.bos_token_id:
                    correct_id = correct_id[:, 1].squeeze()
                else:
                    correct_id = correct_id[:, 0].squeeze()
            else:
                correct_id = correct_id[:, 0].squeeze()
        
        # Text comparison for accuracy
        prediction_text = [tok.decode(token).strip().lower() for token in ans]
        original_text = [t.strip().lower() for t in (target if isinstance(target, list) else [target] * len(ans))]
        text_comparison = []
        
        # Debug prints for each prompt-prediction pair
        print(f"Processing {len(prompts)} prompts:")
        for i in range(min(len(prediction_text), len(original_text), 10)):  # Show first 10
            is_correct = prediction_text[i] == original_text[i]
            text_comparison.append(is_correct)
            
            print(f"  [{i:3d}] Prompt: '{prompts[i]}'")
            print(f"        Target: '{target[i] if isinstance(target, list) else target}' (expected: '{original_text[i]}')")
            print(f"        Prediction: '{prediction_text[i]}' | Correct: {is_correct}")
            print(f"        Token IDs - Predicted: {ans[i].item()}, Expected: {correct_id[i].item() if correct_id.dim() > 0 else correct_id.item()}")
            print(f"        Last non-masked position: {last_non_masked[i].item()}")
            print()
        
        if len(prompts) > 10:
            # Continue processing remaining prompts without detailed output
            for i in range(10, len(prediction_text)):
                is_correct = prediction_text[i] == original_text[i]
                text_comparison.append(is_correct)
        
        accuracy = sum(text_comparison) / len(text_comparison) if text_comparison else 0
        print(f"Batch Accuracy: {accuracy:.3f} ({sum(text_comparison)}/{len(text_comparison)})")
        print(f"--- END {debug_label} BATCH ---\n")
        
        # FIXED: Return consistent format for both model types
        if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path or 'deepseek' in model_path:
            return text_comparison
        else:
            # For other models, keep token ID comparison
            return (ans == correct_id).detach().cpu().numpy().tolist()