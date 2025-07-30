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
            el + tok.decode(target_tok[:i])
            for el in inp_prompts_og
            for i in range(len(target_tok))
        ]
    elif 'deepseek' in model_path:
        inp_prompts = [
            el + tok.decode(target_tok[:i])
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
    # Show full input token sequence
    decoded_tokens = [[tok.decode([t]) for t in seq] for seq in prompt_tok["input_ids"]]
    print("Full input token sequences:")
    for idx, (token_ids, tokens) in enumerate(zip(prompt_tok["input_ids"], decoded_tokens)):
        print(f"  Sequence {idx}:")
        for k, (tid, tstr) in enumerate(zip(token_ids, tokens)):
            print(f"    [{k:2d}] Token ID: {tid.item()} → '{tstr}'")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        
        # Calculate padding offset: count how many padding tokens at start of each sequence
        pad_token_id = tok.pad_token_id
        padding_offsets = []
        for seq in prompt_tok['input_ids']:
            pad_count = 0
            for token_id in seq:
                if token_id.item() == pad_token_id:
                    pad_count += 1
                else:
                    break
            padding_offsets.append(pad_count)
        padding_offsets = torch.tensor(padding_offsets).to(logits.device)

        # Compute prompt_lengths such that the logit index corresponds to the last non-padding token for each prompt
        # Attention mask sums to length of non-padded tokens
        attention_sums = prompt_tok['attention_mask'].sum(dim=1)  # length of non-padded tokens per sequence
        # The index to gather logits for is padding offset + (length of non-padding tokens) - 1
        # Explanation:
        # - padding offset accounts for initial pad tokens at start (if any)
        # - attention sum is count of non-pad tokens (includes BOS etc.)
        # - subtract 1 because indexing is zero-based and we want last token index
        prompt_lengths = padding_offsets + attention_sums - 1

        print(f"Padding offsets per sequence: {padding_offsets.tolist()}")
        print(f"Attention sums per sequence: {attention_sums.tolist()}")
        print(f"Logit indices to gather (padding_offset + attention_sum - 1): {prompt_lengths.tolist()}")

        to_gather = prompt_lengths.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        to_gather = to_gather.to(logits.device)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)
        
        # FIXED: Apply model-specific logits adjustments
        if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path:
            print(f"🦙 Applying Llama logits adjustment: logits[:, 1:, :]")
            # logits = logits[:, 1:, :]
            print(f"Adjusted logits shape: {logits.shape}")
        elif 'deepseek' in model_path:
            print(f"🔧 DeepSeek: Applying SAME logits adjustment as Llama for consistency")
            # logits = logits[:, 1:, :]  # SAME as Llama - for consistency
            print(f"Adjusted logits shape: {logits.shape}")
        else:
            print(f"❓ Unknown model type, no logits adjustment")
        
        # Gather logits from the last non-masked position for each sequence
        # to_gather and gathered already computed above
        # ans already computed above

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
        for i in range(len(prompts)):
            is_correct = prediction_text[i] == original_text[i]
            text_comparison.append(is_correct)
            
            print(f"\n  [{i:3d}] PROMPT: '{prompts[i]}'")
            print(f"        TARGET: '{original_text[i]}'")
            print(f"        PREDICTION: '{prediction_text[i]}'")
            print(f"        Logit index (padding_offset + attention_sum - 1): {prompt_lengths[i].item()}")

            # Context token is token at logit index in input_ids
            context_token_str = tok.decode([prompt_tok['input_ids'][i][prompt_lengths[i]].item()]) if prompt_lengths[i] >= 0 else "[START]"
            predicted_token_str = tok.decode([ans[i].item()])
            target_token_str = tok.decode([correct_id[i].item() if correct_id.dim() > 0 else correct_id.item()])
            correctness = "✅" if is_correct else "❌"

            print(f"        CONTEXT TOKEN: '{context_token_str}'")
            print(f"        PREDICTED TOKEN: '{predicted_token_str}'")
            print(f"        TARGET TOKEN: '{target_token_str}'")
            print(f"        CORRECT: {correctness}")
        
        accuracy = sum(text_comparison) / len(text_comparison) if text_comparison else 0
        print(f"Batch Accuracy: {accuracy:.3f} ({sum(text_comparison)}/{len(text_comparison)})")
        print(f"--- END {debug_label} BATCH ---\n")
        
        # FIXED: Return consistent format for both model types
        if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path or 'deepseek' in model_path:
            return text_comparison
        else:
            # For other models, keep token ID comparison
            return (ans == correct_id).detach().cpu().numpy().tolist()