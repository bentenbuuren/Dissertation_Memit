"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
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

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    target_tok = tok(" " + target_new["str"])["input_ids"]
    if 'llama' in model.config._name_or_path.lower():
        target_tok = target_tok[1:]
    inp_prompts_og = list(chain(*prob_prompts))
    inp_prompts = [
        el + tok.decode(target_tok[:i]) if 'llama' not in model.config._name_or_path.lower() or i ==0 else el + ' ' + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    print(f"\n=== REWRITE EVALUATION DEBUG ===")
    print(f"Subject: {subject}")
    print(f"Target New: {target_new['str']}")
    print(f"Target True: {target_true['str']}")
    print(f"Target tokens: {target_tok}")
    print(f"Number of input prompts: {len(inp_prompts)}")
    print(f"Number of targets: {len(inp_targets)}")

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
    # Structure the restuls as a dictionary.
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
    print(f"\n--- {debug_label} BATCH PREDICTION DEBUG ---")
    
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        if 'llama' in model.config._name_or_path.lower():
            correct_id = correct_id[:, 1].squeeze()
        else:
            correct_id = correct_id[:, 0].squeeze() #this is the original code
        
        prediction_text = [tok.decode(token).strip().lower() for token in ans]
        original_text = [token.strip().lower() for token in target]
        text_comparison = []
        
        # Debug prints for each prompt-prediction pair
        print(f"Processing {len(prompts)} prompts:")
        for i in range(min(len(prediction_text), len(original_text))):
            is_correct = prediction_text[i] == original_text[i]
            text_comparison.append(is_correct)
            
            print(f"  [{i:3d}] Prompt: '{prompts[i]}'")
            print(f"        Target: '{target[i]}' (expected: '{original_text[i]}')")
            print(f"        Prediction: '{prediction_text[i]}' | Correct: {is_correct}")
            if 'llama' not in model.config._name_or_path.lower():
                pred_token_id = ans[i].item()
                expected_token_id = correct_id[i].item() if correct_id.dim() > 0 else correct_id.item()
                print(f"        Token IDs - Predicted: {pred_token_id}, Expected: {expected_token_id}")
            print()
        
        accuracy = sum(text_comparison) / len(text_comparison) if text_comparison else 0
        print(f"Batch Accuracy: {accuracy:.3f} ({sum(text_comparison)}/{len(text_comparison)})")
        print(f"--- END {debug_label} BATCH ---\n")
        
        if 'llama' in model.config._name_or_path.lower():
            return text_comparison
        else:
            return (ans == correct_id).detach().cpu().numpy().tolist()