"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.

ENHANCED WITH DEBUGGING: This version includes comprehensive debugging output
to track what's happening during evaluation.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity


def compute_rewrite_quality_counterfact(
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
    :param snips: AttributeSnippets for generation testing
    :param vec: TfidfVectorizer for similarity computation

    :return: Dictionary containing rewriting metrics
    """
    
    print("\n" + "="*80)
    print("🔍 STARTING COUNTERFACT EVALUATION")
    print("="*80)

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    generation_prompts = record["generation_prompts"]

    print(f"📋 EVALUATION SETUP:")
    print(f"   Subject: '{subject}'")
    print(f"   Target (new): '{target_new['str']}'")
    print(f"   Target (true): '{target_true['str']}'")
    print(f"   Original prompt: '{record['requested_rewrite']['prompt']}'")
    print(f"   Formatted rewrite prompt: '{rewrite_prompts[0]}'")
    print(f"   Number of paraphrase prompts: {len(paraphrase_prompts)}")
    print(f"   Number of neighborhood prompts: {len(neighborhood_prompts)}")
    print(f"   Number of generation prompts: {len(generation_prompts)}")
    
    # Show sample prompts for debugging
    if paraphrase_prompts:
        print(f"   Sample paraphrase: '{paraphrase_prompts[0]}'")
    if neighborhood_prompts:
        print(f"   Sample neighborhood: '{neighborhood_prompts[0]}'")

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],      # 0 = target_new is correct
        [0 for _ in range(len(paraphrase_prompts))],   # 0 = target_new is correct
        [1 for _ in range(len(neighborhood_prompts))], # 1 = target_true is correct
    ]
    
    print(f"\n🎯 PROMPT CATEGORIES:")
    print(f"   Rewrite prompts: {len(prob_prompts[0])} (expecting NEW target)")
    print(f"   Paraphrase prompts: {len(prob_prompts[1])} (expecting NEW target)")
    print(f"   Neighborhood prompts: {len(prob_prompts[2])} (expecting TRUE target)")
    
    # Flatten all the evaluated prefixes into one list.
    print(f"\n⚡ RUNNING BATCH PREDICTION TEST...")
    total_prompts = sum(len(p) for p in prob_prompts)
    print(f"   Total prompts to evaluate: {total_prompts}")
    
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )
    
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    
    print(f"\n📊 PROBABILITY RESULTS:")
    print(f"   Cutoffs: {cutoffs}")
    for i, key in enumerate(["rewrite_prompts", "paraphrase_prompts", "neighborhood_prompts"]):
        probs_section = ret_probs[i]
        correct_section = ret_corrects[i]
        accuracy = np.mean(correct_section) if correct_section else 0.0
        
        # Calculate average probabilities from prob dictionaries
        if probs_section:
            if isinstance(probs_section[0], dict):
                avg_new_prob = np.mean([p['target_new'] for p in probs_section])
                avg_true_prob = np.mean([p['target_true'] for p in probs_section])
                print(f"   {key}:")
                print(f"     - Accuracy: {accuracy:.3f} ({sum(correct_section)}/{len(correct_section)})")
                print(f"     - Avg NEW target NLL: {avg_new_prob:.3f}")
                print(f"     - Avg TRUE target NLL: {avg_true_prob:.3f}")
            else:
                avg_prob = np.mean(probs_section)
                print(f"   {key}:")
                print(f"     - Accuracy: {accuracy:.3f} ({sum(correct_section)}/{len(correct_section)})")
                print(f"     - Avg probability: {avg_prob:.3f}")

    # Structure the results as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts", 
                "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }

    if snips is not None:
        print(f"\n🎭 RUNNING GENERATION TESTS...")
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        
        print(f"   Relation ID: {rel_id}")
        print(f"   Target new ID: {target_new['id']}")
        print(f"   Consistency texts found: {len(consistency_texts)}")
        print(f"   Essence texts found: {len(essence_texts)}")
        
        if consistency_texts:
            print(f"   Sample consistency text: '{consistency_texts[0][:100]}...'")
        if essence_texts:
            print(f"   Sample essence text: '{essence_texts[0][:100]}...'")
        
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)
    else:
        print(f"\n⚠️  SKIPPING GENERATION TESTS (snips is None)")

    print(f"\n✅ EVALUATION COMPLETE")
    print(f"   Final metrics keys: {list(ret.keys())}")
    print("="*80 + "\n")

    return ret


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: typing.List[int],
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    FIXED IMPLEMENTATION - No more logits adjustment for DeepSeek
    """
    
    print(f"\n🔬 BATCH PREDICTION DEBUG:")
    print(f"   Number of prefixes: {len(prefixes)}")
    print(f"   Target new: '{target_new}'")
    print(f"   Target true: '{target_true}'")
    print(f"   Model: {model.config._name_or_path}")
    print(f"   Which correct pattern: {which_correct[:10]}{'...' if len(which_correct) > 10 else ''}")

    # Get model path for tokenization handling
    model_path = model.config._name_or_path.lower()
    
    # Calculate prefix lengths and handle BOS tokens properly
    if 'deepseek' in model_path:
        print(f"   🔧 DEEPSEEK MODEL DETECTED - Using DeepSeek-specific handling")
        # For DeepSeek, check if BOS tokens are automatically added
        sample_prefix_tokens = tok(prefixes[0])["input_ids"]
        has_bos = sample_prefix_tokens[0] == tok.bos_token_id if tok.bos_token_id is not None else False
        
        if has_bos:
            print(f"   DeepSeek: BOS token detected, adjusting prefix lengths")
            prefix_lens = [len(tok(prefix)["input_ids"]) - 1 for prefix in prefixes]
        else:
            print(f"   DeepSeek: No BOS token, using original prefix lengths")
            prefix_lens = [len(tok(prefix)["input_ids"]) for prefix in prefixes]
    elif 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path:
        print(f"   🦙 LLAMA MODEL DETECTED - Using Llama-specific handling")
        prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
        prefix_lens = [lengths - 1 for lengths in prefix_lens]  # Remove BOS
    else:
        print(f"   ❓ UNKNOWN MODEL TYPE - Using default handling")
        prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    
    # Create prompts with both targets
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Tokenize targets and handle BOS tokens
    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    
    print(f"   Original prefix lengths: {prefix_lens[:5]}{'...' if len(prefix_lens) > 5 else ''}")
    print(f"   Target new tokens (before adjustment): {a_tok}")
    print(f"   Target true tokens (before adjustment): {b_tok}")
    print(f"   Prompt batch shape: {prompt_tok['input_ids'].shape}")

    # Remove BOS tokens from target tokens for both models
    if 'llama-3.1' in model_path or 'llama-3' in model_path or 'llama-2' in model_path:
        print(f"   🦙 Removing BOS token from Llama target tokens")
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
    elif 'deepseek' in model_path:
        print(f"   🔧 Removing BOS token from DeepSeek target tokens")
        a_tok = a_tok[1:]  # Remove BOS token from targets
        b_tok = b_tok[1:]  # Remove BOS token from targets
    
    print(f"   Adjusted target new tokens: {a_tok}")
    print(f"   Adjusted target true tokens: {b_tok}")
    print(f"   Final prefix lengths: {prefix_lens[:5]}{'...' if len(prefix_lens) > 5 else ''}")

    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    print(f"   Choice A (new) length: {choice_a_len}")
    print(f"   Choice B (true) length: {choice_b_len}")

    # Get model predictions
    with torch.no_grad():
        model = model.to("cuda")
        prompt_tok = prompt_tok.to("cuda")
        logits = model(**prompt_tok).logits

    print(f"   Raw logits shape: {logits.shape}")

    # Logits adjustment removed for all model types as per latest instructions.
    
    # Initialize probability and correctness arrays
    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    print(f"\n📝 INDIVIDUAL PROMPT ANALYSIS:")
    print(f"Processing {logits.size(0)} sequences (2 per prefix: NEW and TRUE targets)")
    
    # Process each sequence
    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        is_target_new = (i % 2 == 0)
        current_target = target_new if is_target_new else target_true
        current_tokens = a_tok if is_target_new else b_tok

        prefix_idx = i // 2
        expected_for_accuracy = target_new if which_correct[prefix_idx] == 0 else target_true

        # Show details for ALL sequences
        print(f"\n  [{i:3d}] PROMPT: '{prefixes[prefix_idx]}'")
        print(f"        TESTING: '{current_target}'")
        print(f"        EXPECTED: {expected_for_accuracy}")

        # DEBUG: Print tokenized input sequence with indices
        batch_sequence = prompt_tok['input_ids'][i]
        decoded_tokens = [tok.decode([t]) for t in batch_sequence]
        token_debug_str = "\n".join([f"    [{k:2d}] Token ID: {t} → '{decoded_tokens[k]}'" for k, t in enumerate(batch_sequence)])
        print("    Full input token sequence:")
        print(token_debug_str)

        # Compute suffix probabilities
        sequence_log_prob = 0.0
        sequence_correct = True
        predicted_token_id = None

        for j in range(cur_len):
            cur_tok = current_tokens[j]

            # CRITICAL FIX: Account for padding in batch processing
            # Find where actual content starts (skip padding tokens)
            batch_sequence = prompt_tok['input_ids'][i]
            padding_offset = 0

            # Count padding tokens at the beginning
            pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0
            while (padding_offset < len(batch_sequence) and
                   batch_sequence[padding_offset] == pad_token_id):
                padding_offset += 1

            # Index calculation, now always:
            logit_idx = padding_offset + prefix_lens[prefix_idx] + j

            if logit_idx >= 0 and logit_idx < logits.size(1):
                # Calculate negative log likelihood
                nll = -torch.nn.functional.log_softmax(
                    logits[i, logit_idx, :], dim=0
                )[cur_tok].item()
                sequence_log_prob += nll

                # Check if token is predicted correctly
                predicted_token_id = logits[i, logit_idx, :].argmax().item()
                token_correct = (predicted_token_id == cur_tok)
                if not token_correct:
                    sequence_correct = False

                # Show the main token prediction
                predicted_token_str = tok.decode([predicted_token_id])
                current_token_str = tok.decode([cur_tok])
                print(f"        Token {j+1}: '{predicted_token_str}' (NLL of '{current_token_str}': {nll:.4f})")
                print(f"        Logit index: {logit_idx} (padding_offset: {padding_offset}, prefix_len: {prefix_lens[prefix_idx]}, j: {j})")

                # Inline debug print for each token
                context_token_str = tok.decode([batch_sequence[logit_idx]]) if logit_idx > 0 and logit_idx - 1 < len(batch_sequence) else "[START]"
                target_token_str = tok.decode([cur_tok])
                predicted_token_str = tok.decode([predicted_token_id])
                correctness = "True" if token_correct else "False"
                print(
                    f"Debugging Token position {j}:\n"
                    f"  Context token ID: {batch_sequence[logit_idx - 1]} → '{context_token_str}'\n"
                    f"  Target token ID:  {cur_tok} → '{target_token_str}'\n"
                    f"  Predicted token ID: {predicted_token_id} → '{predicted_token_str}'\n"
                    f"  Logit position: {logit_idx}\n"
                    f"  Correct Prediction: {correctness}"
                )

        # Average negative log likelihood
        probs[i] = sequence_log_prob / cur_len

        # Compute accuracy on targets that should be correct
        counts_toward_accuracy = (which_correct[prefix_idx] == 0 and i % 2 == 0) or (which_correct[prefix_idx] == 1 and i % 2 == 1)

        if counts_toward_accuracy:
            targets_correct.append(sequence_correct)
            if sequence_correct:
                print(f"        OUTCOME: ✅ Successful edit")
            else:
                print(f"        OUTCOME: ❌ Failed edit")
        else:
            # For informational sequences, determine if it shows successful editing
            if predicted_token_id is not None:
                predicted_token_str = tok.decode([predicted_token_id])
                if predicted_token_str.strip() == target_new:
                    print(f"        OUTCOME: ✅ Successful edit (informational)")
                else:
                    print(f"        OUTCOME: Informational only")
            else:
                print(f"        OUTCOME: Informational only")

    # Return format - list of dictionaries with target_new and target_true probabilities
    prob_results = [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ]
    
    accuracy = np.mean(targets_correct) if targets_correct else 0.0
    print(f"\nBatch Accuracy: {accuracy:.3f} ({sum(targets_correct)}/{len(targets_correct)})")
    print(f"Average NLL - NEW: {np.mean([p['target_new'] for p in prob_results]):.3f}")
    print(f"Average NLL - TRUE: {np.mean([p['target_true'] for p in prob_results]):.3f}")
    print(f"--- END BATCH PREDICTION DEBUG ---\n")

    return prob_results, targets_correct


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    """
    ORIGINAL IMPLEMENTATION with enhanced debugging
    """
    print(f"\n🎨 GENERATION TEST DEBUG:")
    print(f"   Number of generation prompts: {len(prefixes)}")
    print(f"   Number of consistency texts: {len(consistency_texts)}")
    print(f"   Number of essence texts: {len(essence_texts)}")
    
    # Show sample prompts and reference texts
    if prefixes:
        print(f"   Sample generation prompt: '{prefixes[0]}'")
    if consistency_texts:
        print(f"   Sample consistency text: '{consistency_texts[0][:100]}...'")
    if essence_texts:
        print(f"   Sample essence text: '{essence_texts[0][:100]}...'")
    
    print(f"\n🤖 Generating {len(prefixes)} texts...")

    # Generate all texts at once
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    print(f"Generated texts:")
    for i, (prompt, gen_text) in enumerate(zip(prefixes, gen_texts)):
        print(f"  [{i+1}] Prompt: '{prompt}'")
        print(f"       Generated: '{gen_text}'")
        print()

    # Calculate n-gram entropy
    ngram_entropy = n_gram_entropy(gen_texts)
    print(f"N-gram entropy: {ngram_entropy:.4f}")
    
    # Calculate consistency TF-IDF similarity
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )
    print(f"Consistency TF-IDF similarity: {consistency_tfidf:.4f}")

    # Prepare return dictionary
    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    # Calculate essence score and perplexity if essence texts exist
    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})
        print(f"Essence perplexity: {ppl:.4f}")

    print(f"--- END GENERATION TEST DEBUG ---\n")
    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    entropy_scores = [compute_n_gram_entropy(txt) for txt in gen_texts]
    result = (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_scores).item()
    
    print(f"Individual n-gram entropies: {[f'{e:.3f}' for e in entropy_scores[:3]]}{'...' if len(entropy_scores) > 3 else ''}")
    
    return result


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        if len(freqs) == 0:
            entropy_list.append(0.0)
        else:
            freqs = freqs / freqs.sum()
            entropy = np.sum(-freqs * np.log(freqs) / np.log(2))
            entropy_list.append(entropy)

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()