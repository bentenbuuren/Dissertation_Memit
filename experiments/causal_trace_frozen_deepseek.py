import argparse
import json
import os
import re
from collections import defaultdict

import numpy
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import KnownsDataset
from util import nethook
from util.globals import DATA_DIR


def calculate_selection_thresholds(gap_values):
    """
    Calculate different thresholds for layer selection based on various methods
    """
    # Ensure tensor is float type for quantile calculation
    gap_values = gap_values.float() if gap_values.dtype != torch.float32 else gap_values
    
    gap_mean = torch.mean(gap_values).item()
    gap_std = torch.std(gap_values).item()
    gap_max = torch.max(gap_values).item()
    
    thresholds = {
        "60_percent": gap_max * 0.6,                    # Original arbitrary choice
        "statistical": gap_mean + gap_std,              # Mean + 1 standard deviation
        "top_percentile": torch.quantile(gap_values, 0.75).item(),  # Top 25% of layers
        "elbow_method": calculate_elbow_threshold(gap_values),      # Elbow in sorted gaps
        "memit_style": gap_max * 0.3,                   # 30% of peak (adapted for higher values)
        "significant": gap_mean + 2 * gap_std,          # Mean + 2 std (more conservative)
    }
    
    return thresholds


def calculate_elbow_threshold(gap_values):
    """
    Find elbow point in sorted gap values using knee/elbow detection
    """
    sorted_gaps = torch.sort(gap_values, descending=True)[0]
    
    # Calculate differences between consecutive gaps
    diffs = torch.diff(sorted_gaps)
    
    # Find the largest drop (elbow point)
    if len(diffs) > 0:
        elbow_idx = torch.argmax(torch.abs(diffs)).item()
        elbow_threshold = sorted_gaps[elbow_idx + 1].item()
    else:
        elbow_threshold = torch.mean(gap_values).item()
    
    return elbow_threshold


def analyze_layer_selection(ordinary_trace, mlp_frozen_trace, model_name, min_layers=3, max_layers=10):
    """
    Analyze causal traces to suggest optimal layers for MEMIT editing
    Based on ROME/MEMIT methodology using gap analysis
    """
    # Calculate gap: where MLP modules are most important
    gap = ordinary_trace - mlp_frozen_trace
    
    # Find peak MLP effect
    peak_layer = torch.argmax(gap).item()
    peak_value = gap[peak_layer].item()
    
    # Multiple threshold methods for comparison
    thresholds = calculate_selection_thresholds(gap)
    
    print(f"\n📊 THRESHOLD ANALYSIS:")
    for method, threshold in thresholds.items():
        above_threshold = gap >= threshold
        count = torch.sum(above_threshold).item()
        print(f"   {method}: {threshold:.4f} ({count} layers)")
    
    # Use memit_style threshold (adapted for higher gap values)
    threshold = thresholds["memit_style"]
    
    # Find all layers above threshold
    above_threshold = gap >= threshold
    candidate_layers = torch.where(above_threshold)[0].tolist()
    
    # Find consecutive ranges
    consecutive_ranges = find_consecutive_ranges(candidate_layers)
    
    # Select optimal range with constraints
    optimal_range = select_optimal_range(consecutive_ranges, peak_layer, gap, 
                                       min_layers=min_layers, max_layers=max_layers)
    
    # Print detailed gap analysis for manual inspection
    print(f"\n📋 DETAILED GAP ANALYSIS (for manual layer selection)")
    print(f"   Model: {model_name}")
    print(f"   Peak gap: {peak_value:.4f} at layer {peak_layer}")
    print(f"\n   Layer | Gap Value | % of Peak | Above Threshold?")
    print(f"   ------|-----------|-----------|----------------")
    
    for layer_idx in range(len(gap)):
        gap_val = gap[layer_idx].item()
        percent_of_peak = (gap_val / peak_value * 100) if peak_value > 0 else 0
        above_thresh = "✅" if gap_val >= threshold else "❌"
        
        print(f"   {layer_idx:5d} | {gap_val:8.4f} | {percent_of_peak:8.1f}% | {above_thresh}")
    
    print(f"\n🎯 AUTOMATIC SELECTION ANALYSIS")
    print(f"   Selected threshold (memit_style): {threshold:.4f} ({threshold/peak_value*100:.1f}% of peak)")
    print(f"   Layers above threshold: {candidate_layers}")
    print(f"   Consecutive ranges: {consecutive_ranges}")
    print(f"   Layer constraints: {min_layers}-{max_layers} layers")
    print(f"   📍 RECOMMENDED MEMIT RANGE: {optimal_range[0]}-{optimal_range[-1]} ({len(optimal_range)} layers)")
    print(f"   Gap values in range: {[f'{gap[i].item():.4f}' for i in optimal_range]}")
    print(f"   Average gap in range: {torch.mean(gap[optimal_range]).item():.4f}")
    
    # Print manual selection suggestions
    print(f"\n💡 MANUAL SELECTION SUGGESTIONS:")
    print(f"   Conservative (≥50% of peak): Layers {[i for i in range(len(gap)) if gap[i] >= peak_value * 0.5]}")
    print(f"   Moderate (≥30% of peak):     Layers {[i for i in range(len(gap)) if gap[i] >= peak_value * 0.3]}")
    print(f"   Liberal (≥20% of peak):      Layers {[i for i in range(len(gap)) if gap[i] >= peak_value * 0.2]}")
    print(f"   Very Liberal (≥10% of peak): Layers {[i for i in range(len(gap)) if gap[i] >= peak_value * 0.1]}")
    
    return optimal_range


def find_consecutive_ranges(layer_list):
    """
    Find consecutive ranges in a list of layer indices
    """
    if not layer_list:
        return []
    
    ranges = []
    current_range = [layer_list[0]]
    
    for i in range(1, len(layer_list)):
        if layer_list[i] == layer_list[i-1] + 1:
            current_range.append(layer_list[i])
        else:
            ranges.append(current_range)
            current_range = [layer_list[i]]
    
    ranges.append(current_range)
    return ranges


def select_optimal_range(consecutive_ranges, peak_layer, gap_values, min_layers=3, max_layers=10):
    """
    Select the best consecutive range for MEMIT editing
    """
    if not consecutive_ranges:
        # Fallback: create range around peak
        target_size = (min_layers + max_layers) // 2  # Use middle of range
        start = max(0, peak_layer - target_size//2)
        end = min(len(gap_values), start + target_size)
        return list(range(start, end))
    
    # Score each range
    best_range = None
    best_score = -1
    
    for range_layers in consecutive_ranges:
        # Criteria for scoring:
        # 1. Contains peak layer (bonus)
        # 2. Good average gap value
        # 3. Appropriate size (prefer target_size)
        
        # Filter ranges that are too small or too large
        if len(range_layers) < min_layers or len(range_layers) > max_layers:
            continue
            
        contains_peak = peak_layer in range_layers
        avg_gap = torch.mean(gap_values[range_layers]).item()
        
        # Prefer ranges closer to middle of min-max range
        target_size = (min_layers + max_layers) // 2
        size_penalty = abs(len(range_layers) - target_size) / target_size
        
        # Combined score
        score = avg_gap * (1.5 if contains_peak else 1.0) * (1 - size_penalty * 0.2)
        
        if score > best_score:
            best_score = score
            best_range = range_layers
    
    # If no valid range found, create one around peak
    if best_range is None or len(best_range) < min_layers:
        # Force create minimum range around peak
        start = max(0, peak_layer - min_layers//2)
        end = min(len(gap_values), start + min_layers)
        # Adjust if we hit boundary
        if end - start < min_layers:
            start = max(0, end - min_layers)
        best_range = list(range(start, end))
    
    return best_range


def save_layer_analysis(ordinary_trace, mlp_frozen_trace, optimal_layers, filename):
    """
    Save layer analysis results to JSON file
    """
    gap = ordinary_trace - mlp_frozen_trace
    
    analysis_data = {
        "model_analysis": {
            "total_layers": len(gap),
            "peak_layer": torch.argmax(gap).item(),
            "peak_gap_value": gap.max().item(),
            "recommended_layers": optimal_layers,
            "layer_range": f"{optimal_layers[0]}-{optimal_layers[-1]}",
            "num_layers_selected": len(optimal_layers)
        },
        "gap_analysis": {
            "all_gaps": gap.tolist(),
            "selected_gaps": [gap[i].item() for i in optimal_layers],
            "avg_gap_in_selection": torch.mean(gap[optimal_layers]).item(),
            "threshold_used": gap.max().item() * 0.6
        },
        "hyperparameters_suggestion": {
            "memit_layers": optimal_layers,
            "rome_layer": torch.argmax(gap).item(),
            "editing_range": f"layers {optimal_layers[0]} to {optimal_layers[-1]}"
        }
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"\n💾 Layer analysis saved to: {filename}")
    print(f"   Use these layers in your MEMIT hparams file:")
    print(f"   \"layers\": {optimal_layers}")


def main():
    parser = argparse.ArgumentParser(description="Frozen Causal Tracing for DeepSeek Models")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        choices=[
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ],
    )
    aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/frozen_causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--num_samples", default=1000, type=int, help="Number of samples to process")
    aa("--min_layers", default=3, type=int, help="Minimum number of layers to select for MEMIT")
    aa("--max_layers", default=10, type=int, help="Maximum number of layers to select for MEMIT")
    args = parser.parse_args()

    modeldir = f'{args.model_name.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Half precision for large models (DeepSeek is 8B)
    torch_dtype = torch.float16

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype)

    if args.fact_file is None:
        knowns = KnownsDataset(DATA_DIR)
    else:
        with open(args.fact_file) as f:
            knowns = json.load(f)

    noise_level = args.noise_level
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, [k["subject"] for k in knowns]
            )
            print(f"Using noise_level {noise_level} to match model times {factor}")

    # Process samples and calculate frozen causal traces
    all_ordinary = []
    all_no_attn = []
    all_no_mlp = []
    successful_cases = 0
    skipped_cases = 0

    for i, knowledge in enumerate(tqdm(knowns[:args.num_samples])):
        cache_dir = f"{result_dir}/case_{i}"
        
        ordinary, no_attn, no_mlp = calculate_last_subject(
            mt,
            knowledge["prompt"],
            knowledge["subject"],
            expect=knowledge["attribute"],  # Pass expected answer for validation
            cache=cache_dir,
            noise_level=noise_level,
        )
        
        # Check if case was successful (not skipped)
        if ordinary is not None and no_attn is not None and no_mlp is not None:
            all_ordinary.append(ordinary)
            all_no_attn.append(no_attn)
            all_no_mlp.append(no_mlp)
            successful_cases += 1
            
            # Plot individual case
            case_chart_path = f"{pdf_dir}/case_{i}.pdf"
            print(f"📊 Case {i}: ✅ Success - Generating chart: {case_chart_path}")
            plot_comparison(
                ordinary, no_attn, no_mlp, 
                f"{knowledge['prompt']} -> {knowledge['subject']}",
                savepdf=case_chart_path
            )
        else:
            skipped_cases += 1
            print(f"📊 Case {i}: ❌ Skipped - Model prediction mismatch")

    # Calculate and plot average results
    avg_ordinary = torch.stack(all_ordinary).mean(dim=0)
    avg_no_attn = torch.stack(all_no_attn).mean(dim=0)
    avg_no_mlp = torch.stack(all_no_mlp).mean(dim=0)

    # Add sample count to chart title
    title = f"Causal effect with Attn or MLP modules frozen ({args.model_name})\n(Average of {successful_cases} successful cases)"
    chart_path = f"{pdf_dir}/average_frozen_causal_trace.pdf"
    
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"   Total cases processed: {args.num_samples}")
    print(f"   ✅ Successful cases: {successful_cases}")
    print(f"   ❌ Skipped cases: {skipped_cases}")
    print(f"   Success rate: {successful_cases/args.num_samples*100:.1f}%")
    
    print(f"\n📊 Generating main chart: {chart_path}")
    plot_comparison(
        avg_ordinary, avg_no_attn, avg_no_mlp, title,
        savepdf=chart_path
    )
    print(f"✅ Main chart saved to: {chart_path}")

    print(f"Peak differences: ordinary={avg_ordinary.max():.4f}, no_attn={avg_no_attn.max():.4f}, no_mlp={avg_no_mlp.max():.4f}")
    
    # Analyze and suggest optimal layers for MEMIT
    optimal_layers = analyze_layer_selection(avg_ordinary, avg_no_mlp, args.model_name, 
                                            min_layers=args.min_layers, max_layers=args.max_layers)
    
    # Save layer selection results
    layer_analysis_file = f"{output_dir}/optimal_layers_analysis.json"
    save_layer_analysis(avg_ordinary, avg_no_mlp, optimal_layers, layer_analysis_file)
    
    # Final summary of generated files
    print(f"\n🎯 GENERATED FILES SUMMARY:")
    print(f"   📂 Output directory: {output_dir}")
    print(f"   📊 Main chart: {pdf_dir}/average_frozen_causal_trace.pdf")
    print(f"   📊 Individual charts: {pdf_dir}/case_*.pdf ({successful_cases} successful cases)")
    print(f"   📋 Layer analysis: {layer_analysis_file}")
    print(f"   💾 Cached results: {result_dir}/case_*/")
    print(f"\n   To view the main chart: open {pdf_dir}/average_frozen_causal_trace.pdf")


def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
):
    """
    Three-way intervention:
    1. Corrupt subset of input
    2. Restore subset of hidden states
    3. Freeze set of MLP/Attn modules at corrupted state
    """
    prng = numpy.random.RandomState(1)  # For reproducibility
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule
    def patch_rep(x, layer):
        if layer == embed_layername:
            # Corrupt token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # Restore uncorrupted hidden state for selected tokens
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # Run with patching rules
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # Return softmax probabilities
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    return probs


def calculate_hidden_flow_frozen(
    mt,
    prompt,
    subject,
    expect=None,
    token_range="last_subject",
    samples=10,
    noise=0.1,
    disable_mlp=False,
    disable_attn=False,
):
    """
    Runs frozen causal tracing over every token/layer combination
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    
    # Check if model prediction matches expected answer
    if expect is not None and answer.strip() != expect:
        print(f"   ❌ SKIPPING '{prompt}' - Expected: '{expect}', Got: '{answer.strip()}'")
        return None  # Return None for skipped cases
    
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    if token_range == "last_subject":
        token_range = [e_range[1] - 1]
    
    # Get low score with corrupted input
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    
    differences = trace_important_states_frozen(
        mt.model,
        mt.num_layers,
        inp,
        e_range,
        answer_t,
        noise=noise,
        disable_mlp=disable_mlp,
        disable_attn=disable_attn,
        token_range=token_range,
    )
    differences = differences.detach().cpu()
    
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
    )


def trace_important_states_frozen(
    model,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    disable_mlp=False,
    disable_attn=False,
    token_range=None,
):
    """
    Trace important states with frozen MLP/Attn modules
    """
    ntoks = inp["input_ids"].shape[1]
    table = []
    
    if token_range is None:
        token_range = range(ntoks)
    
    for tnum in token_range:
        zero_mlps = []
        if disable_mlp:
            zero_mlps = [
                (tnum, layername(model, L, "mlp")) for L in range(0, num_layers)
            ]
        if disable_attn:
            zero_mlps += [
                (tnum, layername(model, L, "attn")) for L in range(0, num_layers)
            ]
        
        row = []
        for layer in range(0, num_layers):
            r = trace_with_repatch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                zero_mlps,  # states_to_unpatch
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_with_patch(
    model, inp, states_to_patch, answers_t, tokens_to_mix, noise=0.1
):
    """
    Standard causal trace with patch
    """
    rs = numpy.random.RandomState(1)
    prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    def patch_rep(x, layer):
        if layer == embed_layername:
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    return probs


def calculate_last_subject(mt, prompt, entity, expect=None, cache=None, token_range="last_subject", noise_level=0.1):
    """
    Calculate frozen causal traces for last subject token
    """
    def load_from_cache(filename):
        try:
            dat = numpy.load(f"{cache}/{filename}")
            return {
                k: v if not isinstance(v, numpy.ndarray)
                else str(v) if v.dtype.type is numpy.str_
                else torch.from_numpy(v)
                for k, v in dat.items()
            }
        except (FileNotFoundError, TypeError):
            return None

    if cache is not None:
        no_attn_r = load_from_cache("no_attn_r.npz")
        no_mlp_r = load_from_cache("no_mlp_r.npz")
        ordinary_r = load_from_cache("ordinary.npz")
    else:
        no_attn_r = no_mlp_r = ordinary_r = None

    uncached_no_attn_r = no_attn_r is None
    uncached_no_mlp_r = no_mlp_r is None
    uncached_ordinary_r = ordinary_r is None

    if uncached_no_attn_r:
        no_attn_r = calculate_hidden_flow_frozen(
            mt, prompt, entity,
            expect=expect,
            disable_attn=True,
            token_range=token_range,
            noise=noise_level,
        )
        if no_attn_r is None:  # Skipped case
            return None, None, None
            
    if uncached_no_mlp_r:
        no_mlp_r = calculate_hidden_flow_frozen(
            mt, prompt, entity,
            expect=expect,
            disable_mlp=True,
            token_range=token_range,
            noise=noise_level,
        )
        if no_mlp_r is None:  # Skipped case
            return None, None, None
            
    if uncached_ordinary_r:
        ordinary_r = calculate_hidden_flow_frozen(
            mt, prompt, entity,
            expect=expect,
            token_range=token_range,
            noise=noise_level,
        )
        if ordinary_r is None:  # Skipped case
            return None, None, None

    if cache is not None:
        os.makedirs(cache, exist_ok=True)
        for u, r, filename in [
            (uncached_no_attn_r, no_attn_r, "no_attn_r.npz"),
            (uncached_no_mlp_r, no_mlp_r, "no_mlp_r.npz"),
            (uncached_ordinary_r, ordinary_r, "ordinary.npz"),
        ]:
            if u:
                numpy.savez(
                    f"{cache}/{filename}",
                    **{
                        k: v.cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in r.items()
                    },
                )

    return (
        ordinary_r["scores"][0] - ordinary_r["low_score"],
        no_attn_r["scores"][0] - ordinary_r["low_score"],
        no_mlp_r["scores"][0] - ordinary_r["low_score"],
    )


def plot_comparison(ordinary, no_attn, no_mlp, title, savepdf=None):
    """
    Plot comparison of ordinary, no attention, and no MLP traces
    """
    import matplotlib.ticker as mtick

    fig, ax = plt.subplots(1, figsize=(8, 3), dpi=300)
    
    # Create bars with different colors
    width = 0.25
    x = numpy.arange(len(ordinary))
    
    ax.bar(x - width, ordinary, width=width, color="#7261ab", 
           label="Impact of single state on P")
    ax.bar(x, no_attn, width=width, color="#f3201b", 
           label="Impact with Attn frozen")
    ax.bar(x + width, no_mlp, width=width, color="#20b020", 
           label="Impact with MLP frozen")
    
    ax.set_title(title)
    ax.set_ylabel("Indirect Effect")
    ax.set_xlabel("Layer at which the single hidden state is restored")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, max(0.025, ordinary.max() * 1.1))
    ax.legend(frameon=False)
    
    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


class ModelAndTokenizer:
    """
    Model and tokenizer holder with layer counting
    """
    def __init__(self, model_name=None, model=None, tokenizer=None, torch_dtype=None):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="original_models")
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, cache_dir="original_models"
            )
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
        
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n) or 
                re.match(r"^model\.layers\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    """Get layer name for different model architectures"""
    if hasattr(model, "transformer"):
        # GPT-2 style models
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    elif hasattr(model, "gpt_neox"):
        # GPT-NeoX style models
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama style models (includes DeepSeek-R1-Distill-Llama-8B)
        if kind == "embed":
            return "model.embed_tokens"
        elif kind == "mlp":
            return f"model.layers.{num}.mlp"
        elif kind == "attn":
            return f"model.layers.{num}.self_attn"
        else:
            return f"model.layers.{num}"
    else:
        # Try to detect based on model config if available
        if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
            if 'llama' in model.config.model_type.lower():
                if kind == "embed":
                    return "model.embed_tokens"
                elif kind == "mlp":
                    return f"model.layers.{num}.mlp"
                elif kind == "attn":
                    return f"model.layers.{num}.self_attn"
                else:
                    return f"model.layers.{num}"
        
        assert False, f"Unknown transformer structure for model type: {type(model)}"


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    """Create model inputs from prompts"""
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    """Decode tokens to strings"""
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    """Find token range for substring in token array"""
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def predict_from_input(model, inp):
    """Get predictions from model input"""
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def collect_embedding_std(mt, subjects):
    """Collect embedding standard deviation for noise calibration"""
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        embed_layer = layername(mt.model, 0, "embed")
        with nethook.Trace(mt.model, embed_layer) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level


if __name__ == "__main__":
    main()