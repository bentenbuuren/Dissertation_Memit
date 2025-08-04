#!/usr/bin/env python3
"""
Create Figure 3 style chart: Causal effect of hidden states with Attn or MLP modules severed
This script analyzes the comparative causal effects when different components are disabled.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

class Avg:
    def __init__(self):
        self.sum = 0
        self.count = 0
        
    def add(self, value):
        self.sum += value
        self.count += 1
        
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0

def read_causal_trace_data(results_dir, arch, count, kind=None):
    """Read causal trace results for a specific kind (None, 'mlp', or 'attn')"""
    
    dirname = f"{results_dir}/{arch}/causal_trace/cases/"
    if not os.path.exists(dirname):
        dirname = f"{results_dir}/causal_trace/cases/"
    
    kindcode = "" if not kind else f"_{kind}"
    
    # Data structure to store per-layer effects
    layer_effects = defaultdict(list)
    successful_cases = 0
    
    for i in range(count):
        try:
            filepath = f"{dirname}/knowledge_{i}{kindcode}.npz"
            if not os.path.exists(filepath):
                continue
                
            data = np.load(filepath, allow_pickle=True)
            
            # Only consider cases where model begins with correct prediction
            if "correct_prediction" in data and not data["correct_prediction"]:
                continue
            
            scores = data["scores"]
            subject_range = data["subject_range"]
            
            # Get last subject token position
            if len(subject_range) == 2:
                last_subject_pos = subject_range[1] - 1
            else:
                last_subject_pos = subject_range[-1]
            
            # Extract effects for each layer at the last subject token
            if last_subject_pos < scores.shape[0]:
                layer_scores = scores[last_subject_pos, :]
                base_score = data.get("low_score", 0)
                
                # Store the indirect effect for each layer
                for layer_idx, score in enumerate(layer_scores):
                    indirect_effect = score - base_score
                    layer_effects[layer_idx].append(indirect_effect)
                
                successful_cases += 1
                
        except Exception as e:
            continue
    
    # Calculate average effects per layer
    avg_effects = []
    max_layer = max(layer_effects.keys()) if layer_effects else 0
    
    for layer in range(max_layer + 1):
        if layer in layer_effects and layer_effects[layer]:
            avg_effect = np.mean(layer_effects[layer])
            avg_effects.append(avg_effect)
        else:
            avg_effects.append(0.0)
    
    print(f"Successfully processed {successful_cases} cases for {kind if kind else 'single'}")
    return np.array(avg_effects)

def create_figure3_chart(results_dir, arch, archname, count, output_dir, max_layers=25):
    """Create Figure 3 style chart comparing causal effects"""
    
    print("Reading causal trace data...")
    
    # Read data for all three conditions
    single_effects = read_causal_trace_data(results_dir, arch, count, None)
    attn_effects = read_causal_trace_data(results_dir, arch, count, "attn")  
    mlp_effects = read_causal_trace_data(results_dir, arch, count, "mlp")
    
    # Ensure all arrays have same length (pad with zeros if needed)
    max_len = min(max_layers, max(len(single_effects), len(attn_effects), len(mlp_effects)))
    
    def pad_array(arr, target_len):
        if len(arr) >= target_len:
            return arr[:target_len]
        else:
            padded = np.zeros(target_len)
            padded[:len(arr)] = arr
            return padded
    
    single_effects = pad_array(single_effects, max_len)
    attn_effects = pad_array(attn_effects, max_len)
    mlp_effects = pad_array(mlp_effects, max_len)
    
    # Create the chart
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
    
    # Layer positions
    x = np.arange(max_len)
    width = 0.25
    
    # Create bars with slight offsets
    bars1 = ax.bar(x - width, single_effects, width, 
                   color='#7261ab', alpha=0.8, label='Effect of single state')
    bars2 = ax.bar(x, attn_effects, width,
                   color='#f3201b', alpha=0.8, label='Effect w/ Attn severed')  
    bars3 = ax.bar(x + width, mlp_effects, width,
                   color='#20b020', alpha=0.8, label='Effect w/ MLP severed')
    
    # Formatting
    ax.set_xlabel('Layer at which hidden state is restored')
    ax.set_ylabel('Average Indirect Effect')
    ax.set_title('Causal effect of hidden states Attn or MLP modules severed')
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Set x-axis
    ax.set_xticks(range(0, max_len, 5))
    ax.set_xticklabels(range(0, max_len, 5))
    ax.set_xlim(-0.5, max_len - 0.5)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add grid for readability
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as PDF and PNG
    pdf_path = f"{output_dir}/figure3-causal-effects-{arch}.pdf"
    png_path = f"{output_dir}/figure3-causal-effects-{arch}.png"
    
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    
    print(f"Saved Figure 3 chart to: {pdf_path}")
    print(f"Saved Figure 3 chart to: {png_path}")
    
    plt.show()
    
    # Create additional plot showing MLP gap analysis
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6), dpi=200)
    
    # Calculate MLP gap
    mlp_gap = single_effects - mlp_effects
    
    # Plot the gap
    ax2.bar(x, mlp_gap, width=0.6, color='orange', alpha=0.7, 
            label='MLP Mediating Gap (Single - MLP Severed)')
    
    # Highlight critical layers
    gap_threshold = mlp_gap.max() * 0.3
    critical_mask = mlp_gap > gap_threshold
    
    if np.any(critical_mask):
        ax2.bar(x[critical_mask], mlp_gap[critical_mask], width=0.6, 
                color='red', alpha=0.8, label='Critical MLP Layers')
    
    # Add threshold line
    ax2.axhline(y=gap_threshold, color='red', linestyle='--', alpha=0.5,
                label=f'Critical Threshold ({gap_threshold:.1%})')
    
    # Formatting
    ax2.set_xlabel('Layer at which hidden state is restored')
    ax2.set_ylabel('MLP Mediating Gap (Average Indirect Effect)')
    ax2.set_title('MLP Mediating Role: Gap Between Single States and MLP Severed')
    
    # Format y-axis as percentages
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    
    # Set x-axis
    ax2.set_xticks(range(0, max_len, 5))
    ax2.set_xticklabels(range(0, max_len, 5))
    ax2.set_xlim(-0.5, max_len - 0.5)
    
    # Add legend
    ax2.legend(loc='upper right')
    
    # Add grid for readability
    ax2.grid(True, alpha=0.3)
    
    # Save the MLP gap plot
    gap_pdf_path = f"{output_dir}/mlp-gap-analysis-{arch}.pdf"
    gap_png_path = f"{output_dir}/mlp-gap-analysis-{arch}.png"
    
    plt.savefig(gap_pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(gap_png_path, bbox_inches='tight', dpi=300)
    
    print(f"Saved MLP gap analysis to: {gap_pdf_path}")
    print(f"Saved MLP gap analysis to: {gap_png_path}")
    
    plt.show()
    
    # Calculate the MLP mediating gap (difference between single state and MLP severed)
    # Larger gap = MLP more important at that layer
    mlp_gap = single_effects - mlp_effects
    
    # Print summary statistics
    print("\n" + "="*50)
    print("FIGURE 3 SUMMARY STATISTICS")
    print("="*50)
    print(f"Peak single state effect: {single_effects.max():.1%} at layer {single_effects.argmax()}")
    print(f"Peak Attn severed effect: {attn_effects.max():.1%} at layer {attn_effects.argmax()}")
    print(f"Peak MLP severed effect: {mlp_effects.max():.1%} at layer {mlp_effects.argmax()}")
    print()
    
    # Identify critical MLP layers based on the gap analysis
    print("MLP MEDIATING ROLE ANALYSIS")
    print("="*30)
    print("Gap between single states and MLP severed (indicates MLP importance):")
    
    # Find layers with significant MLP gap (> threshold)
    gap_threshold = mlp_gap.max() * 0.3  # 30% of maximum gap
    critical_layers = []
    
    for layer in range(len(mlp_gap)):
        gap_value = mlp_gap[layer]
        print(f"Layer {layer:2d}: {gap_value:6.1%} gap", end="")
        
        if gap_value > gap_threshold:
            critical_layers.append(layer)
            print(" ← CRITICAL")
        else:
            print()
    
    # Identify the range R of critical MLP layers
    if critical_layers:
        # Find contiguous ranges
        ranges = []
        start = critical_layers[0]
        
        for i in range(1, len(critical_layers)):
            if critical_layers[i] != critical_layers[i-1] + 1:
                # End of current range
                ranges.append((start, critical_layers[i-1]))
                start = critical_layers[i]
        
        # Add the final range
        ranges.append((start, critical_layers[-1]))
        
        print(f"\nCRITICAL MLP LAYER RANGES:")
        for start, end in ranges:
            if start == end:
                print(f"R includes layer {start}")
            else:
                print(f"R includes layers {start}-{end}")
        
        # Main range (largest contiguous range)
        main_range = max(ranges, key=lambda x: x[1] - x[0])
        main_range_list = list(range(main_range[0], main_range[1] + 1))
        
        print(f"\nMAIN CRITICAL RANGE: R = {main_range_list}")
        print(f"This corresponds to: R = {{{','.join(map(str, main_range_list))}}}")
    else:
        print("No critical MLP layers identified with current threshold")
    
    # Analysis of gap diminishing after certain layer
    print(f"\nGAP DIMINISHING ANALYSIS:")
    peak_gap_layer = mlp_gap.argmax()
    print(f"Peak MLP gap at layer {peak_gap_layer}: {mlp_gap[peak_gap_layer]:.1%}")
    
    # Find where gap diminishes to < 50% of peak
    diminish_threshold = mlp_gap[peak_gap_layer] * 0.5
    diminish_layer = None
    
    for layer in range(peak_gap_layer, len(mlp_gap)):
        if mlp_gap[layer] < diminish_threshold:
            diminish_layer = layer
            break
    
    if diminish_layer:
        print(f"Gap diminishes significantly after layer {diminish_layer}")
        print(f"(Gap drops below {diminish_threshold:.1%} at layer {diminish_layer})")
    
    return single_effects, attn_effects, mlp_effects, mlp_gap, critical_layers

def main():
    parser = argparse.ArgumentParser(description='Create Figure 3 causal effects chart')
    parser.add_argument('--arch', required=True, help='Architecture name')
    parser.add_argument('--archname', required=True, help='Architecture display name')
    parser.add_argument('--count', type=int, required=True, help='Number of cases to analyze')
    parser.add_argument('--results_dir', default='results', help='Results directory')
    parser.add_argument('--output_dir', default='analysis_results', help='Output directory')
    parser.add_argument('--max_layers', type=int, default=25, help='Maximum layers to show')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FIGURE 3 CAUSAL EFFECTS ANALYSIS")
    print("="*60)
    print(f"Architecture: {args.arch}")
    print(f"Display name: {args.archname}")
    print(f"Cases to analyze: {args.count}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    create_figure3_chart(
        args.results_dir, 
        args.arch, 
        args.archname, 
        args.count,
        args.output_dir,
        args.max_layers
    )

if __name__ == "__main__":
    main()