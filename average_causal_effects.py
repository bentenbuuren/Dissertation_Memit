#!/usr/bin/env python3
"""
Average Causal Effects Analysis Script
Converts the average_causal_effects.ipynb notebook into a standalone script for HPC usage.

Usage:
    python average_causal_effects.py --arch meta-llama_Llama-3.1-8B-Instruct --count 150 --output_dir results_analysis

Arguments:
    --arch: Model architecture name (used for directory structure)
    --archname: Display name for plots (optional, defaults to arch)
    --count: Number of cases to analyze (default: 150)
    --output_dir: Directory to save output plots and data
    --results_dir: Base results directory (default: "results")
"""

import numpy as np
import os
import argparse
import math
from matplotlib import pyplot as plt
from pathlib import Path
import json


class Avg:
    """Helper class for computing running averages and statistics"""
    def __init__(self):
        self.d = []

    def add(self, v):
        self.d.append(v[None])

    def add_all(self, vv):
        self.d.append(vv)

    def avg(self):
        if not self.d:
            return np.array([])
        return np.concatenate(self.d).mean(axis=0)

    def std(self):
        if not self.d:
            return np.array([])
        return np.concatenate(self.d).std(axis=0)

    def size(self):
        return sum(datum.shape[0] for datum in self.d)

def read_knowledge(count=150, kind=None, arch="gpt2-xl", results_dir="results"):
    """
    Read and aggregate causal tracing results
    
    Args:
        count: Number of cases to read
        kind: Type of experiment (None, "mlp", "attn")
        arch: Model architecture name
        results_dir: Base results directory
    
    Returns:
        Dictionary with aggregated statistics
    """
    dirname = f"{results_dir}/{arch}/causal_trace/cases/"
    
    # Check if directory exists
    if not os.path.exists(dirname):
        # Try alternative directory structure
        alt_dirname = f"{results_dir}/causal_trace/cases/"
        if os.path.exists(alt_dirname):
            dirname = alt_dirname
        else:
            raise FileNotFoundError(f"Could not find causal trace results in {dirname} or {alt_dirname}")
    
    print(f"Reading from: {dirname}")
    
    kindcode = "" if not kind else f"_{kind}"
    
    # Initialize averagers for different metrics
    (avg_fe, avg_ee, avg_le, avg_fa, avg_ea, avg_la, 
     avg_hs, avg_ls, avg_fs, avg_fle, avg_fla) = [Avg() for _ in range(11)]
    
    successful_cases = 0
    failed_cases = 0
    
    for i in range(count):
        try:
            # Load the numpy file
            filepath = f"{dirname}/knowledge_{i}{kindcode}.npz"
            if not os.path.exists(filepath):
                failed_cases += 1
                continue
                
            data = np.load(filepath, allow_pickle=True)
            
            # Only consider cases where the model begins with the correct prediction
            if "correct_prediction" in data and not data["correct_prediction"]:
                failed_cases += 1
                continue
            
            scores = data["scores"]
            subject_range = data["subject_range"]
            
            # Handle different subject_range formats
            if len(subject_range) == 2:
                first_e, first_a = subject_range
            else:
                # Handle numpy array or other formats
                subject_range = list(subject_range)
                first_e, first_a = subject_range[0], subject_range[1]
            
            last_e = first_a - 1
            last_a = len(scores) - 1
            
            # Original prediction (high score)
            if "high_score" in data:
                avg_hs.add(data["high_score"])
            
            # Prediction after subject is corrupted (low score)
            avg_ls.add(data["low_score"])
            avg_fs.add(scores.max())
            
            # Some maximum computations
            avg_fle.add(scores[last_e].max())
            avg_fla.add(scores[last_a].max())
            
            # First subject, middle subject, last subject
            avg_fe.add(scores[first_e])
            if first_e + 1 < last_e:
                avg_ee.add_all(scores[first_e + 1:last_e])
            avg_le.add(scores[last_e])
            
            # First after, middle after, last after
            avg_fa.add(scores[first_a])
            if first_a + 1 < last_a:
                avg_ea.add_all(scores[first_a + 1:last_a])
            avg_la.add(scores[last_a])
            
            successful_cases += 1
            
        except Exception as e:
            print(f"Error reading case {i}: {e}")
            failed_cases += 1
            continue
    
    print(f"Successfully processed {successful_cases} cases, failed on {failed_cases} cases")
    
    if successful_cases == 0:
        raise ValueError("No valid cases found. Check your results directory and file format.")
    
    # Compute aggregated results
    result = np.stack([
        avg_fe.avg(),
        avg_ee.avg(),
        avg_le.avg(), 
        avg_fa.avg(),
        avg_ea.avg(),
        avg_la.avg(),
    ])
    
    result_std = np.stack([
        avg_fe.std(),
        avg_ee.std(),
        avg_le.std(),
        avg_fa.std(),
        avg_ea.std(),
        avg_la.std(),
    ])
    
    # Print summary statistics
    print("\n" + "="*50)
    print(f"SUMMARY STATISTICS ({kind if kind else 'single vector'})")
    print("="*50)
    
    if avg_hs.size() > 0:
        print(f"Average Total Effect: {avg_hs.avg() - avg_ls.avg():.4f}")
    
    print(f"Best average indirect effect on last subject: {avg_le.avg().max() - avg_ls.avg():.4f}")
    print(f"Best average indirect effect on last token: {avg_la.avg().max() - avg_ls.avg():.4f}")
    print(f"Average best-fixed score: {avg_fs.avg():.4f}")
    print(f"Average best-fixed on last subject token score: {avg_fle.avg():.4f}")
    print(f"Average best-fixed on last word score: {avg_fla.avg():.4f}")
    print(f"Argmax at last subject token: {np.argmax(avg_le.avg())}")
    print(f"Max at last subject token: {np.max(avg_le.avg()):.4f}")
    print(f"Argmax at last prompt token: {np.argmax(avg_la.avg())}")
    print(f"Max at last prompt token: {np.max(avg_la.avg()):.4f}")
    
    return {
        "low_score": avg_ls.avg(), 
        "result": result, 
        "result_std": result_std, 
        "size": avg_fe.size(),
        "kind": kind
    }

def plot_line_graph(arch, archname, count, results_dir="results", output_dir="results_analysis"):
    """Create the main line plot comparing single vector, MLP, and attention restoration"""
    
    labels = [
        "First subject token",
        "Middle subject tokens", 
        "Last subject token",
        "First subsequent token",
        "Further tokens",
        "Last token",
    ]
    color_order = [0, 1, 2, 4, 5, 3]
    x = None

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True, dpi=200)
    
    results_data = {}
    
    for j, (kind, title) in enumerate([
        (None, "single hidden vector"),
        ("mlp", "run of 10 MLP lookups"), 
        ("attn", "run of 10 Attn modules"),
    ]):
        print(f"\nReading {kind if kind else 'single vector'} results...")
        
        try:
            d = read_knowledge(count, kind, arch, results_dir)
            results_data[kind if kind else 'single'] = d
            
            for i, label in enumerate(labels):
                y = d["result"][i] - d["low_score"]
                if x is None:
                    x = list(range(len(y)))
                    
                std = d["result_std"][i] 
                error = std * 1.96 / math.sqrt(d["size"])
                
                axes[j].fill_between(
                    x, y - error, y + error, alpha=0.3, color=cmap.colors[color_order[i]]
                )
                axes[j].plot(x, y, label=label, color=cmap.colors[color_order[i]])

            axes[j].set_title(f"Average indirect effect of a {title}")
            axes[j].set_ylabel("Average indirect effect on p(o)")
            axes[j].set_xlabel(f"Layer number in {archname}")
            
        except Exception as e:
            print(f"Error processing {kind}: {e}")
            continue
    
    # Add legend to middle plot
    if len(axes) > 1:
        axes[1].legend(frameon=False)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = f"{output_dir}/lineplot-causaltrace-{arch}.pdf"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved line plot to: {plot_path}")
    
    # Also save as PNG for easier viewing
    png_path = f"{output_dir}/lineplot-causaltrace-{arch}.png"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"Saved line plot to: {png_path}")
    
    plt.show()
    
    return results_data

def plot_heatmaps(results_data, arch, archname, output_dir="results_analysis"):
    """Create heatmap visualizations for each experiment type"""
    
    labels = [
        "First subject token",
        "Middle subject tokens",
        "Last subject token", 
        "First subsequent token",
        "Further tokens",
        "Last token",
    ]
    
    high_score = None  # Scale all plots according to the first plot
    
    for kind_key, kind_name, cmap_name in [
        ('single', 'single vector', 'Purples'),
        ('mlp', 'MLP', 'Greens'),
        ('attn', 'Attention', 'Reds')
    ]:
        if kind_key not in results_data:
            print(f"Skipping {kind_name} heatmap - no data available")
            continue
            
        d = results_data[kind_key]
        count = d["size"]
        
        what = {
            'single': "Indirect Effect of $h_i^{(l)}$",
            'mlp': "Indirect Effect of MLP", 
            'attn': "Indirect Effect of Attn",
        }[kind_key]
        
        title = f"Avg {what} over {count} prompts"
        result = np.clip(d["result"] - d["low_score"], 0, None)
        
        if kind_key == 'single':
            high_score = result.max()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            result,
            cmap=cmap_name,
            vmin=0.0,
            vmax=high_score,
        )
        
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(result))])
        ax.set_xticks([0.5 + i for i in range(0, result.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, result.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        
        if kind_key == 'single':
            ax.set_xlabel(f"single patched layer within {archname}")
        else:
            ax.set_xlabel(f"center of interval of 10 patched {kind_name} layers")
        
        cb = plt.colorbar(h)
        cb.ax.set_title("AIE", y=-0.16, fontsize=10)
        
        # Save heatmap
        heatmap_path = f"{output_dir}/heatmap-{kind_key}-{arch}.pdf"
        plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
        print(f"Saved {kind_name} heatmap to: {heatmap_path}")
        
        # Also save as PNG
        png_path = f"{output_dir}/heatmap-{kind_key}-{arch}.png"
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        
        plt.show()

def generate_memit_recommendations(results_data, arch, output_dir="results_analysis"):
    """Generate MEMIT layer recommendations based on causal tracing results"""
    
    if 'single' not in results_data:
        print("Cannot generate MEMIT recommendations without single vector results")
        return
    
    # Focus on last subject token (index 2) as it's most important for MEMIT
    last_subject_effects = results_data['single']['result'][2] - results_data['single']['low_score']
    
    # Find top layers
    layer_effects = [(i, effect) for i, effect in enumerate(last_subject_effects)]
    layer_effects.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("MEMIT LAYER RECOMMENDATIONS")
    print("="*60)
    
    print("\nTop 10 layers by causal effect on last subject token:")
    for i, (layer, effect) in enumerate(layer_effects[:10]):
        print(f"  {i+1:2d}. Layer {layer:2d}: {effect:.4f}")
    
    # Generate different layer selection strategies
    top_5_layers = [layer for layer, _ in layer_effects[:5]]
    top_3_layers = [layer for layer, _ in layer_effects[:3]]
    
    # Find peak region (consecutive high-effect layers)
    peak_start = top_5_layers[0]
    for i in range(len(top_5_layers)-1):
        if top_5_layers[i+1] - top_5_layers[i] > 2:  # Gap of more than 2
            peak_start = top_5_layers[i+1]
            break
    
    peak_region = list(range(max(0, peak_start-2), min(len(last_subject_effects), peak_start+3)))
    
    recommendations = {
        "model_name": arch,
        "analysis_date": str(np.datetime64('now')),
        "top_5_layers": top_5_layers,
        "top_3_layers": top_3_layers, 
        "peak_region": peak_region,
        "layer_effects": {str(i): float(effect) for i, effect in enumerate(last_subject_effects)},
        "recommendations": {
            "conservative": {
                "layers": top_3_layers,
                "mom2_update_weight": 15000,
                "description": "Use top 3 layers for focused, high-precision editing"
            },
            "balanced": {
                "layers": top_5_layers,
                "mom2_update_weight": 5000,
                "description": "Use top 5 layers for good coverage and effectiveness"
            },
            "comprehensive": {
                "layers": peak_region,
                "mom2_update_weight": 1000,
                "description": "Use peak region for maximum coverage"
            }
        }
    }
    
    print(f"\nRECOMMENDED LAYER SELECTIONS:")
    print(f"  Conservative (3 layers): {top_3_layers}")
    print(f"  Balanced (5 layers):     {top_5_layers}")
    print(f"  Comprehensive (region):  {peak_region}")
    
    # Compare MLP vs Attention if available
    if 'mlp' in results_data and 'attn' in results_data:
        mlp_effects = results_data['mlp']['result'][2] - results_data['mlp']['low_score']
        attn_effects = results_data['attn']['result'][2] - results_data['attn']['low_score']
        
        mlp_peak = np.max(mlp_effects)
        attn_peak = np.max(attn_effects)
        
        print(f"\nMODULE TARGETING RECOMMENDATION:")
        if mlp_peak > attn_peak:
            print(f"  Target MLP (peak effect: {mlp_peak:.4f} > {attn_peak:.4f})")
            recommendations["target_module"] = "model.layers.{}.mlp.down_proj"
        else:
            print(f"  Target Attention (peak effect: {attn_peak:.4f} > {mlp_peak:.4f})")
            recommendations["target_module"] = "model.layers.{}.self_attn.o_proj"
    
    # Save recommendations
    rec_path = f"{output_dir}/memit_recommendations_{arch}.json"
    with open(rec_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"\nSaved recommendations to: {rec_path}")
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='Analyze average causal effects from causal tracing results')
    parser.add_argument('--arch', required=True, help='Model architecture name (e.g., meta-llama_Llama-3.1-8B-Instruct)')
    parser.add_argument('--archname', help='Display name for plots (defaults to arch)')
    parser.add_argument('--count', type=int, default=150, help='Number of cases to analyze')
    parser.add_argument('--output_dir', default='results_analysis', help='Directory to save outputs')
    parser.add_argument('--results_dir', default='results', help='Base results directory')
    
    args = parser.parse_args()
    
    archname = args.archname if args.archname else args.arch.replace('_', ' ')
    
    print("="*60)
    print("AVERAGE CAUSAL EFFECTS ANALYSIS")
    print("="*60)
    print(f"Architecture: {args.arch}")
    print(f"Display name: {archname}")
    print(f"Cases to analyze: {args.count}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Create line plots and get aggregated data
        results_data = plot_line_graph(args.arch, archname, args.count, args.results_dir, args.output_dir)
        
        # Create heatmaps
        plot_heatmaps(results_data, args.arch, archname, args.output_dir)
        
        # Generate MEMIT recommendations
        generate_memit_recommendations(results_data, args.arch, args.output_dir)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"All outputs saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())