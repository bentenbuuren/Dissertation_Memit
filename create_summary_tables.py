#!/usr/bin/env python3
"""
Script to create summary tables from MEMIT evaluation results.
Creates separate CSV files for ZSRE and MCF datasets.
"""

import os
import ast
import pandas as pd
import numpy as np
from pathlib import Path

def parse_filename(filename):
    """Parse filename to extract experiment details."""
    # Remove .txt extension
    name = filename.replace('.txt', '')
    parts = name.split('_')
    
    # Parse components
    model_code = parts[0]  # ds or lm
    method = parts[1]      # mm (MEMIT)
    dataset = parts[2]     # zsre or mcf
    num_edits = int(parts[3])
    layers = parts[4]      # e.g., "1-5", "13-17"
    
    # Check for adapter (lora)
    adapter = 'lora' if len(parts) > 5 and parts[5] == 'lora' else 'none'
    
    # Map model codes to full names
    model_map = {
        'ds': 'DeepSeek-R1-Distill-Llama-8B',
        'lm': 'LLaMA-3.1-8B-Instruct'
    }
    
    return {
        'model': model_map[model_code],
        'method': 'MEMIT',
        'dataset': dataset.upper(),
        'num_edits': num_edits,
        'layers': layers,
        'adapter': adapter,
        'filename': filename
    }

def read_results_file(filepath):
    """Read and parse a results file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse the dictionary safely by using eval with restricted globals
    try:
        # Create safe namespace for evaluation
        safe_dict = {
            "__builtins__": {},
            "nan": float('nan'),
            "inf": float('inf')
        }
        results = eval(content, safe_dict)
        return results
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def format_mean_std(value):
    """Format (mean, std) tuple as 'mean ± std'."""
    if isinstance(value, tuple) and len(value) == 2:
        mean, std = value
        
        # Handle infinite and NaN values
        if np.isinf(mean):
            mean_str = "inf" if mean > 0 else "-inf"
        else:
            mean_str = f"{mean:.2f}"
            
        if np.isnan(std) or np.isinf(std):
            return mean_str
        else:
            return f"{mean_str} ± {std:.2f}"
    else:
        return str(value)

def create_summary_tables(summarize_dir):
    """Create summary tables for ZSRE and MCF datasets."""
    
    # Get all .txt files
    files = [f for f in os.listdir(summarize_dir) if f.endswith('.txt')]
    
    # Parse all files
    all_data = []
    for filename in files:
        filepath = os.path.join(summarize_dir, filename)
        
        # Parse filename
        file_info = parse_filename(filename)
        
        # Read results
        results = read_results_file(filepath)
        if results is None:
            continue
            
        # Combine file info with results
        row_data = file_info.copy()
        
        # Add metrics (excluding non-metric fields)
        exclude_keys = {'num_cases', 'run_dir', 'time'}
        for key, value in results.items():
            if key not in exclude_keys:
                row_data[key] = format_mean_std(value)
        
        all_data.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Define metrics for each dataset
    zsre_metrics = ['post_neighborhood_acc', 'post_paraphrase_acc', 'post_rewrite_acc']
    mcf_metrics = [
        'post_neighborhood_acc', 'post_neighborhood_diff', 'post_neighborhood_success',
        'post_ngram_entropy', 'post_paraphrase_acc', 'post_paraphrase_diff', 
        'post_paraphrase_success', 'post_reference_score', 'post_rewrite_acc', 
        'post_rewrite_diff', 'post_rewrite_success', 'post_score'
    ]
    
    # Create ZSRE table
    zsre_df = df[df['dataset'] == 'ZSRE'].copy()
    if not zsre_df.empty:
        # Select columns
        base_cols = ['model', 'layers', 'adapter', 'method', 'num_edits']
        zsre_cols = base_cols + [col for col in zsre_metrics if col in zsre_df.columns]
        zsre_table = zsre_df[zsre_cols].copy()
        
        # Sort by model, num_edits, then layers
        zsre_table['layer_order'] = zsre_table['layers'].apply(lambda x: int(x.split('-')[0]))
        zsre_table = zsre_table.sort_values(['model', 'num_edits', 'layer_order'])
        zsre_table = zsre_table.drop('layer_order', axis=1)
        
        # Save to CSV
        zsre_table.to_csv('summary_tables/zsre_summary.csv', index=False)
        print(f"ZSRE summary table saved to summary_tables/zsre_summary.csv ({len(zsre_table)} rows)")
    
    # Create MCF table
    mcf_df = df[df['dataset'] == 'MCF'].copy()
    if not mcf_df.empty:
        # Select columns
        base_cols = ['model', 'layers', 'adapter', 'method', 'num_edits']
        mcf_cols = base_cols + [col for col in mcf_metrics if col in mcf_df.columns]
        mcf_table = mcf_df[mcf_cols].copy()
        
        # Sort by model, num_edits, then layers
        mcf_table['layer_order'] = mcf_table['layers'].apply(lambda x: int(x.split('-')[0]))
        mcf_table = mcf_table.sort_values(['model', 'num_edits', 'layer_order'])
        mcf_table = mcf_table.drop('layer_order', axis=1)
        
        # Save to CSV
        mcf_table.to_csv('summary_tables/mcf_summary.csv', index=False)
        print(f"MCF summary table saved to summary_tables/mcf_summary.csv ({len(mcf_table)} rows)")
    
    print("\nSummary:")
    print(f"Total files processed: {len(all_data)}")
    print(f"ZSRE experiments: {len(zsre_df) if not zsre_df.empty else 0}")
    print(f"MCF experiments: {len(mcf_df) if not mcf_df.empty else 0}")

if __name__ == "__main__":
    summarize_dir = "/Users/benjamintenbuuren/Desktop/Dissertation_Memit/summarize"
    create_summary_tables(summarize_dir)