#!/usr/bin/env python3
"""
Script to create a chart showing rewrite accuracy for ZSRE dataset.
Creates separate plots for each model with different lines for different layers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_mean_std(value_str):
    """Extract mean and std from 'mean ± std' format."""
    if '±' in value_str:
        parts = value_str.split('±')
        mean = float(parts[0].strip())
        std = float(parts[1].strip())
        return mean, std
    else:
        # Handle cases with only mean value
        try:
            mean = float(value_str.strip())
            return mean, 0.0
        except:
            return 0.0, 0.0

def create_zsre_rewrite_accuracy_chart():
    """Create chart showing rewrite accuracy for ZSRE dataset."""
    
    # Read the ZSRE summary table
    df = pd.read_csv('summary_tables/zsre_summary.csv')
    
    # Extract mean and std from rewrite accuracy
    df[['rewrite_acc_mean', 'rewrite_acc_std']] = df['post_rewrite_acc'].apply(
        lambda x: pd.Series(extract_mean_std(x))
    )
    
    # Filter out rows with LoRA adapter for cleaner visualization
    df_no_lora = df[df['adapter'] == 'none'].copy()
    
    # Get unique models
    models = df_no_lora['model'].unique()
    
    # Create plots
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 6))
    if len(models) == 1:
        axes = [axes]  # Make it iterable
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = df_no_lora[df_no_lora['model'] == model]
        
        # Get unique layers
        layers = sorted(model_data['layers'].unique())
        
        for j, layer in enumerate(layers):
            layer_data = model_data[model_data['layers'] == layer].sort_values('num_edits')
            
            if not layer_data.empty:
                x = layer_data['num_edits']
                y = layer_data['rewrite_acc_mean']
                yerr = layer_data['rewrite_acc_std']
                
                # Plot line with error bars
                ax.errorbar(x, y, yerr=yerr, 
                           marker='o', linewidth=2, markersize=6,
                           label=f'Layers {layer}', 
                           color=colors[j % len(colors)],
                           capsize=4, capthick=1)
        
        # Customize the plot
        ax.set_xlabel('Number of Edits', fontsize=12)
        ax.set_ylabel('Rewrite Accuracy (%)', fontsize=12)
        ax.set_title(f'{model.replace("-", "-\\n")}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Set x-axis to log scale for better visualization
        ax.set_xscale('log')
        
        # Set y-axis limits
        ax.set_ylim(0, 110)
        
        # Format x-axis ticks
        ax.set_xticks([1, 10, 100, 1000, 10000])
        ax.set_xticklabels(['1', '10', '100', '1K', '10K'])
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('summary_tables', exist_ok=True)
    plt.savefig('summary_tables/zsre_rewrite_accuracy_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig('summary_tables/zsre_rewrite_accuracy_chart.pdf', bbox_inches='tight')
    
    print("ZSRE Rewrite Accuracy chart saved to:")
    print("- summary_tables/zsre_rewrite_accuracy_chart.png")
    print("- summary_tables/zsre_rewrite_accuracy_chart.pdf")
    
    # Show the plot
    plt.show()

def create_individual_model_charts():
    """Create individual charts for each model."""
    
    # Read the ZSRE summary table
    df = pd.read_csv('summary_tables/zsre_summary.csv')
    
    # Extract mean and std from rewrite accuracy
    df[['rewrite_acc_mean', 'rewrite_acc_std']] = df['post_rewrite_acc'].apply(
        lambda x: pd.Series(extract_mean_std(x))
    )
    
    # Filter out rows with LoRA adapter for cleaner visualization
    df_no_lora = df[df['adapter'] == 'none'].copy()
    
    # Get unique models
    models = df_no_lora['model'].unique()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for model in models:
        plt.figure(figsize=(10, 6))
        model_data = df_no_lora[df_no_lora['model'] == model]
        
        # Get unique layers
        layers = sorted(model_data['layers'].unique())
        
        for j, layer in enumerate(layers):
            layer_data = model_data[model_data['layers'] == layer].sort_values('num_edits')
            
            if not layer_data.empty:
                x = layer_data['num_edits']
                y = layer_data['rewrite_acc_mean']
                yerr = layer_data['rewrite_acc_std']
                
                # Plot line with error bars
                plt.errorbar(x, y, yerr=yerr, 
                           marker='o', linewidth=2, markersize=8,
                           label=f'Layers {layer}', 
                           color=colors[j % len(colors)],
                           capsize=5, capthick=2)
        
        # Customize the plot
        plt.xlabel('Number of Edits', fontsize=14)
        plt.ylabel('Rewrite Accuracy (%)', fontsize=14)
        plt.title(f'ZSRE Rewrite Accuracy - {model}', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Set x-axis to log scale for better visualization
        plt.xscale('log')
        
        # Set y-axis limits
        plt.ylim(0, 110)
        
        # Format x-axis ticks
        plt.xticks([1, 10, 100, 1000, 10000], ['1', '10', '100', '1K', '10K'])
        
        # Save individual model chart
        model_name = model.replace('/', '_').replace('-', '_')
        plt.savefig(f'summary_tables/zsre_rewrite_accuracy_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'summary_tables/zsre_rewrite_accuracy_{model_name}.pdf', bbox_inches='tight')
        
        print(f"Individual chart saved for {model}:")
        print(f"- summary_tables/zsre_rewrite_accuracy_{model_name}.png")
        print(f"- summary_tables/zsre_rewrite_accuracy_{model_name}.pdf")
        
        plt.show()

if __name__ == "__main__":
    print("Creating ZSRE Rewrite Accuracy charts...")
    
    # Create combined chart
    create_zsre_rewrite_accuracy_chart()
    
    print("\nCreating individual model charts...")
    
    # Create individual charts
    create_individual_model_charts()
    
    print("\nAll charts created successfully!")