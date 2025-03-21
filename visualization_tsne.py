#!/usr/bin/env python3

"""
CWLE Visualization with t-SNE

This script visualizes CWLE patterns in a 2D space using t-SNE dimensionality reduction.
It can compare generated CWLEs with handcrafted ones.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import argparse
import os
from cwle_generator import CWLEGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize CWLEs using t-SNE.')
    parser.add_argument('--config', type=str, default='cwle_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--generated', type=str, default='cwles.json',
                        help='Path to generated CWLEs file')
    parser.add_argument('--handcrafted', type=str, default=None,
                        help='Path to handcrafted CWLEs file for comparison')
    parser.add_argument('--output', type=str, default='tsne_visualization.png',
                        help='Path to output visualization file')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found.")
        return
    
    if not os.path.exists(args.generated):
        print(f"Generated CWLEs file {args.generated} not found.")
        return
    
    # Initialize generator for pattern generation
    generator = CWLEGenerator(args.config)
    
    # Load generated CWLEs
    with open(args.generated, 'r') as f:
        generated_cwles = json.load(f)
    
    # Generate patterns for CWLEs
    generated_patterns = [generator._generate_wave_pattern(cwle) for cwle in generated_cwles]
    
    # Prepare data for t-SNE
    data = []
    labels = []
    
    # Add generated patterns
    for i, pattern in enumerate(generated_patterns):
        data.append(pattern.flatten())
        labels.append(f"Gen-{i}")
    
    # Add handcrafted patterns if provided
    if args.handcrafted and os.path.exists(args.handcrafted):
        with open(args.handcrafted, 'r') as f:
            handcrafted_cwles = json.load(f)
        
        handcrafted_patterns = [generator._generate_wave_pattern(cwle) for cwle in handcrafted_cwles]
        
        for i, pattern in enumerate(handcrafted_patterns):
            data.append(pattern.flatten())
            labels.append(f"Hand-{i}")
    
    # Convert to numpy array
    data_array = np.array(data)
    
    # Apply t-SNE
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    tsne_results = tsne.fit_transform(data_array)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot generated patterns
    n_generated = len(generated_patterns)
    plt.scatter(
        tsne_results[:n_generated, 0],
        tsne_results[:n_generated, 1],
        c=range(n_generated),
        cmap='tab10',
        marker='o',
        s=100,
        alpha=0.8,
        label='Generated'
    )
    
    # Add labels for generated patterns
    for i in range(n_generated):
        plt.annotate(
            i,
            (tsne_results[i, 0], tsne_results[i, 1]),
            fontsize=10,
            ha='center',
            va='center',
            color='white',
            fontweight='bold'
        )
    
    # Plot handcrafted patterns if any
    if args.handcrafted and os.path.exists(args.handcrafted):
        plt.scatter(
            tsne_results[n_generated:, 0],
            tsne_results[n_generated:, 1],
            c=range(len(tsne_results) - n_generated),
            cmap='tab10',
            marker='s',
            s=100,
            alpha=0.8,
            label='Handcrafted'
        )
        
        # Add labels for handcrafted patterns
        for i in range(n_generated, len(tsne_results)):
            plt.annotate(
                i - n_generated,
                (tsne_results[i, 0], tsne_results[i, 1]),
                fontsize=10,
                ha='center',
                va='center',
                color='white',
                fontweight='bold'
            )
    
    # Add legend and labels
    plt.title('t-SNE Visualization of CWLE Patterns')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {args.output}")
    plt.close()
    
    # Calculate and display distances
    print("\nDistance analysis:")
    
    # Calculate distances for generated patterns
    dist_matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            dist = np.linalg.norm(data[i] - data[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # Display minimum distances for each pattern
    for i in range(len(data)):
        min_dist_idx = np.argmin(dist_matrix[i, [j for j in range(len(data)) if j != i]])
        min_dist_idx = [j for j in range(len(data)) if j != i][min_dist_idx]
        print(f"{labels[i]} - Closest to {labels[min_dist_idx]} (distance: {dist_matrix[i, min_dist_idx]:.4f})")
    
    # Display overall minimum, maximum, and average distances
    non_zero_dists = dist_matrix[dist_matrix > 0]
    print(f"\nOverall statistics:")
    print(f"  Minimum distance: {np.min(non_zero_dists):.4f}")
    print(f"  Maximum distance: {np.max(non_zero_dists):.4f}")
    print(f"  Average distance: {np.mean(non_zero_dists):.4f}")
    
    # If we have both generated and handcrafted, compare them
    if args.handcrafted and os.path.exists(args.handcrafted):
        gen_vs_hand = dist_matrix[:n_generated, n_generated:]
        print(f"\nGenerated vs Handcrafted:")
        print(f"  Minimum distance: {np.min(gen_vs_hand):.4f}")
        print(f"  Maximum distance: {np.max(gen_vs_hand):.4f}")
        print(f"  Average distance: {np.mean(gen_vs_hand):.4f}")

if __name__ == "__main__":
    main()