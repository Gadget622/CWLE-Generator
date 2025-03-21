#!/usr/bin/env python3

"""
CWLE Visualizer

This script loads CWLE parameters from a JSON file and generates 
visualization images for each CWLE pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize CWLEs from a JSON file.')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to JSON file containing CWLE parameters')
    parser.add_argument('--output-dir', type=str, default='cwle_images',
                        help='Directory to save output images')
    parser.add_argument('--img-size', type=int, default=28,
                        help='Size of output images (width and height)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='DPI for saved images')
    parser.add_argument('--combined', action='store_true',
                        help='Create a combined image with all patterns')
    parser.add_argument('--grayscale', action='store_true',
                        help='Save images in grayscale instead of color map')
    return parser.parse_args()

def generate_wave_pattern(cwle, img_size):
    """
    Generate a 2D pattern from CWLE parameters.
    
    Args:
        cwle: List of wave parameters (dict with frequency, orientation, phase, weight)
        img_size: Image size (width & height)
        
    Returns:
        numpy.ndarray: 2D array of pixel values
    """
    # Initialize empty pattern
    pattern = np.zeros((img_size, img_size))
    
    # Create coordinate grid
    x = np.arange(img_size)
    y = np.arange(img_size)
    X, Y = np.meshgrid(x, y)
    
    # Sum contributions from each wave
    for wave in cwle:
        freq = wave["frequency"]
        theta = wave["orientation"] * (np.pi / 180)  # Convert to radians
        phase = wave["phase"]
        weight = wave["weight"]
        
        # Apply rotation to coordinates
        X_rot = X * np.cos(theta) + Y * np.sin(theta)
        
        # Calculate wave value at this position
        wave_values = np.sin(2 * np.pi * freq * X_rot + phase)
        
        # Normalize from [-1,1] to [0,1] range
        wave_values = (wave_values + 1) / 2
        
        # Add weighted contribution to pattern
        pattern += weight * wave_values
    
    # Rescale pattern to ensure full 0-1 range
    min_val = np.min(pattern)
    max_val = np.max(pattern)
    if max_val > min_val:  # Avoid division by zero
        pattern = (pattern - min_val) / (max_val - min_val)
    
    return pattern

def save_pattern_image(pattern, filename, use_grayscale=False, dpi=100):
    """
    Save pattern as an image.
    
    Args:
        pattern: 2D array of pixel values
        filename: Output file path
        use_grayscale: If True, use grayscale instead of colormap
        dpi: DPI for saved image
    """
    plt.figure(figsize=(5, 5))
    
    if use_grayscale:
        plt.imshow(pattern, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    else:
        plt.imshow(pattern, cmap='viridis', interpolation='nearest')
        plt.colorbar(shrink=0.8)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

def save_combined_image(patterns, filename, use_grayscale=False, dpi=100):
    """
    Save all patterns as a combined image.
    
    Args:
        patterns: List of 2D arrays
        filename: Output file path
        use_grayscale: If True, use grayscale instead of colormap
        dpi: DPI for saved image
    """
    n = len(patterns)
    
    # Determine grid dimensions
    grid_size = int(np.ceil(np.sqrt(n)))
    
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(2*grid_size, 2*grid_size))
    
    # Flatten axes array for easier iteration
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    for i, pattern in enumerate(patterns):
        if i < len(axs):
            if use_grayscale:
                im = axs[i].imshow(pattern, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            else:
                im = axs[i].imshow(pattern, cmap='viridis', interpolation='nearest')
            
            axs[i].set_title(f"Class {i}")
            axs[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(patterns), len(axs)):
        axs[i].axis('off')
    
    # Add colorbar if not grayscale
    if not use_grayscale:
        fig.colorbar(im, ax=axs, shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CWLE parameters from JSON file
    with open(args.input, 'r') as f:
        cwles = json.load(f)
    
    print(f"Loaded {len(cwles)} CWLEs from {args.input}")
    
    # Generate and save pattern images
    patterns = []
    for i, cwle in enumerate(cwles):
        # Generate pattern
        pattern = generate_wave_pattern(cwle, args.img_size)
        patterns.append(pattern)
        
        # Save individual pattern image
        output_path = os.path.join(args.output_dir, f"cwle_{i}.png")
        save_pattern_image(pattern, output_path, args.grayscale, args.dpi)
        print(f"Saved pattern {i} to {output_path}")
    
    # Save combined image if requested
    if args.combined:
        combined_path = os.path.join(args.output_dir, "cwle_combined.png")
        save_combined_image(patterns, combined_path, args.grayscale, args.dpi)
        print(f"Saved combined image to {combined_path}")
    
    print(f"All CWLE patterns visualized successfully!")

if __name__ == "__main__":
    main()