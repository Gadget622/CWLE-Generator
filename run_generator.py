#!/usr/bin/env python3

"""
Run script for CWLE Generator

This script runs the CWLE Generator with the specified configuration file.
It also provides options for visualizing the results.
"""

import argparse
import os
from cwle_generator import CWLEGenerator
import json
import numpy as np

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run CWLE Generator.')
    parser.add_argument('--config', type=str, default='cwle_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='cwles.json',
                        help='Path to output file')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the generated patterns')
    parser.add_argument('--no-convergence', action='store_true',
                        help='Do not show convergence plots')
    parser.add_argument('--compare', type=str, default=None,
                        help='Path to file with handcrafted CWLEs for comparison')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found.")
        return
    
    # Initialize generator
    generator = CWLEGenerator(args.config)
    
    # Generate CWLEs
    print("Generating CWLEs...")
    cwles, patterns = generator.generate_cwles()
    
    # Save CWLEs to file
    with open(args.output, "w") as f:
        json.dump(cwles, f, indent=4)
    print(f"CWLEs saved to {args.output}")
    
    # Visualize patterns if requested
    if args.visualize:
        print("Visualizing patterns...")
        generator.visualize_patterns(patterns, "cwle_patterns.png")
    
    # Visualize convergence if requested
    if not args.no_convergence:
        print("Visualizing convergence...")
        generator.visualize_convergence("cwle_convergence.png")
    
    # Compare with handcrafted CWLEs if provided
    if args.compare and os.path.exists(args.compare):
        print(f"Comparing with handcrafted CWLEs from {args.compare}...")
        try:
            with open(args.compare, 'r') as f:
                handcrafted_cwles = json.load(f)
                
            # Generate patterns for handcrafted CWLEs
            handcrafted_patterns = [generator._generate_wave_pattern(cwle) 
                                   for cwle in handcrafted_cwles]
            
            # Calculate distances for both sets
            generator_dist = generator._calculate_distances(patterns)
            handcrafted_dist = generator._calculate_distances(handcrafted_patterns)
            
            # Print comparison metrics
            print("Comparison metrics:")
            print(f"  Generated CWLEs - Min distance: {np.min(generator_dist[generator_dist > 0]):.4f}")
            print(f"  Generated CWLEs - Avg distance: {np.mean(generator_dist[generator_dist > 0]):.4f}")
            print(f"  Handcrafted CWLEs - Min distance: {np.min(handcrafted_dist[handcrafted_dist > 0]):.4f}")
            print(f"  Handcrafted CWLEs - Avg distance: {np.mean(handcrafted_dist[handcrafted_dist > 0]):.4f}")
        except Exception as e:
            print(f"Error comparing CWLEs: {e}")

if __name__ == "__main__":
    main()