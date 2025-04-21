#!/usr/bin/env python3

"""
Run script for CWLE Generator

This script runs the CWLE Generator with the specified configuration file.
It also provides options for visualizing the results.
"""

import argparse
import os
import sys
import json
import yaml
import numpy as np
import datetime
import shutil
import logging
from cwle_generator import CWLEGenerator

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
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed to use (overrides config file setting)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes to generate')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Image size (width/height)')
    parser.add_argument('--num-waves', type=int, default=None,
                        help='Number of waves per CWLE')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of optimization iterations')
    parser.add_argument('--step-size', type=float, default=None,
                        help='Step size for parameter updates')
    parser.add_argument('--centering-force', type=float, default=None,
                        help='Centering force strength')
    parser.add_argument('--expanding-force', type=float, default=None,
                        help='Expanding force strength')
    parser.add_argument('--equalizing-force', type=float, default=None,
                        help='Equalizing force strength')
    
    return parser.parse_args()

def setup_output_directory():
    """Create a timestamped output directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(".output", f"execution_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_logging(output_dir, args):
    """Set up logging to file."""
    log_path = os.path.join(output_dir, "execution.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log the command used to run the script
    cmd_line = f"python {sys.argv[0]} " + " ".join(sys.argv[1:])
    logging.info(f"Command: {cmd_line}")
    logging.info(f"Arguments: {args}")
    
    return logging.getLogger(__name__)

def main():
    """Main function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = setup_output_directory()
    
    # Setup logging
    logger = setup_logging(output_dir, args)
    logger.info(f"Output directory: {output_dir}")
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file {args.config} not found.")
        return
    
    # Load the config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Make a copy of the original config
    original_config = config.copy()
    
    # Update config with command line parameters
    if args.seed is not None:
        config['random_seed'] = args.seed
        logger.info(f"Setting random seed to {args.seed} (from command line)")
    
    # Apply new hyperparameters if provided
    if args.num_classes is not None:
        config['num_classes'] = args.num_classes
        logger.info(f"Setting num_classes to {args.num_classes} (from command line)")
        
    if args.img_size is not None:
        config['img_size'] = args.img_size
        logger.info(f"Setting img_size to {args.img_size} (from command line)")
        
    if args.num_waves is not None:
        config['num_waves_per_cwle'] = args.num_waves
        logger.info(f"Setting num_waves_per_cwle to {args.num_waves} (from command line)")
        
    if args.iterations is not None:
        config['optimization']['iterations'] = args.iterations
        logger.info(f"Setting iterations to {args.iterations} (from command line)")
        
    if args.step_size is not None:
        config['optimization']['step_size'] = args.step_size
        logger.info(f"Setting step_size to {args.step_size} (from command line)")
        
    if args.centering_force is not None:
        config['optimization']['forces']['centering'] = args.centering_force
        logger.info(f"Setting centering force to {args.centering_force} (from command line)")
        
    if args.expanding_force is not None:
        config['optimization']['forces']['expanding'] = args.expanding_force
        logger.info(f"Setting expanding force to {args.expanding_force} (from command line)")
        
    if args.equalizing_force is not None:
        config['optimization']['forces']['equalizing'] = args.equalizing_force
        logger.info(f"Setting equalizing force to {args.equalizing_force} (from command line)")
    
    # Save the modified config to the output directory
    config_output_path = os.path.join(output_dir, "config_used.yaml")
    with open(config_output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create a temporary config file for generator
    temp_config_path = os.path.join(output_dir, "temp_config.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # The rest of the function remains the same...
    
    # Initialize generator with the temporary config
    logger.info("Initializing CWLE Generator...")
    generator = CWLEGenerator(temp_config_path, output_dir=output_dir, logger=logger)
    
    # Generate CWLEs
    logger.info("Generating CWLEs...")
    cwles, patterns = generator.generate_cwles()
    
    # Save CWLEs to output directory
    output_file = os.path.join(output_dir, "cwles.json")
    with open(output_file, "w") as f:
        json.dump(cwles, f, indent=4)
    logger.info(f"CWLEs saved to {output_file}")
    
    # Visualize patterns if requested
    if args.visualize:
        logger.info("Visualizing patterns...")
        patterns_file = os.path.join(output_dir, "cwle_patterns.png")
        generator.visualize_patterns(patterns, patterns_file)
    
    # Visualize convergence if requested
    if not args.no_convergence:
        logger.info("Visualizing convergence...")
        convergence_file = os.path.join(output_dir, "cwle_convergence.png")
        generator.visualize_convergence(convergence_file)
    
    # Compare with handcrafted CWLEs if provided
    if args.compare and os.path.exists(args.compare):
        logger.info(f"Comparing with handcrafted CWLEs from {args.compare}...")
        try:
            # Copy the comparison file to the output directory
            compare_file_name = os.path.basename(args.compare)
            compare_output_path = os.path.join(output_dir, compare_file_name)
            shutil.copy(args.compare, compare_output_path)
            
            with open(args.compare, 'r') as f:
                handcrafted_cwles = json.load(f)
                
            # Generate patterns for handcrafted CWLEs
            handcrafted_patterns = [generator._generate_wave_pattern(cwle) 
                                   for cwle in handcrafted_cwles]
            
            # Calculate distances for both sets
            generator_dist = generator._calculate_distances(patterns)
            handcrafted_dist = generator._calculate_distances(handcrafted_patterns)
            
            # Log comparison metrics
            min_gen_dist = np.min(generator_dist[generator_dist > 0])
            avg_gen_dist = np.mean(generator_dist[generator_dist > 0])
            min_hand_dist = np.min(handcrafted_dist[handcrafted_dist > 0])
            avg_hand_dist = np.mean(handcrafted_dist[handcrafted_dist > 0])
            
            logger.info("Comparison metrics:")
            logger.info(f"  Generated CWLEs - Min distance: {min_gen_dist:.4f}")
            logger.info(f"  Generated CWLEs - Avg distance: {avg_gen_dist:.4f}")
            logger.info(f"  Handcrafted CWLEs - Min distance: {min_hand_dist:.4f}")
            logger.info(f"  Handcrafted CWLEs - Avg distance: {avg_hand_dist:.4f}")
            
            # Save comparison metrics to file
            comparison_file = os.path.join(output_dir, "comparison_metrics.json")
            comparison_metrics = {
                "generated": {
                    "min_distance": float(min_gen_dist),
                    "avg_distance": float(avg_gen_dist)
                },
                "handcrafted": {
                    "min_distance": float(min_hand_dist),
                    "avg_distance": float(avg_hand_dist)
                }
            }
            with open(comparison_file, "w") as f:
                json.dump(comparison_metrics, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error comparing CWLEs: {e}")
    
    logger.info(f"Execution completed. All outputs saved to {output_dir}")
    
    # Print final message to console
    print(f"\nExecution completed successfully.")
    print(f"All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()