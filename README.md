# CWLE Generator

Composite Wave Label Encoding (CWLE) Generator is a tool for automatically generating optimized label encodings for neural network training. It creates a set of wave-based patterns that are maximally distinct from one another, ensuring efficient and accurate learning.

## Overview

CWLE Generator creates encodings by:
1. Initializing a set of wave patterns with random parameters within constraints
2. Optimizing these patterns using a force-directed approach to maximize distinctness
3. Visualizing the results and comparing with handcrafted encodings

## Key Features

- **Force-Directed Optimization**: Uses three forces to shape the distinctness of encodings:
  - Centering force: Pulls the center of mass toward center of latent space
  - Expanding force: Pushes patterns away from center of mass
  - Equalizing force: Makes distances between patterns more uniform

- **Configurable Parameters**: All aspects can be tuned via a YAML configuration file
  
- **Validation Tools**: Includes visualization and comparison capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cwle-generator.git
cd cwle-generator

# Install dependencies
pip install numpy matplotlib pyyaml scikit-learn
```

## Usage

### Basic Usage

```bash
# Generate CWLEs using the default configuration
python run_generator.py
```

### Advanced Usage

```bash
# Specify a custom configuration file
python run_generator.py --config custom_config.yaml

# Visualize the results
python run_generator.py --visualize

# Compare with handcrafted CWLEs
python run_generator.py --compare handcrafted_cwles.json
```

### t-SNE Visualization

```bash
# Visualize CWLEs in 2D space using t-SNE
python visualization_tsne.py --generated cwles.json --handcrafted handcrafted_cwles.json
```

## Configuration

The `cwle_config.yaml` file controls all aspects of the generation:

```yaml
# Basic Parameters
num_classes: 10         # Number of classes
img_size: 28            # Image size (28 for MNIST)
num_waves_per_cwle: 2   # Number of waves per CWLE

# Parameter Constraints
constraints:
  min_frequency: 2      # Minimum frequency
  max_frequency: 0.5    # Maximum frequency
  min_orientation: 0    # Minimum orientation (degrees)
  max_orientation: 180  # Maximum orientation (degrees)
  min_weight: 0.05      # Minimum wave weight
  max_weight: 1.0       # Maximum wave weight

# Optimization Parameters
optimization:
  iterations: 500       # Number of iterations
  step_size: 0.05       # Parameter update step size
  
  # Force parameters (learning rates)
  forces:
    centering: 0.01     # Centering force strength
    expanding: 0.02     # Expanding force strength
    equalizing: 0.015   # Equalizing force strength
```

## Wave Parameters

Each CWLE consists of one or more waves, each defined by four parameters:

1. **Frequency (f)**: Controls how many oscillations appear across the image
   - Range: [2/img_size, 0.5]
   - Lower bound ensures at least 2 cycles across the image
   - Upper bound prevents oscillation within a single pixel

2. **Orientation (θ)**: Controls the angle of the wave pattern
   - Range: [0°, 180°)
   - Restricted to this range to avoid redundancy

3. **Phase (p)**: Controls the offset of the wave
   - Range: [0, 0.5/frequency]
   - Restriction based on frequency to avoid redundancy

4. **Weight (k)**: Controls the contribution of each wave
   - Range: [0.05, 1.0]
   - Weights are normalized to sum to 1 within each CWLE

## Algorithm Explanation

The algorithm optimizes CWLEs using the following steps:

1. **Initialization**: 
   - Random parameters within constraints
   - Weights normalized to sum to 1

2. **Force Calculation**:
   - Centering force pulls center of mass toward center of latent space
   - Expanding force pushes nodes away from center of mass
   - Equalizing force adjusts nodes to make distances more uniform

3. **Parameter Update**:
   - Generate candidates with small parameter variations
   - Select candidate that moves in direction of combined force
   - Apply update to all CWLEs simultaneously

4. **Convergence Tracking**:
   - Monitor minimum and average distances
   - Track center of mass position

## Visualization

The package includes tools to visualize:
- Generated CWLE patterns
- Convergence metrics during optimization
- 2D projection using t-SNE for comparison with handcrafted encodings

## License

This project is licensed under the MIT License - see the LICENSE file for details.