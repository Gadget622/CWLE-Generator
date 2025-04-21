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

## Reproducibility with Random Seeds

CWLE Generator now supports deterministic pattern generation through the use of random seeds, ensuring consistent results across multiple runs.

### Configuration

Random seeds can be set in two ways:

1. **Via configuration file** (`cwle_config.yaml`):
   ```yaml
   # Basic Parameters
   num_classes: 10
   img_size: 32
   num_waves_per_cwle: 1
   random_seed: 42        # Set a specific seed for reproducible results
   ```

2. **Via command line** (overrides config file):
   ```bash
   python run_generator.py --seed 123
   ```

### Use Cases

Setting a fixed random seed is valuable for several scenarios:

- **Reproducible Research**: Generate the same patterns for scientific papers or experiments
- **Development & Testing**: Debug or validate changes to the algorithm with consistent inputs
- **Benchmarking**: Compare performance across different parameter configurations
- **Demonstrations**: Show consistent results in presentations or documentation

### Disabling Fixed Seeds

To generate unique patterns each time (non-deterministic behavior), set the seed to `null`:

```yaml
random_seed: null   # Will use different randomization each run
```

### Visualization Consistency

For the t-SNE visualization, you can also set a fixed random seed:

```bash
python visualization_tsne.py --tsne-seed 42
```

This ensures that the 2D projection coordinates remain consistent across visualizations, which is especially useful when making comparisons between different runs or parameter settings.

### Example Usage Scenarios

#### Scenario 1: Comparing optimization parameters

```bash
# First run with default parameters
cp cwle_config.yaml config_default.yaml
python run_generator.py --config config_default.yaml --output cwles_default.json --seed 42

# Run with modified force parameters
cp cwle_config.yaml config_modified.yaml
# Edit config_modified.yaml to change force parameters
python run_generator.py --config config_modified.yaml --output cwles_modified.json --seed 42

# Compare results (same seed ensures fair comparison)
python visualization_tsne.py --generated cwles_default.json --handcrafted cwles_modified.json
```

#### Scenario 2: Creating multiple reproducible sets

```bash
# Generate first set
python run_generator.py --seed 100 --output cwles_set1.json

# Generate second set
python run_generator.py --seed 200 --output cwles_set2.json 

# Each set will be different but internally consistent across multiple runs
```

## Visualization

The package includes tools to visualize:
- Generated CWLE patterns
- Convergence metrics during optimization
- 2D projection using t-SNE for comparison with handcrafted encodings

## Output Organization

CWLE Generator now organizes all outputs into timestamped directories for better tracking and reproducibility.

### Directory Structure

Each run creates a new timestamped directory under the `output` folder:

```
output/
└── execution_YYYYMMDD_HHMMSS/
    ├── execution.log            # Complete log of the execution
    ├── config_used.yaml         # The config actually used (with command-line overrides)
    ├── cwles.json               # Generated CWLE parameters
    ├── cwle_patterns.png        # Visualization of the patterns
    ├── cwle_convergence.png     # Convergence metrics plot
    ├── convergence_metrics.csv  # Detailed convergence data for each iteration
    ├── final_metrics.json       # Final optimization metrics
    ├── distance_analysis.txt    # Text report of pattern distances
    ├── distance_matrix.npy      # Numpy array of distances between patterns
    └── tsne_visualization.png   # t-SNE visualization (if run)
```

### Logging System

The execution log captures:

1. **Command Information**:
   - The exact command used to run the script
   - All command-line arguments
   
2. **Configuration**:
   - Complete configuration settings used
   - Any overrides applied via command line

3. **Execution Progress**:
   - Initialization steps
   - Optimization iterations
   - Final metrics
   
4. **Results and Outputs**:
   - Paths to all generated files
   - Summary of key metrics

### Example Output Log

```
2023-10-15 14:32:01 - root - INFO - Command: python run_generator.py --visualize --seed 42
2023-10-15 14:32:01 - root - INFO - Arguments: Namespace(config='cwle_config.yaml', output='cwles.json', visualize=True, no_convergence=False, compare=None, seed=42)
2023-10-15 14:32:01 - root - INFO - Output directory: output/execution_20231015_143201
2023-10-15 14:32:01 - root - INFO - Setting random seed to 42 (from command line)
2023-10-15 14:32:01 - __main__ - INFO - Initializing CWLE Generator...
2023-10-15 14:32:01 - __main__ - INFO - Configuration loaded from output/execution_20231015_143201/temp_config.yaml
2023-10-15 14:32:01 - __main__ - INFO - Using random seed: 42
2023-10-15 14:32:01 - __main__ - INFO - Initialized 10 CWLEs with 1 waves each
2023-10-15 14:32:01 - __main__ - INFO - Starting optimization with 1000 iterations
2023-10-15 14:32:01 - __main__ - INFO - Iteration 0, Min distance: 0.1234, Avg distance: 0.2345
...
```

### Benefits

This structured output approach provides several advantages:

1. **Reproducibility**: Each run's parameters are preserved
2. **Traceability**: Clear logs of execution history
3. **Comparison**: Easy side-by-side analysis of different runs
4. **Data Preservation**: All intermediate and final results are saved

## License

This project is licensed under the MIT License - see the LICENSE file for details.