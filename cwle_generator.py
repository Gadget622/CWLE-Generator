import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import yaml
import os
import random
from scipy.spatial.distance import pdist, squareform

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import yaml
import os
import random
import logging
from scipy.spatial.distance import pdist, squareform

class CWLEGenerator:
    """
    Composite Wave Label Encoding (CWLE) Generator
    
    This class generates optimized CWLEs for a specified number of classes,
    maximizing the distinctness between labels while adhering to constraints.
    """
    
    def __init__(self, config_path, output_dir=None, logger=None):
        """
        Initialize the CWLE Generator with configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            output_dir: Directory to save outputs (default: None)
            logger: Logger instance (default: None)
        """
        # Setup output directory
        self.output_dir = output_dir if output_dir else "."
        
        # Setup logger
        self.logger = logger if logger else logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Log configuration
        self.logger.info(f"Configuration loaded from {config_path}")
        self.logger.info(f"Configuration: {self.config}")
        
        # Extract configuration parameters
        self.num_classes = self.config['num_classes']
        self.img_size = self.config['img_size']
        self.num_waves_per_cwle = self.config['num_waves_per_cwle']
        self.optimization_iterations = self.config['optimization']['iterations']
        
        # Set random seed if provided
        self.random_seed = self.config.get('random_seed', None)
        if self.random_seed is not None:
            self.logger.info(f"Using random seed: {self.random_seed}")
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
        else:
            self.logger.info("No random seed provided, using random behavior")
        
        # Parameter constraints
        self.constraints = self.config['constraints']
        
        # Force hyperparameters
        self.force_params = self.config['optimization']['forces']
        
        # Initialize convergence tracking
        self.min_distances = []
        self.avg_distances = []
        self.center_distances = []
    
    def generate_cwles(self):
        """
        Generate optimized CWLEs for the specified number of classes.
        
        Returns:
            tuple: (cwles, patterns) - The optimized CWLE parameters and resulting patterns
        """
        # Initialize CWLE parameters
        cwles = self._initialize_cwles()
        
        # Optimize for distinctness
        cwles = self._optimize_cwle_distinctness(cwles)
        
        # Generate final patterns
        patterns = [self._generate_wave_pattern(cwle) for cwle in cwles]
        
        return cwles, patterns
    
    def _initialize_cwles(self):
        """
        Initialize CWLEs with random parameters within constraints.
        
        Returns:
            list: List of initialized CWLEs
        """
        cwles = []
        
        for i in range(self.num_classes):
            cwle = []
            for j in range(self.num_waves_per_cwle):
                # Initialize with random parameters within constraints
                wave = {
                    "frequency": np.random.uniform(
                        self.constraints['min_frequency'] / self.img_size, 
                        self.constraints['max_frequency']
                    ),
                    "orientation": np.random.uniform(
                        self.constraints['min_orientation'], 
                        self.constraints['max_orientation']
                    ),
                    "phase": 0,  # Will be set after frequency is finalized
                    "weight": np.random.uniform(
                        self.constraints['min_weight'], 
                        self.constraints['max_weight']
                    )
                }
                
                # Set phase based on frequency to avoid redundancy
                wave["phase"] = np.random.uniform(0, 0.5 / wave["frequency"])
                cwle.append(wave)
            
            # Normalize weights to sum to 1
            self._normalize_weights(cwle)
            cwles.append(cwle)
        
        self.logger.info(f"Initialized {len(cwles)} CWLEs with {self.num_waves_per_cwle} waves each")
        return cwles
    
    def _normalize_weights(self, cwle):
        """
        Normalize weights to sum to 1 within a CWLE.
        
        Args:
            cwle: A single CWLE to normalize
        """
        total_weight = sum(wave["weight"] for wave in cwle)
        for wave in cwle:
            wave["weight"] = wave["weight"] / total_weight
    
    def _generate_wave_pattern(self, cwle):
        """
        Generate a 2D pattern from a CWLE.
        
        Args:
            cwle: A single CWLE
            
        Returns:
            numpy.ndarray: 2D array of pixel values
        """
        # Initialize empty pattern
        pattern = np.zeros((self.img_size, self.img_size))
        
        # Create coordinate grid
        x = np.arange(self.img_size)
        y = np.arange(self.img_size)
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
    
    def _calculate_center_of_mass(self, patterns):
        """
        Calculate the center of mass of the patterns in the latent space.
        
        Args:
            patterns: List of pattern arrays
            
        Returns:
            numpy.ndarray: Center of mass
        """
        # Flatten patterns for simpler computation
        flat_patterns = [pattern.flatten() for pattern in patterns]
        
        # Calculate average
        center = np.mean(flat_patterns, axis=0)
        
        return center
    
    def _calculate_distances(self, patterns):
        """
        Calculate pairwise distances between all patterns.
        
        Args:
            patterns: List of pattern arrays
            
        Returns:
            numpy.ndarray: Distance matrix
        """
        # Flatten patterns for distance calculation
        flat_patterns = [pattern.flatten() for pattern in patterns]
        
        # Calculate pairwise Euclidean distances
        dist_condensed = pdist(flat_patterns, metric='euclidean')
        dist_matrix = squareform(dist_condensed)
        
        return dist_matrix
    
    def _calculate_forces(self, patterns, dist_matrix):
        """
        Calculate forces for each pattern based on current state.
        
        Args:
            patterns: List of pattern arrays
            dist_matrix: Distance matrix between patterns
            
        Returns:
            list: Force vectors for each pattern
        """
        flat_patterns = [pattern.flatten() for pattern in patterns]
        center_of_mass = self._calculate_center_of_mass(patterns)
        
        # Center of latent space (all pixel values = 0.5)
        center_of_space = np.ones_like(center_of_mass) * 0.5
        
        forces = []
        
        for i, pattern in enumerate(flat_patterns):
            # Initialize force vector
            force = np.zeros_like(pattern)
            
            # 1. Centering force: pulls the center of mass toward the center of latent space
            com_to_cos = center_of_space - center_of_mass
            centering_force = self.force_params['centering'] * com_to_cos
            
            # 2. Expanding force: pushes nodes away from center of mass
            node_to_com = pattern - center_of_mass
            node_to_com_norm = np.linalg.norm(node_to_com)
            if node_to_com_norm > 0:
                expanding_force = self.force_params['expanding'] * (node_to_com / node_to_com_norm)
            else:
                expanding_force = np.zeros_like(pattern)
            
            # 3. Equalizing force: adjusts nodes to make distances more uniform
            equalizing_force = np.zeros_like(pattern)
            for j, other_pattern in enumerate(flat_patterns):
                if i != j:
                    # Calculate distance and direction to other pattern
                    diff = other_pattern - pattern
                    dist = dist_matrix[i, j]
                    
                    # Calculate average distance to all other patterns
                    avg_dist = np.mean(dist_matrix[i, :][dist_matrix[i, :] > 0])
                    
                    # If distance is less than average, move away; otherwise, move closer
                    if dist < avg_dist:
                        # Move away from closer patterns
                        if np.linalg.norm(diff) > 0:
                            equalizing_force -= self.force_params['equalizing'] * (diff / np.linalg.norm(diff))
                    else:
                        # Move toward distant patterns
                        if np.linalg.norm(diff) > 0:
                            equalizing_force += self.force_params['equalizing'] * (diff / np.linalg.norm(diff))
            
            # Combine forces
            force = centering_force + expanding_force + equalizing_force
            forces.append(force)
        
        return forces
    
    def _generate_parameter_candidates(self, cwle):
        """
        Generate parameter candidates for a CWLE.
        
        Args:
            cwle: A single CWLE
            
        Returns:
            list: List of candidate CWLEs
        """
        candidates = []
        step_size = self.config['optimization']['step_size']
        
        # For each wave in the CWLE
        for wave_idx, wave in enumerate(cwle):
            # For each parameter
            for param in ['frequency', 'orientation', 'phase', 'weight']:
                # Create two candidates (increase and decrease)
                for direction in [-1, 1]:
                    candidate = [w.copy() for w in cwle]  # Deep copy
                    
                    # Adjust parameter
                    if param == 'frequency':
                        new_val = wave[param] + direction * step_size * wave[param]
                        # Enforce constraints
                        new_val = max(self.constraints['min_frequency'] / self.img_size, 
                                    min(self.constraints['max_frequency'], new_val))
                        candidate[wave_idx][param] = new_val
                        
                        # Update phase if frequency changed
                        candidate[wave_idx]['phase'] = min(candidate[wave_idx]['phase'], 
                                                        0.5 / new_val)
                    
                    elif param == 'orientation':
                        new_val = wave[param] + direction * step_size * 10  # Larger step for orientation
                        # Wrap around to stay in [0, 180)
                        new_val = new_val % 180
                        candidate[wave_idx][param] = new_val
                    
                    elif param == 'phase':
                        max_phase = 0.5 / wave['frequency']
                        new_val = wave[param] + direction * step_size * max_phase
                        # Enforce constraints
                        new_val = max(0, min(max_phase, new_val))
                        candidate[wave_idx][param] = new_val
                    
                    elif param == 'weight':
                        new_val = wave[param] + direction * step_size * 0.1  # Smaller step for weight
                        # Enforce constraints
                        new_val = max(self.constraints['min_weight'], 
                                    min(self.constraints['max_weight'], new_val))
                        candidate[wave_idx][param] = new_val
                        
                        # Renormalize weights
                        self._normalize_weights(candidate)
                    
                    candidates.append(candidate)
        
        return candidates
    
    def _select_best_candidate(self, candidates, cwle, force_vector, patterns):
        """
        Select the best candidate that moves in the direction of the force.
        
        Args:
            candidates: List of candidate CWLEs
            cwle: Current CWLE
            force_vector: Desired direction of movement
            patterns: Current patterns (needed for calculating center of mass)
            
        Returns:
            dict: Best candidate CWLE
        """
        best_projection = -float('inf')
        best_candidate = None
        
        # Generate the current pattern
        current_pattern = self._generate_wave_pattern(cwle).flatten()
        
        # Normalize force vector
        force_norm = np.linalg.norm(force_vector)
        if force_norm > 0:
            force_vector = force_vector / force_norm
        
        for candidate in candidates:
            # Generate candidate pattern
            candidate_pattern = self._generate_wave_pattern(candidate).flatten()
            
            # Calculate movement vector
            movement = candidate_pattern - current_pattern
            
            # Calculate projection of movement onto force vector
            projection = np.dot(movement, force_vector)
            
            if projection > best_projection:
                best_projection = projection
                best_candidate = candidate
        
        # If no improvement, keep original
        if best_projection <= 0 or best_candidate is None:
            return cwle
            
        return best_candidate
    
    def _optimize_cwle_distinctness(self, cwles):
        """
        Optimize CWLEs for distinctness using force-directed approach.
        
        Args:
            cwles: Initial set of CWLEs
            
        Returns:
            list: Optimized CWLEs
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting optimization with {self.optimization_iterations} iterations")
        
        # Create a file for tracking convergence metrics
        convergence_file = os.path.join(self.output_dir, "convergence_metrics.csv")
        with open(convergence_file, 'w') as f:
            f.write("Iteration,MinDistance,AvgDistance,CenterDistance\n")
        
        for iteration in range(self.optimization_iterations):
            # Generate current patterns
            patterns = [self._generate_wave_pattern(cwle) for cwle in cwles]
            
            # Calculate distance matrix
            dist_matrix = self._calculate_distances(patterns)
            
            # Track convergence metrics
            min_dist = np.min(dist_matrix[dist_matrix > 0])
            avg_dist = np.mean(dist_matrix[dist_matrix > 0])
            self.min_distances.append(min_dist)
            self.avg_distances.append(avg_dist)
            
            # Calculate center of mass and its distance from center of space
            com = self._calculate_center_of_mass(patterns)
            center_of_space = np.ones_like(com) * 0.5
            com_distance = np.linalg.norm(com - center_of_space)
            self.center_distances.append(com_distance)
            
            # Log progress to convergence file
            with open(convergence_file, 'a') as f:
                f.write(f"{iteration},{min_dist},{avg_dist},{com_distance}\n")
            
            # Log progress
            if iteration % 10 == 0:
                self.logger.info(f"Iteration {iteration}, Min distance: {min_dist:.4f}, Avg distance: {avg_dist:.4f}")
                self.logger.info(f"Center of mass distance from center: {com_distance:.4f}")
            
            # Calculate forces for each pattern
            forces = self._calculate_forces(patterns, dist_matrix)
            
            # Generate candidates and select best for each CWLE
            new_cwles = []
            for i, cwle in enumerate(cwles):
                candidates = self._generate_parameter_candidates(cwle)
                best_candidate = self._select_best_candidate(candidates, cwle, forces[i], patterns)
                new_cwles.append(best_candidate)
            
            # Update CWLEs
            cwles = new_cwles
        
        # Calculate total optimization time
        end_time = time.time()
        total_time = end_time - start_time
        time_per_iteration = total_time / self.optimization_iterations
        
        self.logger.info(f"Optimization completed in {total_time:.2f} seconds")
        self.logger.info(f"Average time per iteration: {time_per_iteration:.4f} seconds")
        
        # Save final convergence metrics
        final_metrics = {
            "min_distance": float(self.min_distances[-1]),
            "avg_distance": float(self.avg_distances[-1]),
            "center_distance": float(self.center_distances[-1]),
            "optimization_time": float(total_time),
            "time_per_iteration": float(time_per_iteration)
        }
        metrics_file = os.path.join(self.output_dir, "final_metrics.json")
        with open(metrics_file, 'w') as f:
            import json
            json.dump(final_metrics, f, indent=4)
        
        return cwles
        
    def visualize_patterns(self, patterns, filename=None):
        """
        Visualize the generated patterns.
        
        Args:
            patterns: List of pattern arrays
            filename: Optional filename to save the visualization
        """
        n = len(patterns)
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        
        plt.figure(figsize=(cols * 3, rows * 3))
        
        for i, pattern in enumerate(patterns):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(pattern, cmap='viridis', interpolation='nearest')
            plt.title(f"Class {i}")
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            self.logger.info(f"Visualization saved to {filename}")
        else:
            plt.show()
    
    def visualize_convergence(self, filename=None):
        """
        Visualize the convergence of the optimization process.
        
        Args:
            filename: Optional filename to save the visualization
        """
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.min_distances, label='Min Distance')
        plt.plot(self.avg_distances, label='Avg Distance')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.title('Distance Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(self.center_distances, label='Center of Mass Distance')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.title('Center of Mass Distance from Center of Space')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            self.logger.info(f"Convergence plot saved to {filename}")
        else:
            plt.show()


def main():
    """
    Main function to run the CWLE Generator.
    """
    config_path = "cwle_config.yaml"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found.")
        return
    
    # Initialize generator
    generator = CWLEGenerator(config_path)
    
    # Generate CWLEs
    print("Generating CWLEs...")
    cwles, patterns = generator.generate_cwles()
    
    # Visualize patterns
    print("Visualizing patterns...")
    generator.visualize_patterns(patterns, "cwle_patterns.png")
    
    # Visualize convergence
    print("Visualizing convergence...")
    generator.visualize_convergence("cwle_convergence.png")
    
    # Save CWLEs to file
    import json
    with open("cwles.json", "w") as f:
        json.dump(cwles, f, indent=4)
    print("CWLEs saved to cwles.json")

if __name__ == "__main__":
    main()