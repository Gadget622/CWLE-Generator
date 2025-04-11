import os
import json
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

class CWLEModel(QObject):
    """
    Model class for CWLE application.
    Handles data management and CWLE generation.
    """
    # Signals
    cwles_generated = pyqtSignal(list, list)  # Emitted when new CWLEs are generated
    config_updated = pyqtSignal(dict)         # Emitted when config is updated
    
    def __init__(self):
        super().__init__()
        
        # Default configuration values
        self.config = {
            "num_classes": 10,
            "img_size": 28,
            "num_waves_per_cwle": 2,
            "constraints": {
                "min_frequency": 2,
                "max_frequency": 0.5,
                "min_orientation": 0,
                "max_orientation": 180,
                "min_weight": 0.05,
                "max_weight": 1.0
            }
        }
        
        # Store generated CWLEs and patterns
        self.cwles = []
        self.patterns = []
        self.current_index = 0
    
    def update_config(self, config):
        """
        Update the configuration.
        
        Args:
            config (dict): New configuration values
        """
        self.config.update(config)
        self.config_updated.emit(self.config)
    
    def generate_cwles(self):
        """
        Generate CWLEs based on current configuration.
        """
        self.cwles = []
        self.patterns = []
        
        num_classes = self.config["num_classes"]
        img_size = self.config["img_size"]
        waves_per_cwle = self.config["num_waves_per_cwle"]
        constraints = self.config["constraints"]
        
        for i in range(num_classes):
            cwle = []
            for j in range(waves_per_cwle):
                # Generate wave with random parameters within constraints
                wave = {
                    "frequency": np.random.uniform(
                        constraints["min_frequency"] / img_size, 
                        constraints["max_frequency"]
                    ),
                    "orientation": np.random.uniform(
                        constraints["min_orientation"], 
                        constraints["max_orientation"]
                    ),
                    "phase": 0,  # Will be set after frequency is finalized
                    "weight": np.random.uniform(
                        constraints["min_weight"], 
                        constraints["max_weight"]
                    )
                }
                
                # Set phase based on frequency to avoid redundancy
                wave["phase"] = np.random.uniform(0, 0.5 / wave["frequency"])
                cwle.append(wave)
            
            # Normalize weights to sum to 1
            self._normalize_weights(cwle)
            self.cwles.append(cwle)
            
            # Generate pattern
            pattern = self._generate_wave_pattern(cwle, img_size)
            self.patterns.append(pattern)
        
        self.current_index = 0
        self.cwles_generated.emit(self.cwles, self.patterns)
        return self.cwles, self.patterns
    
    def _normalize_weights(self, cwle):
        """
        Normalize weights to sum to 1 within a CWLE.
        
        Args:
            cwle: A single CWLE to normalize
        """
        total_weight = sum(wave["weight"] for wave in cwle)
        for wave in cwle:
            wave["weight"] = wave["weight"] / total_weight
    
    def _generate_wave_pattern(self, cwle, img_size):
        """
        Generate a 2D pattern from a CWLE.
        
        Args:
            cwle: A single CWLE
            img_size: Size of the image
            
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
    
    def generate_single_wave_pattern(self, wave, img_size):
        """
        Generate a pattern for a single wave component.
        
        Args:
            wave: Wave parameters dictionary
            img_size: Size of the image
            
        Returns:
            numpy.ndarray: 2D array of pixel values
        """
        # Create coordinate grid
        x = np.arange(img_size)
        y = np.arange(img_size)
        X, Y = np.meshgrid(x, y)
        
        freq = wave["frequency"]
        theta = wave["orientation"] * (np.pi / 180)  # Convert to radians
        phase = wave["phase"]
        
        # Apply rotation to coordinates
        X_rot = X * np.cos(theta) + Y * np.sin(theta)
        
        # Calculate wave value at this position
        wave_values = np.sin(2 * np.pi * freq * X_rot + phase)
        
        # Normalize from [-1,1] to [0,1] range
        pattern = (wave_values + 1) / 2
        
        return pattern
    
    def get_current_cwle(self):
        """
        Get the currently selected CWLE.
        
        Returns:
            tuple: (cwle, pattern, index)
        """
        if not self.cwles:
            return None, None, 0
            
        return self.cwles[self.current_index], self.patterns[self.current_index], self.current_index
    
    def go_to_next_cwle(self):
        """
        Switch to the next CWLE.
        
        Returns:
            tuple: (cwle, pattern, index)
        """
        if not self.cwles:
            return None, None, 0
            
        self.current_index = (self.current_index + 1) % len(self.cwles)
        return self.cwles[self.current_index], self.patterns[self.current_index], self.current_index
    
    def go_to_previous_cwle(self):
        """
        Switch to the previous CWLE.
        
        Returns:
            tuple: (cwle, pattern, index)
        """
        if not self.cwles:
            return None, None, 0
            
        self.current_index = (self.current_index - 1) % len(self.cwles)
        return self.cwles[self.current_index], self.patterns[self.current_index], self.current_index
    
    def save_cwles(self, filename):
        """
        Save generated CWLEs to a JSON file.
        
        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(self.cwles, f, indent=2)
    
    def save_images(self, output_dir):
        """
        Save patterns as image files.
        
        Args:
            output_dir: Directory to save images
        """
        import matplotlib.pyplot as plt
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual images
        for i, pattern in enumerate(self.patterns):
            plt.figure(figsize=(5, 5))
            plt.imshow(pattern, cmap='viridis', interpolation='nearest')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"cwle_{i}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save combined image
        n = len(self.patterns)
        grid_size = int(np.ceil(np.sqrt(n)))
        
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(2*grid_size, 2*grid_size))
        axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
        
        for i, pattern in enumerate(self.patterns):
            if i < len(axs):
                axs[i].imshow(pattern, cmap='viridis', interpolation='nearest')
                axs[i].set_title(f"Class {i}")
                axs[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(self.patterns), len(axs)):
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cwle_combined.png"), dpi=150, bbox_inches='tight')
        plt.close()