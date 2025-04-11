class CWLEController:
    """
    Controller class for CWLE application.
    Connects the model and view components.
    """
    
    def __init__(self, model, view):
        """
        Initialize the controller.
        
        Args:
            model: CWLEModel instance
            view: MainWindow instance
        """
        self.model = model
        self.view = view
        
        # Connect view signals to controller methods
        self.view.generate_cwles_requested.connect(self.generate_cwles)
        self.view.next_cwle_requested.connect(self.show_next_cwle)
        self.view.prev_cwle_requested.connect(self.show_previous_cwle)
        self.view.save_cwles_requested.connect(self.save_cwles)
        self.view.save_images_requested.connect(self.save_images)
        
        # Connect model signals to controller methods
        self.model.cwles_generated.connect(self.on_cwles_generated)
        self.model.config_updated.connect(self.on_config_updated)
        
        # Connect config panel signals
        self.view.config_panel.config_changed.connect(self.update_config)
        
        # Set initial config
        self.view.config_panel.set_config(self.model.config)
    
    def generate_cwles(self):
        """Generate new CWLEs."""
        # Update model with latest config
        config = self.view.config_panel.get_config()
        self.model.update_config(config)
        
        # Generate CWLEs
        self.model.generate_cwles()
    
    def show_next_cwle(self):
        """Show the next CWLE."""
        cwle, pattern, index = self.model.go_to_next_cwle()
        if cwle:
            self.view.update_cwle_view(cwle, pattern, index, len(self.model.cwles))
    
    def show_previous_cwle(self):
        """Show the previous CWLE."""
        cwle, pattern, index = self.model.go_to_previous_cwle()
        if cwle:
            self.view.update_cwle_view(cwle, pattern, index, len(self.model.cwles))
    
    def save_cwles(self, filename):
        """
        Save CWLEs to file.
        
        Args:
            filename: Output filename
        """
        self.model.save_cwles(filename)
    
    def save_images(self, output_dir):
        """
        Save CWLE images to directory.
        
        Args:
            output_dir: Output directory
        """
        self.model.save_images(output_dir)
    
    def update_config(self, config):
        """
        Update the model with new configuration.
        
        Args:
            config: New configuration dictionary
        """
        self.model.update_config(config)
    
    def on_cwles_generated(self, cwles, patterns):
        """
        Handle CWLEs generated event.
        
        Args:
            cwles: List of generated CWLEs
            patterns: List of generated patterns
        """
        # Show first CWLE
        if cwles and patterns:
            self.model.current_index = 0
            self.view.update_cwle_view(cwles[0], patterns[0], 0, len(cwles))
    
    def on_config_updated(self, config):
        """
        Handle config updated event.
        
        Args:
            config: Updated configuration dictionary
        """
        # Update view's config panel
        self.view.config_panel.set_config(config)