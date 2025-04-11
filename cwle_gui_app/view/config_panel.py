from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, 
    QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal

class ConfigPanel(QWidget):
    """
    Panel for configuring CWLE generation parameters.
    """
    # Signals
    config_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Scroll area for configuration
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Basic parameters group
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QFormLayout()
        
        # Number of classes
        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(1, 100)
        self.num_classes_spin.setValue(10)
        self.num_classes_spin.valueChanged.connect(self.on_config_changed)
        basic_layout.addRow("Number of classes:", self.num_classes_spin)
        
        # Image size
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(8, 512)
        self.img_size_spin.setValue(28)
        self.img_size_spin.valueChanged.connect(self.on_config_changed)
        basic_layout.addRow("Image size:", self.img_size_spin)
        
        # Waves per CWLE
        self.waves_per_cwle_spin = QSpinBox()
        self.waves_per_cwle_spin.setRange(1, 10)
        self.waves_per_cwle_spin.setValue(2)
        self.waves_per_cwle_spin.valueChanged.connect(self.on_config_changed)
        basic_layout.addRow("Waves per CWLE:", self.waves_per_cwle_spin)
        
        basic_group.setLayout(basic_layout)
        scroll_layout.addWidget(basic_group)
        
        # Constraints group
        constraints_group = QGroupBox("Parameter Constraints")
        constraints_layout = QFormLayout()
        
        # Frequency constraints
        self.min_frequency_spin = QDoubleSpinBox()
        self.min_frequency_spin.setRange(0.1, 10)
        self.min_frequency_spin.setValue(2)
        self.min_frequency_spin.setDecimals(2)
        self.min_frequency_spin.valueChanged.connect(self.on_config_changed)
        constraints_layout.addRow("Min frequency:", self.min_frequency_spin)
        
        self.max_frequency_spin = QDoubleSpinBox()
        self.max_frequency_spin.setRange(0.1, 10)
        self.max_frequency_spin.setValue(0.5)
        self.max_frequency_spin.setDecimals(2)
        self.max_frequency_spin.valueChanged.connect(self.on_config_changed)
        constraints_layout.addRow("Max frequency:", self.max_frequency_spin)
        
        # Orientation constraints
        self.min_orientation_spin = QDoubleSpinBox()
        self.min_orientation_spin.setRange(0, 179.9)
        self.min_orientation_spin.setValue(0)
        self.min_orientation_spin.setDecimals(1)
        self.min_orientation_spin.valueChanged.connect(self.on_config_changed)
        constraints_layout.addRow("Min orientation (°):", self.min_orientation_spin)
        
        self.max_orientation_spin = QDoubleSpinBox()
        self.max_orientation_spin.setRange(0.1, 180)
        self.max_orientation_spin.setValue(180)
        self.max_orientation_spin.setDecimals(1)
        self.max_orientation_spin.valueChanged.connect(self.on_config_changed)
        constraints_layout.addRow("Max orientation (°):", self.max_orientation_spin)
        
        # Weight constraints
        self.min_weight_spin = QDoubleSpinBox()
        self.min_weight_spin.setRange(0.001, 0.999)
        self.min_weight_spin.setValue(0.05)
        self.min_weight_spin.setDecimals(3)
        self.min_weight_spin.setSingleStep(0.01)
        self.min_weight_spin.valueChanged.connect(self.on_config_changed)
        constraints_layout.addRow("Min weight:", self.min_weight_spin)
        
        self.max_weight_spin = QDoubleSpinBox()
        self.max_weight_spin.setRange(0.002, 1.0)
        self.max_weight_spin.setValue(1.0)
        self.max_weight_spin.setDecimals(3)
        self.max_weight_spin.setSingleStep(0.01)
        self.max_weight_spin.valueChanged.connect(self.on_config_changed)
        constraints_layout.addRow("Max weight:", self.max_weight_spin)
        
        constraints_group.setLayout(constraints_layout)
        scroll_layout.addWidget(constraints_group)
        
        # Add explanations
        explanation_label = QLabel(
            "Notes:\n"
            "- Min frequency is divided by image size to ensure at least N cycles across the image\n"
            "- Max frequency should be <= 0.5 to prevent oscillation within a single pixel\n"
            "- Orientations range from 0° to 180° (exclusive)\n"
            "- Weights are normalized to sum to 1 within each CWLE\n"
            "- Phase is automatically calculated based on frequency\n"
        )
        explanation_label.setWordWrap(True)
        scroll_layout.addWidget(explanation_label)
        
        # Add stretch at the end
        scroll_layout.addStretch()
        
    def on_config_changed(self, config):
        """
        Handle config updated event.
        
        Args:
            config: Updated configuration dictionary
        """
        # Block signals to prevent recursive calls
        self.view.config_panel.blockSignals(True)
        self.view.config_panel.set_config(config)
        self.view.config_panel.blockSignals(False)
        
    def get_config(self):
        """
        Get the current configuration values.
        
        Returns:
            dict: Configuration dictionary
        """
        return {
            "num_classes": self.num_classes_spin.value(),
            "img_size": self.img_size_spin.value(),
            "num_waves_per_cwle": self.waves_per_cwle_spin.value(),
            "constraints": {
                "min_frequency": self.min_frequency_spin.value(),
                "max_frequency": self.max_frequency_spin.value(),
                "min_orientation": self.min_orientation_spin.value(),
                "max_orientation": self.max_orientation_spin.value(),
                "min_weight": self.min_weight_spin.value(),
                "max_weight": self.max_weight_spin.value()
            }
        }
    
    def set_config(self, config):
        """
        Set configuration values.
        
        Args:
            config (dict): Configuration dictionary
        """
        # Block signals temporarily to avoid triggering on_config_changed multiple times
        self.blockSignals(True)
        
        # Set basic parameters
        self.num_classes_spin.setValue(config.get("num_classes", 10))
        self.img_size_spin.setValue(config.get("img_size", 28))
        self.waves_per_cwle_spin.setValue(config.get("num_waves_per_cwle", 2))
        
        # Set constraints
        constraints = config.get("constraints", {})
        self.min_frequency_spin.setValue(constraints.get("min_frequency", 2))
        self.max_frequency_spin.setValue(constraints.get("max_frequency", 0.5))
        self.min_orientation_spin.setValue(constraints.get("min_orientation", 0))
        self.max_orientation_spin.setValue(constraints.get("max_orientation", 180))
        self.min_weight_spin.setValue(constraints.get("min_weight", 0.05))
        self.max_weight_spin.setValue(constraints.get("max_weight", 1.0))
        
        self.blockSignals(False)
        