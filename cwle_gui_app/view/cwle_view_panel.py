import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QScrollArea, QGroupBox, QSplitter
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

class WaveImageWidget(QLabel):
    """Widget for displaying a wave pattern image."""
    
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(150, 150)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5;")
    
    def set_image(self, pattern):
        """
        Set the image from a numpy array pattern.
        
        Args:
            pattern: 2D numpy array with values in [0, 1]
        """
        if pattern is None:
            self.clear()
            return
            
        # Scale to [0, 255] for QImage
        img_data = (pattern * 255).astype(np.uint8)
        h, w = img_data.shape
        
        # Create colormap (viridis-like)
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Simple colormap: blue to green to yellow
        for i in range(h):
            for j in range(w):
                val = img_data[i, j]
                if val < 64:  # Dark blue to blue
                    r, g, b = 0, 0, val * 4
                elif val < 128:  # Blue to teal
                    r, g, b = 0, (val - 64) * 4, 255
                elif val < 192:  # Teal to green to yellow
                    r, g, b = (val - 128) * 4, 255, 255 - (val - 128) * 4
                else:  # Yellow to bright yellow
                    r, g, b = 255, 255, 0
                
                colored[i, j, 0] = min(r, 255)
                colored[i, j, 1] = min(g, 255)
                colored[i, j, 2] = min(b, 255)
        
        # Create QImage and QPixmap
        q_img = QImage(colored.data, w, h, colored.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale if needed
        if w > self.width() or h > self.height():
            pixmap = pixmap.scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        
        self.setPixmap(pixmap)


class WaveParametersWidget(QWidget):
    """Widget for displaying wave parameters."""
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface."""
        layout = QFormLayout(self)
        
        # Create labels for each parameter
        self.frequency_label = QLabel("--")
        self.orientation_label = QLabel("--")
        self.phase_label = QLabel("--")
        self.weight_label = QLabel("--")
        
        # Add to layout
        layout.addRow("Frequency:", self.frequency_label)
        layout.addRow("Orientation (°):", self.orientation_label)
        layout.addRow("Phase:", self.phase_label)
        layout.addRow("Weight:", self.weight_label)
    
    def update_parameters(self, wave):
        """
        Update the parameters display.
        
        Args:
            wave: Wave parameters dictionary
        """
        if wave is None:
            self.frequency_label.setText("--")
            self.orientation_label.setText("--")
            self.phase_label.setText("--")
            self.weight_label.setText("--")
            return
        
        self.frequency_label.setText(f"{wave.get('frequency', 0):.6f}")
        self.orientation_label.setText(f"{wave.get('orientation', 0):.2f}")
        self.phase_label.setText(f"{wave.get('phase', 0):.6f}")
        self.weight_label.setText(f"{wave.get('weight', 0):.4f}")


class SingleWavePanel(QWidget):
    """Panel for displaying a single wave component."""
    
    def __init__(self, index=0):
        super().__init__()
        self.index = index
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(f"Wave {self.index + 1}")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)
        
        # Wave image
        self.image_widget = WaveImageWidget()
        layout.addWidget(self.image_widget)
        
        # Wave parameters
        self.params_widget = WaveParametersWidget()
        layout.addWidget(self.params_widget)
    
    def update_wave(self, wave, pattern):
        """
        Update the wave display.
        
        Args:
            wave: Wave parameters dictionary
            pattern: Wave pattern array
        """
        self.image_widget.set_image(pattern)
        self.params_widget.update_parameters(wave)


class CWLEViewPanel(QWidget):
    """
    Panel for viewing CWLE patterns and parameters.
    """
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Composite CWLE image
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # CWLE info
        self.info_label = QLabel("No CWLE selected")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(self.info_label)
        
        # CWLE image
        self.composite_image = WaveImageWidget()
        self.composite_image.setMinimumSize(300, 300)
        left_layout.addWidget(self.composite_image)
        
        left_layout.addStretch()
        splitter.addWidget(left_panel)
        
        # Right panel - Individual waves
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Waves title
        waves_title = QLabel("Component Waves")
        waves_title.setAlignment(Qt.AlignCenter)
        waves_title.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(waves_title)
        
        # Scroll area for waves
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area)
        
        splitter.addWidget(right_panel)
        
        # Set initial sizes
        splitter.setSizes([400, 800])
        
        # Create initial wave panels (will be replaced when data is loaded)
        self.wave_panels = []
    
    def update_view(self, cwle, pattern, index, total):
        """
        Update the view with new CWLE data.
        
        Args:
            cwle: Current CWLE parameters
            pattern: Current pattern array
            index: Current index
            total: Total number of CWLEs
        """
        if cwle is None or pattern is None:
            self.info_label.setText("No CWLE selected")
            self.composite_image.clear()
            # Clear wave panels
            self._clear_wave_panels()
            return
        
        # Update info label
        self.info_label.setText(f"CWLE {index + 1} of {total}")
        
        # Update composite image
        self.composite_image.set_image(pattern)
        
        # Clear and recreate wave panels
        self._clear_wave_panels()
        
        # Create image widgets for individual waves
        for i, wave in enumerate(cwle):
            # Create new wave panel
            wave_panel = SingleWavePanel(i)
            
            # Calculate single wave pattern
            img_size = pattern.shape[0]
            x = np.arange(img_size)
            y = np.arange(img_size)
            X, Y = np.meshgrid(x, y)
            
            freq = wave["frequency"]
            theta = wave["orientation"] * (np.pi / 180)
            phase = wave["phase"]
            
            X_rot = X * np.cos(theta) + Y * np.sin(theta)
            wave_values = np.sin(2 * np.pi * freq * X_rot + phase)
            single_pattern = (wave_values + 1) / 2
            
            # Update wave panel
            wave_panel.update_wave(wave, single_pattern)
            
            # Add to layout
            self.scroll_layout.addWidget(wave_panel)
            self.wave_panels.append(wave_panel)
        
        # Add stretch at the end
        self.scroll_layout.addStretch()
    
    def _clear_wave_panels(self):
        """Clear all wave panels."""
        # Remove existing wave panels
        for panel in self.wave_panels:
            self.scroll_layout.removeWidget(panel)
            panel.deleteLater()
        
        # Clear list
        self.wave_panels = []
        
        # Remove any stretches
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()