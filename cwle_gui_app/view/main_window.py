import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFrame, QScrollArea, 
    QSplitter, QFileDialog, QMessageBox, QTabWidget
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from view.config_panel import ConfigPanel
from view.cwle_view_panel import CWLEViewPanel

class MainWindow(QMainWindow):
    """
    Main window for the CWLE GUI application.
    """
    # Signals
    generate_cwles_requested = pyqtSignal()
    next_cwle_requested = pyqtSignal()
    prev_cwle_requested = pyqtSignal()
    save_cwles_requested = pyqtSignal(str)
    save_images_requested = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle('CWLE Generator')
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Add menu bar
        self.create_menu_bar()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create config tab
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # Config panel
        self.config_panel = ConfigPanel()
        config_layout.addWidget(self.config_panel)
        
        # Generate button
        generate_button = QPushButton("Generate CWLEs")
        generate_button.setMinimumHeight(50)
        generate_button.clicked.connect(self.on_generate_clicked)
        config_layout.addWidget(generate_button)
        
        self.tab_widget.addTab(config_tab, "Configuration")
        
        # Create view tab
        view_tab = QWidget()
        view_layout = QHBoxLayout(view_tab)
        
        # Previous button
        prev_button = QPushButton("←")
        prev_button.setFixedWidth(50)
        prev_button.setMinimumHeight(400)
        prev_button.clicked.connect(self.on_prev_clicked)
        view_layout.addWidget(prev_button)
        
        # CWLE View panel
        self.cwle_view_panel = CWLEViewPanel()
        view_layout.addWidget(self.cwle_view_panel, 1)
        
        # Next button
        next_button = QPushButton("→")
        next_button.setFixedWidth(50)
        next_button.setMinimumHeight(400)
        next_button.clicked.connect(self.on_next_clicked)
        view_layout.addWidget(next_button)
        
        self.tab_widget.addTab(view_tab, "View CWLEs")
        
        # Status bar
        self.statusBar().showMessage('Ready')
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Generate action
        generate_action = file_menu.addAction('Generate CWLEs')
        generate_action.triggered.connect(self.on_generate_clicked)
        
        file_menu.addSeparator()
        
        # Save CWLEs action
        save_cwles_action = file_menu.addAction('Save CWLEs...')
        save_cwles_action.triggered.connect(self.on_save_cwles_clicked)
        
        # Save Images action
        save_images_action = file_menu.addAction('Save Images...')
        save_images_action.triggered.connect(self.on_save_images_clicked)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # About action
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about_dialog)
    
    def on_generate_clicked(self):
        """Handle Generate button click."""
        # Get config values from the panel
        config = self.config_panel.get_config()
        
        # Update the model through controller (connected signal)
        self.generate_cwles_requested.emit()
        
        # Switch to view tab
        self.tab_widget.setCurrentIndex(1)
    
    def on_next_clicked(self):
        """Handle Next button click."""
        self.next_cwle_requested.emit()
    
    def on_prev_clicked(self):
        """Handle Previous button click."""
        self.prev_cwle_requested.emit()
    
    def on_save_cwles_clicked(self):
        """Handle Save CWLEs action."""
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save CWLEs', '', 'JSON Files (*.json);;All Files (*)'
        )
        if filename:
            self.save_cwles_requested.emit(filename)
            self.statusBar().showMessage(f'CWLEs saved to {filename}')
    
    def on_save_images_clicked(self):
        """Handle Save Images action."""
        directory = QFileDialog.getExistingDirectory(
            self, 'Select Directory to Save Images'
        )
        if directory:
            self.save_images_requested.emit(directory)
            self.statusBar().showMessage(f'Images saved to {directory}')
    
    def update_cwle_view(self, cwle, pattern, index, total):
        """
        Update the CWLE view panel with new data.
        
        Args:
            cwle: Current CWLE parameters
            pattern: Current pattern array
            index: Current index
            total: Total number of CWLEs
        """
        self.cwle_view_panel.update_view(cwle, pattern, index, total)
        self.statusBar().showMessage(f'Showing CWLE {index+1} of {total}')
    
    def show_about_dialog(self):
        """Show the About dialog."""
        QMessageBox.about(
            self,
            'About CWLE Generator',
            'CWLE Generator\n\nA tool for generating and visualizing Composite Wave Label Encodings.'
        )