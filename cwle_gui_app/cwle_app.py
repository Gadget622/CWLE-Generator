import sys
from PyQt5.QtWidgets import QApplication
from model.cwle_model import CWLEModel
from view.main_window import MainWindow
from controller.cwle_controller import CWLEController

def main():
    """
    Main entry point for the CWLE GUI application.
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a more modern look
    
    # Create MVC components
    model = CWLEModel()
    view = MainWindow()
    controller = CWLEController(model, view)
    
    # Show the main window
    view.show()
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()