import os
from magicgui import magicgui
from PyQt5.QtWidgets import QApplication, QFileDialog

@magicgui(call_button="Select Folder", result_widget=True)
def browse_folder() -> str:
    app = QApplication.instance() or QApplication([])
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
    return folder_path

#if __name__ == "__main__":
#    # Start the Qt application (required for PyQt5)
#    app = QApplication.instance() or QApplication([])
#
#    # Create the GUI
#    gui = browse_folder()
#    print(gui)
#
#    # Run the Qt application event loop
#    app.exec_()
