from PyQt5.QtCore import Qt
import sys
from testui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
import cv2


class ImageEditor(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ImageEditor, self).__init__()
        self.setupUi(self)
        
        self.loaded_image = None
        
            
        # Maps the radio button to the correspoiding slider page's index
        self.slider_map = {
            
            self.radio_uniform: 0,
            self.radio_gaus: 1,
            self.radio_sp: 2
        }
        
        # Connect radio buttons to function that sets sliders according to selection
        for radio in [self.radio_uniform, self.radio_gaus, self.radio_sp]:
            radio.toggled.connect(self.set_stacked_widget)
        
        
        
    #TODO - Rename objects and comment to an appropriate object name
    # Sets the page of the stacked widget based on the radio button selected
    def set_stacked_widget(self):
        pressed_radio = self.toolBox.sender()
        if pressed_radio.isChecked():
            self.stackedWidget.setCurrentIndex(self.slider_map[pressed_radio])    
            
    def open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif)")
        file_dialog.setWindowTitle("Open Image File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            self.load_img_file(selected_file)
            
    def load_img_file(self, image_path):
        self.loaded_image = cv2.imread(image_path)
        





app = QApplication(sys.argv)
win = ImageEditor() 
win.show()
app.exec()