from PyQt5.QtCore import Qt
import sys
from testui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget


class ImageEditor(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setupUi()
        
            
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





app = QApplication(sys.argv)
win = ImageEditor() 
win.show()
app.exec()