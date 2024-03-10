from PyQt5.QtCore import Qt
from testui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget


class ImageEditor(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(self, QMainWindow).__init__()
        self.setupUi()
        
        





app = QApplication(sys.argv)
win = mainWindow() # Change to class name
win.show()
app.exec()