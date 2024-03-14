from PyQt5.QtCore import Qt
import sys
from testui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
import pyqtgraph as pg
import cv2

# from res_rc import *  # Import the resource module
from PyQt5.uic import loadUiType

ui, _ = loadUiType('testui.ui')


class ImageEditor(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ImageEditor, self).__init__()
        self.setupUi(self)

        # Create an image item for each plotwidget
        (self.img_item_input, self.img_item_output, self.img_item_greyscale) = (pg.ImageItem() for i in range(3))

        self.loaded_image = None
        self.setup_plotwidgets()

        # Maps the radio button to the correspoiding slider page's index
        self.slider_map_noise = {

            self.radio_uniform: 0,
            self.radio_gaus: 1,
            self.radio_sp: 2
        }
        
        self.slider_map_edges = {
            self.radio_sobel: 0,
            self.radio_prewitt: 1,
            self.radio_roberts: 1,
            self.radio_canny: 2,
            
        }


        # Connect radio buttons to function that sets sliders according to selection
        for radio in [self.radio_uniform, self.radio_gaus, self.radio_sp]:
            radio.toggled.connect(self.set_stacked_widget)

        # Connect Openfile Action to its function
        self.actionOpen_Image.triggered.connect(self.open_image)

    # TODO - Rename objects and comment to an appropriate object name
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
        # Loads the image using imread, converts it to RGB, then rotates it 90 degrees clockwise
        self.loaded_image = cv2.rotate(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        greyscale_image = cv2.cvtColor(self.loaded_image, cv2.COLOR_RGB2GRAY)
        self.display_image(self.img_item_input, self.loaded_image)
        self.display_image(self.img_item_greyscale, greyscale_image)

    def display_image(self, image_item, image):
        image_item.setImage(image)
        self.remove_plotwidget_margins()

    ################################ Misc Functions ################################

    def setup_plotwidgets(self):
        for plotwidget in self.findChildren(pg.PlotWidget):
            # Removes Axes and Padding
            plotwidget.showAxis('left', False)
            plotwidget.showAxis('bottom', False)
            plotitem = plotwidget.getPlotItem()
            plotitem.getViewBox().setDefaultPadding(0)

        # Adds the image items to their corresponsing plot widgets so they can be used later to display images
        for plotWidget, imgItem in zip([self.wgt_input_img, self.wgt_input_img_greyscale, self.wgt_output_img],
                                       [self.img_item_input, self.img_item_greyscale, self.img_item_output]):
            plotWidget.addItem(imgItem)

    def remove_plotwidget_margins(self):
        plotitem = self.wgt_input_img.getPlotItem()
        plotitem.getViewBox().setDefaultPadding(0)


app = QApplication(sys.argv)
win = ImageEditor()
win.show()
app.exec()
