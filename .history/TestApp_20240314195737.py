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

        
        self.plotwidget_set = (self.wgt_input_img, self.wgt_input_img_greyscale, self.wgt_output_img,
                          self.wgt_histo_red, self.wgt_histo_blue, self.wgt_histo_green,
                          self.wgt_histo_red_dist, self.wgt_histo_blue_dist, self.wgt_histo_green_dist,
                          self.wgt_histo_img_colored, self.wgt_histo_img_greyscale, self.wgt_histo_colored,
                          self.wgt_histo_greyscale, self.wgt_hybrid_img_1, self.wgt_hybrid_img_2, self.wgt_hybrid_img_output,
                          self.wgt_hybrid_img_FT_1, self.wgt_hybrid_img_FT_2)

        # Create an image item for each plotwidget
        self.image_item_set = [self.item_filter_input, self.item_filter_greyscale, self.item_filter_output,
         self.item_histo_red, self.item_histo_blue, self.item_histo_green,
         self.item_histo_red_dist, self.item_histo_blue_dist, self.item_histo_green_dist,
         self.item_histo_img_colored, self.item_histo_img_grey, self.item_histo_colored,
         self.item_histo_grey, self.item_hybrid_1, self.item_hybrid_2, self.item_hybrid_out,
         self.item_hybrid_FT_1, self.item_hybrid_FT_2] = [pg.ImageItem() for i in range(18)]

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
        
        self.set_radio_button_connections()


        # Connect Openfile Action to its function
        self.actionOpen_Image.triggered.connect(self.open_image)


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
        for color_plot, grey_plot in zip([self.item_filter_input, self.item_histo_img_colored], [self.item_filter_greyscale, self.item_histo_img_grey]):
            self.display_image(color_plot, self.loaded_image)
            self.display_image(grey_plot, greyscale_image)


    def display_image(self, image_item, image):
        image_item.setImage(image)
        image_item.getViewBox().autoRange()
        
        

    ################################ Misc Functions ################################

    def setup_plotwidgets(self):
        for plotwidget in self.findChildren(pg.PlotWidget):
            if plotwidget.objectName().find("histo") == -1 or plotwidget.objectName().find("histo_img") != -1 or plotwidget.objectName().find("FT") != -1:
                # Removes Axes and Padding from all plotwidgets intended to display an image
                plotwidget.showAxis('left', False)
                plotwidget.showAxis('bottom', False)
                plotitem = plotwidget.getPlotItem()
                plotitem.getViewBox().setDefaultPadding(0)
                
            else:
                plotwidget.setTitle(f"{plotwidget.objectName()[10:]}")
        
        
        # Adds the image items to their corresponsing plot widgets so they can be used later to display images
        for plotwidget, imgItem in zip(self.plotwidget_set, self.image_item_set):
            plotwidget.addItem(imgItem)
            print(f"{imgItem} added to {plotwidget.objectName()} ")
    
    # Sets the page of the stacked widget based on the radio button selected
    def set_stacked_widget(self, stacked_widget, radio_dict):
        """sets up page transitions according to radio button selection

        Args:
            stacked_widget (stackedWidget): the target stackedwidget object
            radio_dict (dictionary): Dictionary linking each radio button with it's page's index
        """
        pressed_radio = self.toolBox.sender()
        if pressed_radio.isChecked():
            if pressed_radio.text() == "None":
                stacked_widget.setVisible(False)
            else:
                stacked_widget.setVisible(True)
                stacked_widget.setCurrentIndex(radio_dict[pressed_radio])

    def set_radio_button_connections(self):        
        # Connect noise radio buttons to function that sets visible sliders according to selection
        # for noise_radio in [self.radio_uniform, self.radio_gaus, self.radio_sp, self.radio_none_noise]:
        for noise_radio in self.slider_map_noise.keys():
            noise_radio.toggled.connect(lambda: self.set_stacked_widget(self.stackedWidget, self.slider_map_noise))
            
        for noise_radio in self.slider_map_edges.keys():
            noise_radio.toggled.connect(lambda: self.set_stacked_widget(self.stackedWidget, self.slider_map_edges))

        


app = QApplication(sys.argv)
win = ImageEditor()
win.show()
app.exec()
