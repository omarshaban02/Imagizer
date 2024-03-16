import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
import pyqtgraph as pg
import cv2
from classes import Filter
from PyQt5.uic import loadUiType
import numpy as np

ui, _ = loadUiType('testui.ui')


class ImageEditor(QMainWindow, ui):
    def __init__(self):
        super(ImageEditor, self).__init__()
        self.setupUi(self)
        self.filter = None
        self.noisy_img = None
        self.loaded_image = None
        self.loaded_image_second = None

        self.plotwidget_set = (self.wgt_input_img, self.wgt_input_img_greyscale, self.wgt_output_img,
                               self.wgt_histo_red, self.wgt_histo_blue, self.wgt_histo_green,
                               self.wgt_histo_red_dist, self.wgt_histo_blue_dist, self.wgt_histo_green_dist,
                               self.wgt_histo_img_colored, self.wgt_histo_img_greyscale, self.wgt_histo_colored,
                               self.wgt_histo_greyscale, self.wgt_hybrid_img_1, self.wgt_hybrid_img_2,
                               self.wgt_hybrid_img_output,
                               self.wgt_hybrid_img_FT_1, self.wgt_hybrid_img_FT_2)

        # Create an image item for each plot-widget
        self.image_item_set = [self.item_filter_input, self.item_filter_greyscale, self.item_filter_output,
                               self.item_histo_red, self.item_histo_blue, self.item_histo_green,
                               self.item_histo_red_dist, self.item_histo_blue_dist, self.item_histo_green_dist,
                               self.item_histo_img_colored, self.item_histo_img_grey, self.item_histo_colored,
                               self.item_histo_grey, self.item_hybrid_1, self.item_hybrid_2, self.item_hybrid_out,
                               self.item_hybrid_FT_1, self.item_hybrid_FT_2] = [pg.ImageItem() for _ in range(18)]

        self.setup_plotwidgets()

        # Maps the radio button to the corresponding slider page's index
        self.radio_dict_noise = {

            self.radio_uniform: 0,
            self.radio_gaus: 1,
            self.radio_sp: 2,
            self.radio_none_noise: 0
        }

        self.radio_dict_edges = {
            self.radio_sobel: 0,
            self.radio_prewitt: 1,
            self.radio_roberts: 1,
            self.radio_canny: 2,
            self.radio_none_edges: 0,
            self.radio_laplace: 3
        }

        # connect buttons
        self.btn_browse_1.clicked.connect(self.open_image)
        self.kernel_size_slider.valueChanged.connect(self.output_img_display)
        self.radio_kernal_one.toggled.connect(self.output_img_display)
        self.radio_kernal_two.toggled.connect(self.output_img_display)
        self.sigma_slider.valueChanged.connect(self.output_img_display)
        self.low_threshold_slider.valueChanged.connect(self.output_img_display)
        self.high_threshold_slider.valueChanged.connect(self.output_img_display)

        self.set_radio_button_connections()  # Sets up handling Ui changes according to radio button selection

        self.hide_stuff_at_startup()

        # Connect Openfile Action to its function
        self.actionOpen_Image.triggered.connect(self.open_image)

    def output_img_display(self):
        """
        Display the output image based on the selected filter.

        This method checks which radio button is checked and applies the corresponding filter to the loaded image.
        The filtered image is then displayed in the output image widget.

        If no image is loaded, a warning message is displayed.
        """
        if self.loaded_image is not None:
            # Average filter
            if self.radio_average.isChecked():
                if (np.any(self.filter.current_img != self.noisy_img) or
                        self.filter.current_ksize != self.kernel_size_slider.value()):
                    self.filter.average(self.noisy_img, self.kernel_size_slider.value())
                self.display_image(self.item_filter_output, self.filter.img_average)

            # Median filter
            elif self.radio_median.isChecked():
                if (np.any(self.filter.current_img != self.noisy_img) or
                        self.filter.current_ksize != self.kernel_size_slider.value()):
                    self.filter.median_scipy(self.noisy_img, self.kernel_size_slider.value())
                self.display_image(self.item_filter_output, self.filter.img_median)

            # Gaussian smoothing
            elif self.radio_gauss_smooth.isChecked():
                kernel_number_checked = 1 if self.radio_kernal_one.isChecked() else 2
                if np.any(self.filter.current_img != self.noisy_img) or (
                        self.filter.gaussian_kernel_number != kernel_number_checked):
                    print(kernel_number_checked)
                    self.filter.gaussian(self.noisy_img, kernel_number_checked)
                self.display_image(self.item_filter_output, self.filter.img_gaussian)

            # Edge Detection
            # Sobel filter
            elif self.radio_sobel.isChecked():
                if np.any(self.filter.current_img != self.noisy_img):
                    self.filter.sobel(self.noisy_img)
                self.display_image(self.item_filter_output, self.filter.img_sobel)

            # Roberts filter
            elif self.radio_roberts.isChecked():
                if np.any(self.filter.current_img != self.noisy_img):
                    self.filter.roberts(self.noisy_img)
                self.display_image(self.item_filter_output, self.filter.img_roberts)

            # Prewitt filter
            elif self.radio_prewitt.isChecked():
                if np.any(self.filter.current_img != self.noisy_img):
                    self.filter.prewitt(self.noisy_img)
                self.display_image(self.item_filter_output, self.filter.img_prewitt)

            # Canny edge detection
            elif self.radio_canny.isChecked():
                sigma = self.sigma_slider.value()
                low_threshold = self.low_threshold_slider.value()
                high_threshold = self.high_threshold_slider.value()
                kernel_size = self.kernel_size_slider.value()

                if (not np.any(self.filter.current_img != self.noisy_img) and
                        sigma == self.filter.canny_sigma and
                        low_threshold == self.filter.canny_low_threshold and
                        high_threshold == self.filter.canny_high_threshold and
                        kernel_size == self.filter.canny_kernel_size):
                    pass
                else:
                    self.filter.canny(self.noisy_img, sigma, low_threshold, high_threshold, kernel_size)
                self.display_image(self.item_filter_output, self.filter.img_canny)

            # Laplace filter
            elif self.radio_laplace.isChecked():
                kernel_number_checked = 1 if self.radio_kernal_one.isChecked() else 2
                if np.any(self.filter.current_img != self.noisy_img) or (
                        self.filter.laplace_kernel_number != kernel_number_checked):
                    self.filter.laplace(self.noisy_img, kernel_number_checked)
                self.display_image(self.item_filter_output, self.filter.img_laplace)

            # No filter selected, display the original image
            else:
                self.display_image(self.item_filter_output, self.noisy_img)
        else:
            # No image loaded, display a warning message
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first")

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
        grayscale_image = cv2.cvtColor(self.loaded_image, cv2.COLOR_RGB2GRAY)
        self.filter = Filter(grayscale_image)
        self.noisy_img = grayscale_image
        self.noisy_img = grayscale_image
        self.output_img_display()
        for color_plot, grey_plot in zip([self.item_filter_input, self.item_histo_img_colored],
                                         [self.item_filter_greyscale, self.item_histo_img_grey]):
            self.display_image(color_plot, self.loaded_image)
            self.display_image(grey_plot, grayscale_image)

    @staticmethod
    def display_image(image_item, image):
        image_item.setImage(image)
        image_item.getViewBox().autoRange()

    # ############################### Misc Functions ################################

    def setup_plotwidgets(self):
        for plotwidget in self.findChildren(pg.PlotWidget):
            if plotwidget.objectName().find("histo") == -1 or plotwidget.objectName().find(
                    "histo_img") != -1 or plotwidget.objectName().find("FT") != -1:
                # Removes Axes and Padding from all plotwidgets intended to display an image
                plotwidget.showAxis('left', False)
                plotwidget.showAxis('bottom', False)
                plotwidget.setBackground((25, 30, 40))
                plotitem = plotwidget.getPlotItem()
                plotitem.getViewBox().setDefaultPadding(0)

            else:
                plotwidget.setTitle(f"{plotwidget.objectName()[10:]}")
                plotwidget.setBackground((25, 35, 45))
                plotwidget.showGrid(x=True, y=True)

        # Adds the image items to their corresponding plot widgets, so they can be used later to display images
        for plotwidget, imgItem in zip(self.plotwidget_set, self.image_item_set):
            plotwidget.addItem(imgItem)
            print(f"{imgItem} added to {plotwidget.objectName()} ")

    # Sets the page of the stacked widget based on the radio button selected
    def set_stacked_widget(self, stacked_widget, radio_dict):
        """sets up page transitions according to radio button selection

        Args:
            stacked_widget (stackedWidget): the target stackedwidget object
            radio_dict (dictionary): Dictionary linking each radio button with its page's index
        """
        pressed_radio = self.toolBox.sender()
        if pressed_radio.isChecked():
            if pressed_radio.objectName().find("none") != -1:
                stacked_widget.setVisible(False)
                print("None pressed...")
            else:
                stacked_widget.setVisible(True)
                stacked_widget.setCurrentIndex(radio_dict[pressed_radio])

    def show_smoothing_options(self):
        if self.sender().isChecked():
            # Show the Kernel type option for Gaussian And Laplacian smoothing only
            if self.sender().text() == "Gaussian" or self.sender().text() == "Laplacian":
                self.wgt_ktype.setVisible(True)
                self.wgt_smooth_kernel.setVisible(False)

            elif self.sender().text() == "None":
                self.wgt_ktype.setVisible(False)
                self.wgt_smooth_kernel.setVisible(False)

            else:
                self.wgt_ktype.setVisible(False)
                self.wgt_smooth_kernel.setVisible(True)

    def set_radio_button_connections(self):

        # Connect noise radio buttons to function that sets visible sliders according to selection
        for noise_radio in self.radio_dict_noise.keys():
            noise_radio.toggled.connect(lambda: self.set_stacked_widget(self.stackedWidget, self.radio_dict_noise))

        # Connect edges radio buttons to function that sets visible sliders according to selection
        for edge_radio in self.radio_dict_edges.keys():
            edge_radio.toggled.connect(lambda: self.set_stacked_widget(self.stackedWidget_edges, self.radio_dict_edges))
            edge_radio.toggled.connect(self.output_img_display)

        for smooth_radio in self.buttonGroup_smoothing.buttons():
            smooth_radio.toggled.connect(self.show_smoothing_options)
            smooth_radio.toggled.connect(self.output_img_display)

    def hide_stuff_at_startup(self):
        for widget in [self.stackedWidget, self.stackedWidget_edges, self.wgt_smooth_kernel, self.wgt_ktype]:
            widget.setVisible(False)


app = QApplication(sys.argv)
win = ImageEditor()
win.show()
app.exec()
