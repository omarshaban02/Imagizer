import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from mainui import Ui_MainWindow
import pyqtgraph as pg
import cv2
from classes import Filter, FourierTransform, Image, add_gaussian_noise, add_salt_and_pepper_noise, add_uniform_noise
from PyQt5.uic import loadUiType
import numpy as np

ui, _ = loadUiType('main.ui')


class ImageEditor(QMainWindow, ui):
    def __init__(self):
        super(ImageEditor, self).__init__()
        self.setupUi(self)
        self.filter = None
        self.noisy_img = None
        self.loaded_image = None
        self.loaded_image_second = None
        # ######################################## Hybrid image # ##############################################
        self.fft_operation = FourierTransform()
        self.low_pass_img = None
        self.high_pass_img = None
        self.hybrid_images_list = []
        self.images_fft_list = []

        self.img_obj = None
        # #####################################################################################################

        self.plotwidget_set = [self.wgt_input_img, self.wgt_input_img_greyscale, self.wgt_output_img,
                               self.wgt_histo_red_dist, self.wgt_histo_blue_dist, self.wgt_histo_green_dist,
                               self.wgt_histo_img_colored, self.wgt_histo_img_greyscale,
                               self.wgt_histo_curve, self.wgt_hybrid_img_1, self.wgt_hybrid_img_2,
                               self.wgt_hybrid_img_output,
                               self.wgt_hybrid_img_FT_1, self.wgt_hybrid_img_FT_2,  # ]
                               self.wgt_histo_red, self.wgt_histo_blue, self.wgt_histo_green,
                               self.wgt_histo_histogram]

        # Create an image item for each plot-widget
        self.image_item_set = [self.item_filter_input, self.item_filter_greyscale, self.item_filter_output,
                               self.item_histo_red_dist, self.item_histo_blue_dist, self.item_histo_green_dist,
                               self.item_histo_img_colored, self.item_histo_img_grey,
                               self.item_histo_curve, self.item_hybrid_1, self.item_hybrid_2, self.item_hybrid_out,
                               self.item_hybrid_FT_1, self.item_hybrid_FT_2,
                               # self.item_histo_red, self.item_histo_blue, self.item_histo_green,
                               # self.item_histo_histogram,
                               ] = [pg.ImageItem() for _ in range(14)]  # + [pg.HistogramLUTItem for _ in range(4)]

        self.hist_widget_set = [self.wgt_histo_red, self.wgt_histo_blue, self.wgt_histo_green,
                                self.wgt_histo_histogram]
        self.hist_item_set = [
            self.item_histo_red, self.item_histo_blue, self.item_histo_green, self.item_histo_histogram
        ] = [pg.BarGraphItem(x=[0 for _ in range(10)], height=[0 for _ in range(10)], width=0.5, brush='w') for _ in
             range(4)]

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
        self.kernel_size_slider.valueChanged.connect(self.output_img_display)
        self.radio_kernal_one.toggled.connect(self.output_img_display)
        self.radio_kernal_two.toggled.connect(self.output_img_display)
        self.sigma_slider.valueChanged.connect(self.output_img_display)
        self.low_threshold_slider.valueChanged.connect(self.output_img_display)
        self.high_threshold_slider.valueChanged.connect(self.output_img_display)
        # ######################################## Noise sliders # ##############################################
        
        # Uniform
        self.slider_intensity.valueChanged.connect(self.apply_noise)
        
        # Gaussian
        self.slider_mean.valueChanged.connect(self.apply_noise)
        self.slider_std.valueChanged.connect(self.apply_noise)
        
        # Salt and Pepper
        self.slider_salt.valueChanged.connect(self.apply_noise)
        self.slider_pepper.valueChanged.connect(self.apply_noise)
        
        # #####################################################################################################
        
        # ######################################## Hybrid image # ##############################################
        self.btn_browse_1.clicked.connect(lambda: self.open_hybrid_img(self.item_hybrid_1,
                                                                       not self.edge_donor2_radioButton.isChecked(), 0))
        self.btn_browse_2.clicked.connect(lambda: self.open_hybrid_img(self.item_hybrid_2,
                                                                       self.edge_donor2_radioButton.isChecked(), 1))
        self.edge_donor1_radioButton.toggled.connect(self.recalc_high_low_pass_img)
        self.hybrid_threshold_slider.valueChanged.connect(self.recalc_high_low_pass_img)
        # #####################################################################################################
       
        # ######################################## Threshold # ##################################################
        self.radio_global.toggled.connect(self.global_thresholding)
        self.radio_local.toggled.connect(self.local_thresholding)
        self.slider_thresholding.valueChanged.connect(self.slider_value_changed)
        self.slider_thresholding.setRange(1, 255)
        # #####################################################################################################
        self.set_radio_button_connections()  # Sets up handling Ui changes according to radio button selection

        self.hide_stuff_at_startup()

        # Connect Openfile Action to its function
        self.actionOpen_Image.triggered.connect(self.open_image)
        # -------------------------------------- Histogram --------------------------------------------
        self.equalize_check_box.clicked.connect(self.equalize_img)
        self.normalize_check_box.clicked.connect(self.equalize_img)

    def equalize_img(self):
        if self.equalize_check_box.isChecked():
            self.normalize_check_box.setEnabled(True)
            self.display_image(self.item_histo_img_colored, self.img_obj.equalized_img)
            self.display_image(self.item_histo_img_grey, self.img_obj.calculate_gray_scale_image(
                self.img_obj.equalized_img))
            if self.normalize_check_box.isChecked():
                self.display_hist(self.wgt_histo_histogram, self.item_histo_histogram, self.img_obj.normalized_hist)
                self.plot_histogram_and_distribution_pyqtgraph(self.wgt_histo_curve, self.img_obj.normalized_hist,
                                                               "white", "distribution")
            else:
                self.display_hist(self.wgt_histo_histogram, self.item_histo_histogram, self.img_obj.equalized_hist)
                self.plot_histogram_and_distribution_pyqtgraph(self.wgt_histo_curve, self.img_obj.equalized_hist,
                                                               "white", "distribution")
        elif not self.equalize_check_box.isChecked():
            self.display_image(self.item_histo_img_colored, self.img_obj.original_img)
            self.display_image(self.item_histo_img_grey, self.img_obj.gray_scale_image)
            self.display_hist(self.wgt_histo_histogram, self.item_histo_histogram, self.img_obj.img_histogram)
            self.plot_histogram_and_distribution_pyqtgraph(self.wgt_histo_curve, self.img_obj.img_histogram,
                                                           "white", "distribution")
            self.normalize_check_box.setEnabled(False)

    @staticmethod
    def display_hist(plot_widget, item, hist, brush="w"):
        plot_widget.clear()

        item.setOpts(x=range(256), height=hist, brush=brush, pen=brush)
        plot_widget.addItem(item)

        # Set labels and title
        plot_widget.setLabel('left', 'Frequency')
        plot_widget.setLabel('bottom', 'Intensity')
        plot_widget.setTitle('Main Histogram')

        # Update the plot
        plot_widget.repaint()

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

    # ######################################## Hybrid image # ##############################################
    def open_hybrid_img(self, plot_widget, state, img_num):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif *.jpeg)")
        file_dialog.setWindowTitle("Open Image File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            self.loaded_image = cv2.rotate(cv2.cvtColor(cv2.imread(selected_file), cv2.COLOR_BGR2RGB),
                                           cv2.ROTATE_90_CLOCKWISE)
            greyscale_image = cv2.cvtColor(self.loaded_image, cv2.COLOR_RGB2GRAY)
            greyscale_image = cv2.resize(greyscale_image, (361, 410))
            self.display_image(plot_widget, greyscale_image)
            image_fft = self.fft_operation.get_img_fft(greyscale_image)
            self.images_fft_list.insert(img_num, image_fft)
            if state:
                self.calc_high_pass_img(image_fft)
            else:
                self.calc_low_pass_img(image_fft)
            if len(self.hybrid_images_list) >= 2:
                self.calc_hybrid_image()

    def calc_low_pass_img(self, image_fft):
        self.low_pass_img = self.fft_operation.low_pass_filter(image_fft, self.hybrid_threshold_slider.value(), 1)
        self.low_pass_img = self.fft_operation.get_inverse_fft(self.low_pass_img)
        self.hybrid_images_list.append(self.low_pass_img)
        self.low_pass_img = np.real(self.low_pass_img)
        self.low_pass_img = np.uint8(self.low_pass_img)
        self.display_image(self.item_hybrid_FT_1, self.low_pass_img)

    def calc_high_pass_img(self, image_fft):
        self.high_pass_img = self.fft_operation.high_pass_filter(image_fft, self.hybrid_threshold_slider.value(), 1)
        self.high_pass_img = self.fft_operation.get_inverse_fft(self.high_pass_img)
        self.hybrid_images_list.append(self.high_pass_img)
        self.high_pass_img = np.real(self.high_pass_img)
        self.high_pass_img = np.uint8(self.high_pass_img)
        self.display_image(self.item_hybrid_FT_2, self.high_pass_img)

    def recalc_high_low_pass_img(self):
        if len(self.images_fft_list) >= 2:
            if self.edge_donor1_radioButton.isChecked():
                self.calc_high_pass_img(self.images_fft_list[0])
                self.calc_low_pass_img(self.images_fft_list[1])
            else:
                self.calc_high_pass_img(self.images_fft_list[1])
                self.calc_low_pass_img(self.images_fft_list[0])
            self.calc_hybrid_image()

    def calc_hybrid_image(self):
        hybrid_image = self.filter.hybrid_images(self.hybrid_images_list[-1], self.hybrid_images_list[-2])
        self.display_image(self.item_hybrid_out, hybrid_image)

    # #####################################################################################################
    # ######################################## Threshold  # #################################################
    def global_radio_toggled(self):
        """
        Callback function for the global thresholding radio button toggled signal.
        """
        if self.radio_global.isChecked():
            self.slider_thresholding.setValue(1)  # Reset the slider to default value
            self.slider_thresholding.valueChanged.disconnect(self.local_thresholding)
            self.slider_thresholding.valueChanged.connect(self.global_thresholding)

    def local_radio_toggled(self):
        """
        Callback function for the local thresholding radio button toggled signal.
        """
        if self.radio_local.isChecked():
            self.slider_thresholding.setValue(1)  # Reset the slider to default value
            self.slider_thresholding.valueChanged.disconnect(self.global_thresholding)
            self.slider_thresholding.valueChanged.connect(self.local_thresholding)

    def slider_value_changed(self):
        """
        Callback function for the slider value changed signal.
        """
        if self.radio_global.isChecked():
            self.global_thresholding()
        elif self.radio_local.isChecked():
            self.local_thresholding()

    def global_thresholding(self):
        """
        Callback function for global thresholding.
        """
        result = self.filter.global_threshold(self.filter.current_img, self.slider_thresholding.value())
        self.display_image(self.item_filter_output, result)

    def local_thresholding(self):
        """
        Callback function for local thresholding.
        """
        result = self.filter.local_threshold(self.filter.current_img, round(self.slider_thresholding.value() // 10))
        self.display_image(self.item_filter_output, result)

    @staticmethod
    def plot_histogram_and_distribution_pyqtgraph(plot_widget, data, color_channel, plot_type):
        # Calculate distribution data if necessary
        if plot_type == 'distribution':
            data = np.cumsum(data)

        # Clear any existing plots
        plot_widget.clear()

        # Plot data
        plot_widget.plot(data, pen=color_channel)

        # Set labels and title
        plot_widget.setLabel('left', 'Frequency')
        plot_widget.setLabel('bottom', 'Intensity')
        if plot_type == 'histogram':
            plot_widget.setTitle(f'{color_channel.capitalize()} Channel Histogram')
        elif plot_type == 'distribution':
            plot_widget.setTitle(f'{color_channel.capitalize()} Channel Distribution')

        # Update the plot
        plot_widget.repaint()

    # #####################################################################################################

    def open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif *.jpeg)")
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
        self.img_obj = Image(self.loaded_image)
        self.noisy_img = grayscale_image
        self.noisy_img = grayscale_image
        self.output_img_display()
        for color_plot, grey_plot in zip([self.item_filter_input, self.item_histo_img_colored],
                                         [self.item_filter_greyscale, self.item_histo_img_grey]):
            self.display_image(color_plot, self.loaded_image)
            self.display_image(grey_plot, grayscale_image)
        hist_b, hist_g, hist_r, hist_gray = self.img_obj.bgr_img_histograms
        dist_widgets = [
            # (self.wgt_histo_green, hist_b, 'green', 'histogram'),
            # (self.wgt_histo_blue, hist_g, 'blue', 'histogram'),
            # (self.wgt_histo_red, hist_r, 'red', 'histogram'),
            # (self.wgt_histo_img_greyscale, hist_gray, 'gray', 'histogram'),
            (self.wgt_histo_green_dist, hist_b, 'green', 'distribution'),
            (self.wgt_histo_blue_dist, hist_g, 'blue', 'distribution'),
            (self.wgt_histo_red_dist, hist_r, 'red', 'distribution'),
            # (self.wgt_histo_histogram, self.img_obj.img_histogram, "white", "histogram"),
            (self.wgt_histo_curve, self.img_obj.img_histogram, "white", "distribution")
        ]

        histogram_widgets = [
            (self.wgt_histo_green, self.item_histo_green, hist_g, "g"),
            (self.wgt_histo_blue, self.item_histo_blue, hist_b, "b"),
            (self.wgt_histo_red, self.item_histo_red, hist_r, "r"),
            (self.wgt_histo_histogram, self.item_histo_histogram, self.img_obj.img_histogram, "w")
        ]

        # Iterate over the list of tuples and call the plot_histogram_and_distribution_pyqtgraph function
        for widget, histogram_data, color, plot_type in dist_widgets:
            self.plot_histogram_and_distribution_pyqtgraph(widget, histogram_data, color, plot_type)

        for widget, item, histogram_data, brush in histogram_widgets:
            self.display_hist(widget, item, histogram_data, brush)

    @staticmethod
    def display_image(image_item, image):
        image_item.setImage(image)
        image_item.getViewBox().autoRange()
        
    def apply_noise(self):
        if self.noisy_img != None:
            if self.radio_uniform.isChecked():
                self.noisy_img = add_uniform_noise(self.noisy_img, self.slider_intensity.value())
            elif self.radio_gaus.isChecked():
                self.noisy_img = add_gaussian_noise(self.noisy_img, self.slider_mean.value(), self.slider_std.value())
            elif self.radio_sp.isChecked():
                self.noisy_img = add_salt_and_pepper_noise(self.noisy_img, self.slider_salt.value(), self.slider_pepper.value())
        
        
        self.display_image(self.item_filter_output, self.noisy_img)
        
    

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

        for hist_wgt, hist_item in zip(self.hist_widget_set, self.hist_item_set):
            hist_wgt.addItem(hist_item)

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
