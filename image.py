class Image():
    def __init__(self, path):

        self._original_image = cv2.resize(cv2.imread(f'{path}'), (320, 170), cv2.INTER_LINEAR)
        self._gray_scale_image = None
        self._image_size = None
        self._image_ifft = None
        self._image_histograms = []
        self._image_distribtion_function = None
    
    def bgr2gray(image):
        """
        Convert an image from BGR to grayscale.

        Parameters:
            image (numpy.ndarray): Input image in BGR format.

        Returns:
            numpy.ndarray: Grayscale image.
        """
        # Extract the B, G, and R channels
        b, g, r = image[:,:,0], image[:,:,1], image[:,:,2]
        
        # Convert to grayscale using the formula: Y = 0.299*R + 0.587*G + 0.114*B
        gray_image = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Convert to uint8 data type
        gray_image = gray_image.astype(np.uint8)
        
        return gray_image
    
    def calculate_histogram_or_distribution(image, plot_type='histogram'):
        # Split the image into its three color channels: B, G, and R
        b, g, r = image[:,:,0], image[:,:,1], image[:,:,2]

        if plot_type == 'histogram':
            # Calculate histograms for each color channel
            hist_b, _ = np.histogram(b.flatten(), bins=256, range=[0, 256])
            hist_g, _ = np.histogram(g.flatten(), bins=256, range=[0, 256])
            hist_r, _ = np.histogram(r.flatten(), bins=256, range=[0, 256])

            return hist_b, hist_g, hist_r
        elif plot_type == 'distribution':
            # Calculate histograms for each color channel
            hist_b, _ = np.histogram(b.flatten(), bins=256, range=[0, 256])
            hist_g, _ = np.histogram(g.flatten(), bins=256, range=[0, 256])
            hist_r, _ = np.histogram(r.flatten(), bins=256, range=[0, 256])

            return hist_b / hist_b.sum(), hist_g / hist_g.sum(), hist_r / hist_r.sum()


    @property
    def original_image(self):
        return self._original_image
    
    @property
    def gray_scale_image(self):
        self._gray_scale_image = self.bgr2gray(self._original_image)
        return self._gray_scale_image
    
    @property
    def image_size(self):
        return self._image_size
    
    @property
    def image_rgb_hsitogram(self):
        self._image_histograms = self.calculate_histogram_or_distribution(self._original_image , 'histogram')
    
    @property
    def image_rgb_distribution_function(self):
        self._image_distribtion_function = self.calculate_histogram_or_distribution(self._original_image , 'distribution')

    @image_size.setter
    def image_size(self, value):
        self._original_image = cv2.resize(self._original_image, value)
        self._image_size = value