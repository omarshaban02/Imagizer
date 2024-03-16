import numpy as np
import cv2
from scipy.signal import convolve2d
import threading
from scipy.ndimage import median_filter


def matrix_padding(img, size):
    pad_size = size // 2
    result = np.zeros((img.shape[0] + 2 * pad_size, img.shape[1] + 2 * pad_size))
    result[pad_size:img.shape[0] + pad_size, pad_size:img.shape[1] + pad_size] = img
    return result


class Filter:
    """
    A class used to apply various filters to an image.

    Attributes:
        current_img (numpy.ndarray): The current image to be processed.
        img_average (numpy.ndarray): The image after applying the average filter.
        img_median (numpy.ndarray): The image after applying the median filter.
        img_gaussian (numpy.ndarray): The image after applying the Gaussian filter.
        img_roberts (numpy.ndarray): The image after applying the Roberts operator.
        img_prewitt (numpy.ndarray): The image after applying the Prewitt operator.
        img_canny (numpy.ndarray): The image after applying the Canny edge detection.
        img_sobel (numpy.ndarray): The image after applying the Sobel operator.
        img_laplace (numpy.ndarray): The image after applying the Laplace filter.
        current_ksize (int): The current kernel size.
        gaussian_kernel_number (int): The number of the Gaussian kernel to be used.
        laplace_kernel_number (int): The number of the Laplace kernel to be used.
        canny_sigma (float): The standard deviation for the Gaussian kernel in Canny edge detection.
        canny_low_threshold (int): The low threshold for hysteresis in Canny edge detection.
        canny_high_threshold (int): The high threshold for hysteresis in Canny edge detection.
        fft_operation (FourierTransform): An instance of the FourierTransform class.
    """

    def __init__(self, img):
        """
        The constructor for the Filter class.

        Parameters:
            img (numpy.ndarray): The image to be processed.
        """
        self.current_img = img
        self.img_average = None
        self.img_median = None
        self.img_gaussian = None
        self.img_roberts = None
        self.img_prewitt = None
        self.img_canny = None
        self.img_sobel = None
        self.img_laplace = None
        self.current_ksize = 3
        self.gaussian_kernel_number = 1
        self.laplace_kernel_number = 1
        self.canny_sigma = 1.0
        self.canny_low_threshold = 30
        self.canny_high_threshold = 60

        self.fft_operation = FourierTransform()

        calculations_thread = threading.Thread(target=self.calc_filters, args=(img,))
        calculations_thread.start()

    def calc_filters(self, img):
        """
        Apply all filters to the image.

        Parameters:
            img (numpy.ndarray): The image to be processed.
        """
        self.average(img, 3)
        self.median(img, 3)
        self.gaussian(img, 3)
        self.roberts(img)
        self.prewitt(img)
        self.canny(img)
        self.sobel(img)
        self.laplace(img)

    # Smoothing filters
    def average(self, img, ksize):
        """
        Apply average filter to an image.

        The average filter is a simple sliding window spatial filter that replaces the center value in the window
        with the average (mean) of all the pixel values in the window.

        Parameters:
            img (numpy.ndarray): Input image.
            ksize (int): Kernel size.

        Returns:
            result (numpy.ndarray): The image after applying the average filter.
        """
        kernel = np.ones((ksize, ksize)) / (ksize * ksize)
        self.current_ksize = ksize
        result = (convolve2d(matrix_padding(img, ksize), kernel, mode='same', boundary='symm')).astype(np.uint8)
        self.img_average = result
        return result

    def median(self, img, ksize):
        """
        Apply median filter to an image.

        The median filter is a non-linear digital filtering technique, often used to remove noise from an image or
        signal.

        Parameters:
            img (numpy.ndarray): Input image.
            ksize (int): Kernel size.

        Returns:
            result (numpy.ndarray): The image after applying the median filter.
        """
        result = np.zeros_like(img)
        img_padded = matrix_padding(img, ksize)

        half_size = ksize // 2

        for i in range(half_size, img_padded.shape[0] - half_size):
            for j in range(half_size, img_padded.shape[1] - half_size):
                window = img_padded[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1].flatten()
                result[i - half_size, j - half_size] = np.median(window)

        self.current_ksize = ksize
        self.img_median = result.astype(np.uint8)
        return result.astype(np.uint8)

    def median_scipy(self, img, ksize):
        """
        Apply median filter to an image using SciPy.

        The median filter is a non-linear digital filtering technique, often used to remove noise from an image or
        signal.

        Parameters:
            img (numpy.ndarray): Input image.
            ksize (int): Kernel size.

        Returns:
            result (numpy.ndarray): The image after applying the median filter.
        """
        result = median_filter(img, size=ksize)

        self.current_ksize = ksize
        self.img_median = result.astype(np.uint8)
        return result.astype(np.uint8)

    def gaussian(self, img, kernel_number=1):
        """
        Apply Gaussian filter to an image.

        The Gaussian filter is a type of convolution filter that is used to 'blur' the image or reduce detail and noise.

        Parameters:
            img (numpy.ndarray): Input image.
            kernel_number (int): Determines the type of Gaussian kernel to be used.

        Returns:
            result (numpy.ndarray): The image after applying the Gaussian filter.
        """
        if kernel_number == 1:
            kernel = (1 / 12) * np.array([[1, 1, 1],
                                          [1, 4, 1],
                                          [1, 1, 1]])
        else:
            kernel = (1 / 126) * np.array([[1, 1, 1, 1, 1],
                                           [1, 5, 10, 5, 1],
                                           [1, 10, 50, 10, 1],
                                           [1, 5, 10, 5, 1],
                                           [1, 1, 1, 1, 1]])

        result = convolve2d(img, kernel, mode='same', boundary='symm').astype(np.uint8)
        self.img_gaussian = result
        self.gaussian_kernel_number = kernel_number
        return result

    def roberts(self, img):
        """
        Apply Roberts operator to an image.

        The Roberts Cross operator performs a simple, quick to compute, 2-D spatial gradient measurement on an image.

        Parameters:
            img (numpy.ndarray): Input image.

        Returns:
            result (numpy.ndarray): The image after applying the Roberts operator.
        """
        k1 = np.array([[1, 0], [0, -1]])
        k2 = np.array([[0, 1], [-1, 0]])

        grad1 = convolve2d(img, k1, mode='same', boundary='symm')
        grad2 = convolve2d(img, k2, mode='same', boundary='symm')

        result = (np.abs(grad1) + np.abs(grad2)).astype(np.uint8)
        self.img_roberts = result
        return result

    def prewitt(self, img):
        """
        Apply Prewitt operator to an image.

        The Prewitt operator is used in image processing, particularly within edge detection algorithms.

        Parameters:
            img (numpy.ndarray): Input image.

        Returns:
            result (numpy.ndarray): The image after applying the Prewitt operator.
        """
        kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])

        kernel_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])

        grad_x = convolve2d(img, kernel_x, mode='same', boundary='symm')
        grad_y = convolve2d(img, kernel_y, mode='same', boundary='symm')

        result = (np.sqrt(grad_x ** 2 + grad_y ** 2)).astype(np.uint8)
        self.img_prewitt = result
        return result

    def sobel(self, img):
        """
        Apply Sobel operator to an image.

        The Sobel operator is used in image processing and computer vision, particularly within edge detection
        algorithms.

        Parameters:
            img (numpy.ndarray): Input image.

        Returns:
            result (numpy.ndarray): The image after applying the Sobel operator.
        """
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = np.abs(convolve2d(img, kernel_x, mode='same', boundary='symm'))
        grad_y = np.abs(convolve2d(img, kernel_y, mode='same', boundary='symm'))

        result = (grad_x + grad_y).astype(np.uint8)
        self.img_sobel = result
        return result

    def canny(self, img, sigma=1.0, low_threshold=30, high_threshold=60, ksize=3):
        """
        Apply Canny edge detection to an image.

        The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide
        range of edges in images.

        Parameters:
            img (numpy.ndarray): Input image.
            sigma (float): Standard deviation for Gaussian kernel.
            low_threshold (int): Low threshold for hysteresis.
            high_threshold (int): High threshold for hysteresis.
            ksize (int): Kernel size.

        Returns:
            edges (numpy.ndarray): The image after applying the Canny edge detection.
        """
        blurred_img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        edges = cv2.Canny(blurred_img, low_threshold, high_threshold)
        self.img_canny = edges.astype(np.uint8)
        return edges.astype(np.uint8)

    def laplace(self, img, kernel_number=1):
        """
        Apply Laplace filter to an image.

        The Laplace filter is used for edge detection. It calculates the second derivative of the image,
        where the edges are the zero crossings. This function provides two types of Laplace kernels.

        Parameters:
            img (numpy.ndarray): Input image.
            kernel_number (int): Determines the type of Laplace kernel to be used.
                                 If 1, uses a 3x3 kernel with -8 at the center and rest 1.
                                 Otherwise, uses a 3x3 kernel with -4 at the center and rest 1.

        Returns:
            laplace_result (numpy.ndarray): The image after applying the Laplace filter.
        """

        # Choose the Laplace kernel based on the kernel_number
        if kernel_number == 1:
            laplace_kernel = np.array([[1, 1, 1],
                                       [1, -8, 1],
                                       [1, 1, 1]])
        else:
            laplace_kernel = np.array([[0, 1, 0],
                                       [1, -4, 1],
                                       [0, 1, 0]])

        # Apply the Laplace kernel to the image
        laplace_result = convolve2d(img, laplace_kernel, mode='same', boundary='symm')

        # Convert the result to uint8
        laplace_result = laplace_result.astype(np.uint8)

        # Store the result and the kernel number in the instance
        self.img_laplace = laplace_result
        self.laplace_kernel_number = kernel_number

        return laplace_result

    def global_threshold(image, threshold_value=127, max_value=255):
        """
        Apply global thresholding to a grayscale image.

        Parameters:
            image (numpy.ndarray): Input grayscale image.
            threshold_value (int): Threshold value.
            max_value (int): Maximum value for pixels above the threshold.

        Returns:
            numpy.ndarray: Thresholded image.
        """
        thresholded = np.where(image > threshold_value, max_value, 0).astype(np.uint8)
        return thresholded

    def local_threshold(image, blockSize=11, C=2, max_value=255):
        """
        Apply local thresholding to a grayscale image.

        Parameters:
            image (numpy.ndarray): Input grayscale image.
            blockSize (int): Size of the local neighborhood for computing the threshold value.
            C (int): Constant subtracted from the mean or weighted mean.
            max_value (int): Maximum value for pixels above the threshold.

        Returns:
            numpy.ndarray: Thresholded image.
        """
        thresholded = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Define the region of interest
                roi = image[max(0, i - blockSize // 2): min(image.shape[0], i + blockSize // 2),
                      max(0, j - blockSize // 2): min(image.shape[1], j + blockSize // 2)]
                # Compute the threshold value for the region
                threshold_value = np.mean(roi) - C
                # Apply thresholding
                thresholded[i, j] = max_value if image[i, j] > threshold_value else 0
        return thresholded

    def hybrid_images(self, image1_low_pass_filter, image2_high_pass_filter, low_threshold=10, low_gain=1, high_threshold=10, high_gain=1):
        hybrid_image = image1_low_pass_filter + image2_high_pass_filter
        hybrid_image = np.real(hybrid_image)
        hybrid_image = cv2.normalize(hybrid_image, None, 0, 255, cv2.NORM_MINMAX)
        hybrid_image = np.uint8(hybrid_image)

        return hybrid_image


class FourierTransform:
    def __init__(self):
        pass

    @staticmethod
    def get_img_fft(img):
        img_fft = np.fft.fft2(img)

        # apply shift
        img_fft_shifted = np.fft.fftshift(img_fft)

        return img_fft_shifted

    @staticmethod
    def get_inverse_fft(img_fft_shifted):
        img_fft = np.fft.ifftshift(img_fft_shifted)
        img = np.fft.ifft2(img_fft)

        return img

    @staticmethod
    def low_pass_filter(img_fft, threshold, gain):
        mask = np.zeros_like(img_fft)
        rows, cols = img_fft.shape
        center_row, center_col = rows // 2, cols // 2
        mask[center_row - threshold:center_row + threshold, center_col - threshold:center_col + threshold] = gain
        return img_fft * mask

    @staticmethod
    def high_pass_filter(img_fft, threshold, gain):
        mask = np.ones_like(img_fft) * gain
        rows, cols = img_fft.shape
        center_row, center_col = rows // 2, cols // 2
        mask[center_row - threshold:center_row + threshold, center_col - threshold:center_col + threshold] = 0

        return img_fft * mask


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Add salt noise
    salt_pixels = int(total_pixels * salt_prob)
    salt_coordinates = [np.random.randint(0, i - 1, salt_pixels) for i in image.shape]
    noisy_image[salt_coordinates[0], salt_coordinates[1]] = 255

    # Add pepper noise
    pepper_pixels = int(total_pixels * pepper_prob)
    pepper_coordinates = [np.random.randint(0, i - 1, pepper_pixels) for i in image.shape]
    noisy_image[pepper_coordinates[0], pepper_coordinates[1]] = 0

    return noisy_image


def add_uniform_noise(image, intensity=50):
    """Add uniform noise to an image.

    Args:
        image (numpy.ndarry): Input image.
        intensity (int): Intensity of uniform noise.

    Returns:
        noisy_image (numpy.ndarray): The image after applying noise modifier.

    """

    noise = np.random.uniform(-intensity, intensity, image.shape).astype('uint8')
    noisy_image = cv2.add(image, noise)

    return noisy_image
