import numpy as np
import cv2
from scipy.signal import convolve2d
import threading


def matrix_padding(img, size):
    # Function to pad the image matrix for filter operations
    return np.pad(img, ((size // 2, size // 2), (size // 2, size // 2)), mode='constant')


class Filter:
    def __init__(self, img):
        self.img_average = None
        self.img_median = None
        self.img_gaussian = None
        self.img_roberts = None
        self.img_prewitt = None
        self.img_canny = None
        self.img_sobel = None
        self.img_laplacian = None

        calculations_thread = threading.Thread(target=self.calc_filters, args=(img,))
        calculations_thread.start()

    def calc_filters(self, img):
        self.img_average = self.average(img, 3)
        self.img_median = self.median(img, 3)
        self.img_gaussian = self.gaussian_opencv(img, 3, 1.5)
        self.img_roberts = self.roberts(img)
        self.img_prewitt = self.prewitt(img)
        self.img_canny = self.canny(img)
        self.img_sobel = self.sobel(img)
        self.img_laplacian = self.laplace(img)

    # Smoothing filters
    def average(self, img, size):
        kernel = np.ones((size, size)) / (size * size)
        return (convolve2d(img, kernel, mode='same', boundary='symm')).astype(np.uint8)

    def median(self, img, size):
        result = np.zeros_like(img)

        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                window = img[i - 1:i + 2, j - 1:j + 2].flatten()
                result[i, j] = np.median(window)

        return (result).astype(np.uint8)

    def gaussian_opencv(self, img, size, sigma):
        kernel = cv2.getGaussianKernel(size, sigma)
        return (convolve2d(matrix_padding(img, size), kernel, mode='same', boundary='symm')).astype(np.uint8)

    def gaussian(self, img, kernel_number=1):
        if kernel_number == 1:
            kernel = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
        else:
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

        return (convolve2d(img, kernel, mode='same', boundary='')).astype(np.uint8)

    # Edge detection operators

    # # Using 1st derivative operators

    def roberts(self, img):
        k1 = np.array([[1, 0], [0, -1]])
        k2 = np.array([[0, 1], [-1, 0]])

        grad1 = convolve2d(img, k1, mode='same', boundary='symm')
        grad2 = convolve2d(img, k2, mode='same', boundary='symm')

        return (np.abs(grad1) + np.abs(grad2)).astype(np.uint8)

    def prewitt(self, img):
        kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])

        grad_x = convolve2d(img, kernel_x, mode='same', boundary='symm')
        grad_y = convolve2d(img, kernel_y, mode='same', boundary='symm')

        return (np.sqrt(grad_x ** 2 + grad_y ** 2)).astype(np.uint8)

    def sobel(self, img):
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = np.abs(convolve2d(img, kernel_x, mode='same', boundary='symm'))
        grad_y = np.abs(convolve2d(img, kernel_y, mode='same', boundary='symm'))

        return (grad_x + grad_y).astype(np.uint8)

    # # Using 2nd derivative operators

    def canny(self, img, sigma=1.0, low_threshold=30, high_threshold=60, ksize=3):
        blurred_img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        edges = cv2.Canny(blurred_img, low_threshold, high_threshold)
        return edges.astype(np.uint8)

    def laplace(self, img, kernel_number=1):
        if kernel_number == 1:
            laplace_kernel = np.array([[1, 1, 1],
                                       [1, -8, 1],
                                       [1, 1, 1]])
        else:
            laplace_kernel = np.array([[0, 1, 0],
                                       [1, -4, 1],
                                       [0, 1, 0]])

        laplace_result = convolve2d(img, laplace_kernel, mode='same', boundary='symm')

        # Convert to uint8 to ensure correct data type for image display and saving
        laplace_result = laplace_result.astype(np.uint8)

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


class FourierTransform:
    def __init__(self):
        pass

    def get_img_fft(self, img):
        img_fft = np.fft.fft2(img)

        # apply shift
        img_fft_shifted = np.fft.fftshift(img_fft)

        return img_fft_shifted

    def get_inverse_fft(self, img_fft_shifted):
        img_fft = np.fft.ifftshift(img_fft_shifted)
        img = np.fft.ifft2(img_fft)

        return img.astype(np.uint8)

    def low_pass_filter(self, img_fft, threshold, gain):
        mask = np.zeros_like(img_fft)
        rows, cols = img_fft.shape
        center_row, center_col = rows // 2, cols // 2
        mask[center_row - threshold:center_row + threshold, center_col - threshold:center_col + threshold] = gain
        return img_fft * mask

    def high_pass_filter(self, img_fft, threshold, gain):
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

    noisy_image = np.copy(image)
    noise = np.random.uniform(-intensity, intensity, image.shape).astype('uint8')
    noisy_image = cv2.add(image, noise)

    return noisy_image
