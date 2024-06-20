import numpy as np
import cv2

class AddNoise:
    def __init__(self):
        pass

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

    def add_gaussian_noise(image, mean=0.0, std=25.0):
        """Add gaussian noise to an image.

        Args:
            image (numpy.ndarry): Input image.
            mean (float): Mean value for Gaussian noise.
            std (float): Standard deviation value for Gaussian noise.

        Returns:
            noisy_image (numpy.ndarray): The image after applying noise modifier.

        """

        noise = np.random.normal(mean, std, image.shape).astype('uint8')
        noisy_image = cv2.add(image, noise)

        return noisy_image
