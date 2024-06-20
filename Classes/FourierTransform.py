import numpy as np

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