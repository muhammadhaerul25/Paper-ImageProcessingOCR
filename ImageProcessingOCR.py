import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

class ImageProcessingOCR:
    def __init__(self, image):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image input. Image must be a NumPy array.")
        if len(image.shape) == 2:
            self.image_gray = image
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                self.image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                self.image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                raise ValueError("Unsupported image format.")
        else:
            raise ValueError("Unsupported image format.")

    def check_skew_angle(self):
        threshold_angle = 1
        edges = cv2.Canny(self.image_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is None:
            return 0.0, False
        angles = []
        for line in lines:
            _, theta = line[0]
            angle = np.degrees(theta) - 90
            angles.append(angle)
        skew_angle = float(np.median(angles))
        needs_enhancement = bool(abs(skew_angle) > threshold_angle)
        return skew_angle, needs_enhancement

    def check_brightness(self):
        min_brightness = 100
        max_brightness = 150
        brightness = float(np.mean(self.image_gray))
        needs_enhancement = bool(brightness < min_brightness or brightness > max_brightness)
        return brightness, needs_enhancement

    def check_contrast(self):
        contrast_threshold = 50
        contrast = float(np.std(self.image_gray))
        needs_enhancement = bool(contrast < contrast_threshold)
        return contrast, needs_enhancement

    def check_sharpness(self):
        sharpness_threshold = 1500
        sharpness = float(cv2.Laplacian(self.image_gray, cv2.CV_64F).var())
        needs_enhancement = bool(sharpness < sharpness_threshold)
        return sharpness, needs_enhancement

    def check_image_quality(self):
        skew_angle, skew_needs_enhancement = self.check_skew_angle()
        brightness, brightness_needs_enhancement = self.check_brightness()
        contrast, contrast_needs_enhancement = self.check_contrast()
        sharpness, sharpness_needs_enhancement = self.check_sharpness()
        
        result = {
            'skew_angle': {'value': skew_angle, 'needs_enhancement': skew_needs_enhancement},
            'brightness': {'value': brightness, 'needs_enhancement': brightness_needs_enhancement},
            'contrast': {'value': contrast, 'needs_enhancement': contrast_needs_enhancement},
            'sharpness': {'value': sharpness, 'needs_enhancement': sharpness_needs_enhancement}
        }
        
        return result

    def apply_adjust_brightness(self):
        avg_brightness = np.mean(self.image_gray)
        target_brightness = 128
        gamma = np.log(target_brightness) / np.log(avg_brightness) if avg_brightness > 0 else 1.0
        image_adjusted_brightness = np.array(255 * (self.image_gray / 255) ** gamma, dtype='uint8')
        return image_adjusted_brightness

    def apply_adjust_contrast(self):
        clip_limit = 2.0
        tile_grid_size = (8, 8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        image_adjusted_contrast = clahe.apply(self.image_gray)
        return image_adjusted_contrast

    def apply_deskew(self):
        edges = cv2.Canny(self.image_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        angles = []
        for line in lines:
            _, theta = line[0]
            angle = np.degrees(theta) - 90
            angles.append(angle)
        if not angles:
            return self.image_gray
        median_angle = np.median(angles)
        image_rotated = rotate(self.image_gray, median_angle, reshape=False)
        mask = (image_rotated == 0)
        image_deskew = image_rotated.copy()
        image_deskew[mask] = 255
        return image_deskew

    def apply_sharpening(self):
        sharpening_kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])
        image_sharpened = cv2.filter2D(self.image_gray, -1, sharpening_kernel)
        return image_sharpened

    def process_image_enhancement(self):
        if self.check_skew_angle()[1]:
            self.image_gray = self.apply_deskew()
        if self.check_brightness()[1]:
            self.image_gray = self.apply_adjust_brightness()
        if self.check_contrast()[1]:
            self.image_gray = self.apply_adjust_contrast()
        if self.check_sharpness()[1]:
            self.image_gray = self.apply_sharpening()
        image_enhanced = self.image_gray.copy()
        return image_enhanced