import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_padding(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Pad the image in cyclic manner based on the kernel size."""
    pad_size = kernel_size // 2

    first_row = image[0:pad_size, :]
    last_row = image[-pad_size:, :]

    wrapped_img = image.copy()
    wrapped_img = np.vstack((last_row, wrapped_img, first_row))

    first_col = wrapped_img[:, 0:pad_size]
    last_col = wrapped_img[:, -pad_size:]

    wrapped_img = np.hstack((last_col, wrapped_img, first_col))
    return wrapped_img

def mean_filter(image_shape: tuple, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a mean filter to the image."""
    filtered_image = np.zeros(image_shape, dtype=np.float32)
    image_height, image_width = image_shape
    pad_size = kernel_size // 2

    for i in range(pad_size, image_height - pad_size):
        for j in range(pad_size, image_width - pad_size):
            region = image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            filtered_image[i, j] = int(np.mean(region))
    return filtered_image



def median_filter(image_shape: tuple, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a median filter to the image."""
    filtered_image = np.zeros(image_shape, dtype=np.float32)
    image_height, image_width = image_shape
    pad_size = kernel_size // 2

    for i in range(pad_size, image_height - pad_size):
        for j in range(pad_size, image_width - pad_size):
            region = image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            filtered_image[i, j] = int(np.median(region))
    return filtered_image



def mid_point_filter(image_shape: tuple, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a midpoint filter to the image."""
    filtered_image = np.zeros(image_shape, dtype=np.float32)
    image_height, image_width = image_shape
    pad_size = kernel_size // 2

    for i in range(pad_size, image_height - pad_size):
        for j in range(pad_size, image_width - pad_size):
            region = image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            filtered_image[i, j] = int((np.min(region) + np.max(region)) / 2)
    return filtered_image

def main():
    # Load the image
    images = ["butterfly_gau.jpg", "butterfly_snp.jpg", "butterfly_str.jpg"]
    for img_name in images:
        image_path = Path(__file__).resolve().parent / "images" / "noised" / img_name
        image_path = str(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}.\nCheck the path and file integrity.\nCurrent working directory: {os.getcwd()}")
        
        print(f"Loaded image shape: {image.shape}")
        print(f"Loaded image : {img_name}")

        # Apply filters
        padded_image = image_padding(image, kernel_size=3)
        mean_filtered = mean_filter(image.shape, padded_image, kernel_size=3)
        median_filtered = median_filter(image.shape, padded_image, kernel_size=3)
        mid_point_filtered = mid_point_filter(image.shape, padded_image, kernel_size=3)

        # Display results
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title('Mean Filtered Image')
        plt.imshow(mean_filtered, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.title('Median Filtered Image')
        plt.imshow(median_filtered, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title('Midpoint Filtered Image')
        plt.imshow(mid_point_filtered, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()