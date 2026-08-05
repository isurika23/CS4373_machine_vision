import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def resample_image(image, scale_factor):
    original_height, original_width = image.shape
    print(f"Original image dimensions: {original_height}x{original_width}")

    # new dimensions based on scale factor
    new_height = max(1, int(round(original_height * scale_factor)))
    new_width = max(1, int(round(original_width * scale_factor)))
    print(f"Resampled image dimensions: {new_height}x{new_width}")

    # Calculate exact scale ratios based on the dimensions
    scale_i = new_height / original_height
    scale_j = new_width / original_width

    # Create a new image with the resampled dimensions
    resampled_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            # Map coordinates using the center-aligned formula
            mapped_i = (i + 0.5) / scale_i - 0.5
            mapped_j = (j + 0.5) / scale_j - 0.5

            # calculate the base indices for bilinear interpolation
            base_i = int(np.floor(mapped_i))
            base_j = int(np.floor(mapped_j))

            # four corner coordinates within the image boundaries
            i1 = np.clip(base_i, 0, original_height - 1)
            i2 = np.clip(base_i + 1, 0, original_height - 1)
            j1 = np.clip(base_j, 0, original_width - 1)
            j2 = np.clip(base_j + 1, 0, original_width - 1)

            # fractional parts for interpolation
            a = mapped_i - base_i
            b = mapped_j - base_j

            # ensure fractional parts are within [0,1]
            a = float(np.clip(a, 0.0, 1.0))
            b = float(np.clip(b, 0.0, 1.0))

            # pixel value calculation using bilinear interpolation (float)
            p11 = float(image[i1, j1])
            p21 = float(image[i2, j1])
            p12 = float(image[i1, j2])
            p22 = float(image[i2, j2])

            pixel_value = (1 - a) * (1 - b) * p11 + \
                          a * (1 - b) * p21 + \
                          (1 - a) * b * p12 + \
                          a * b * p22

            # clamp and assign
            resampled_image[i, j] = np.uint8(np.clip(pixel_value, 0, 255))

    return resampled_image

def display_images(original_image, resampled_image1, resampled_image2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(resampled_image1, cmap='gray')
    plt.title('Resampled Image (Scale 0.7)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(resampled_image2, cmap='gray')
    plt.title('Restored Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def calculate_mse(image1, image2):

    diff = image1.astype(np.float32) - image2.astype(np.float32)
    mse = np.sum(diff ** 2) / (image1.shape[0] * image1.shape[1])
    return mse

def main():
    # build a workspace-relative path (works regardless of current working directory)
    image_path = Path(__file__).resolve().parent.parent / "sample_images" / "test_bnw3.jpg"
    image_path = str(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(image)
    print(image.shape)

    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}.\nCheck the path and file integrity.\nCurrent working directory: {os.getcwd()}")

    # ensure output directory exists
    out_dir = Path(__file__).resolve().parent.parent / "output_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    scale_factor = 0.7
    print(f"Resampling image with scale factor: {scale_factor}")
    resampled_image1 = resample_image(image, scale_factor)
    cv2.imwrite(str(out_dir / "downscaled.jpg"), resampled_image1)
    print(f"Downscaled image dimensions: {resampled_image1.shape[0]}x{resampled_image1.shape[1]}")

    # Step 4: Resample back to original size
    print(f"Resampling image back to original size with scale factor: {1 / scale_factor}")
    resampled_image2 = resample_image(resampled_image1, 1 / scale_factor)
    cv2.imwrite(str(out_dir / "upscaled.jpg"), resampled_image2)

    # error calculation
    mse = calculate_mse(image, resampled_image2)
    print(f"Mean Squared Error: {mse}")

    display_images(image, resampled_image1, resampled_image2)

if __name__ == "__main__":
    main()