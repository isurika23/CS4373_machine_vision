"""
    Run the image processing pipeline on a folder of images.
    python pipeline.py <input_folder> [--output output.txt] [--debug]
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- Image Loading and Gray scaling --------------------------------

def natural_sort_key(path: Path) -> list:
    """ Sort images numerically by filename, so "image2.png" comes before "image10.png". """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]

def load_images(folder: str) -> list:
    """ Load every .png / .jpg image in the folder.  Returns [(file_name, grayscale image), ...]. """
    paths = [p for p in Path(folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    images = []
    for path in sorted(paths, key=natural_sort_key):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping unreadable file: {path.name}")
            continue
        image = image.astype(np.uint8)
        images.append((path.name, image))
    return images

#----------------------------- Noise Reduction --------------------------------
# Reduce noise in the image using a combination of median and Gaussian filters.

def pad_image(image: np.ndarray, pad_rows: int, pad_cols: int) -> np.ndarray:
    """ # Image padding - Replicate edge pixels """
    return np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="edge")

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """ 2-D convolution:  g(x,y) = sum_i sum_j  h(i,j) * f(x-i, y-j) """
    kernel = np.flipud(np.fliplr(kernel))              # convolution flips the kernel
    k_rows, k_cols = kernel.shape
    padded = pad_image(image.astype(np.float64), k_rows // 2, k_cols // 2)
    rows, cols = image.shape
    output = np.zeros((rows, cols), dtype=np.float64)
    for i in range(k_rows):
        for j in range(k_cols):
            if kernel[i, j] != 0:
                output += kernel[i, j] * padded[i:i + rows, j:j + cols]
    return output

def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """ median filter - removes outliers (salt & pepper noise) without blurring edges """
    pad = kernel_size // 2
    padded = pad_image(image, pad, pad)
    # Every k x k neighbourhood as a view: shape (rows, cols, k, k)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    return np.median(windows, axis=(2, 3)).astype(image.dtype)

def gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """ Samples of G(x) = exp(-x^2 / 2 sigma^2), normalised so the weights sum to 1 """
    x = np.arange(size) - size // 2
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def gaussian_kernel_size(sigma: float) -> int:
    """ Kernel must cover +-3 sigma (99.7 % of the bell curve): size = 2 * ceil(3 sigma) + 1 """
    return 2 * int(np.ceil(3 * sigma)) + 1

def gaussian_filter(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """ Gaussian filter - removes Gaussian noise while preserving edges better than mean filter
        Uses separable convolution: 2-D Gaussian = 1-D Gaussian in x * 1-D Gaussian in y """
    g = gaussian_kernel_1d(gaussian_kernel_size(sigma), sigma)
    smoothed = convolve2d(image, g.reshape(1, -1))      # horizontal pass
    smoothed = convolve2d(smoothed, g.reshape(-1, 1))   # vertical pass
    return smoothed


def reduce_noise(image: np.ndarray) -> np.ndarray:
    """ Noise reduction: 
            Median first (remove outliers so they are not smeared)
            then Gaussian (remove Gaussian noise) """
    denoised = median_filter(image, kernel_size=5)
    # Strong smoothing on purpose (sigma = 3 -> 19x19 kernel): only the silhouette is needed,
    # so surface texture is smoothed away to avoid holes/specks after thresholding.
    denoised = gaussian_filter(denoised, sigma=3.0)
    return np.clip(np.round(denoised), 0, 255).astype(np.uint8)


# ----------------------------- Debugging / Visualization --------------------------------

def save_stages(name: str, stages: list, out_dir: Path):
    """Save a side-by-side figure of the processing stages for one image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))
    for ax, (title, img) in zip(np.atleast_1d(axes), stages):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255 if img.max() > 1 else 1)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(name).stem}_stages.png", dpi=80)
    plt.close(fig)

def main():
    """Main function: load images -> grayscale -> denoise -> save debug images if requested. """
    parser = argparse.ArgumentParser(description="Group images of similar objects using RST-invariant chain codes.")
    parser.add_argument("input_folder", help="folder containing the .png / .jpg images")
    parser.add_argument("--output", default="output.txt", help="output text file")
    parser.add_argument("--debug", action="store_true", help="save intermediate images to ./debug")
    args = parser.parse_args()

    images = load_images(args.input_folder)
    print(f"Loaded {len(images)} images from {args.input_folder}")

    for name, image in images:
        denoised = reduce_noise(image)
        print(f"{name}: {image.shape[1]}x{image.shape[0]} denoised")
        if args.debug:
            save_stages(name, [("Original", image), ("Denoised", denoised)], Path("debug"))

if __name__ == "__main__":
    main()
