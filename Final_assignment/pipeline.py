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


# ================================ Image Loading and Preprocessing ================================

# -------------------------------- Image Loading and Gray scaling ---------------------------------

def natural_sort_key(path: Path) -> list:
    """ Sort images numerically by filename, so "image2.png" comes before "image10.png". """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]

def load_images(folder: str) -> list:
    """ Load every .png / .jpg image in the folder. Returns [(file_name, grayscale image), ...]. """
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


# ========================================= Segmentation ==========================================

# ----------------------------------------- Histogram ---------------------------------------------

NEIGHBOURS_8 = [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)] # order for contour tracking
NEIGHBOURS_4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def compute_histogram(image: np.ndarray) -> np.ndarray:
    """ Grey-level histogram: h(i) = number of pixels with intensity i, i = 0..255 """
    histogram = np.zeros(256, dtype=np.int64)
    np.add.at(histogram, image.ravel(), 1)
    return histogram

def smooth_histogram(histogram: np.ndarray, width: int = 9) -> np.ndarray:
    """ Histogram smoothing: h(i) = AVE(h(i - w/2) ... h(i + w/2))
        - removes small false peaks/valleys """
    half = width // 2
    padded = np.pad(histogram.astype(np.float64), half, mode="edge")
    smoothed = np.zeros(256, dtype=np.float64)
    for k in range(width):
        smoothed += padded[k:k + 256]
    return smoothed / width

# -------------------------------------- Threshold Selection --------------------------------------

def inter_means_threshold(histogram: np.ndarray, tolerance: float = 0.5) -> float:
    """ Inter-means algorithm: start at the mean intensity, then repeat T = (mu1 + mu2) / 2
        until the two class means stop changing """
    levels = np.arange(256)
    threshold = np.sum(levels * histogram) / np.sum(histogram)
    mu1 = mu2 = None
    while True:
        below = levels <= threshold
        new_mu1 = np.sum(levels[below] * histogram[below]) / max(np.sum(histogram[below]), 1)
        new_mu2 = np.sum(levels[~below] * histogram[~below]) / max(np.sum(histogram[~below]), 1)
        threshold = (new_mu1 + new_mu2) / 2
        if mu1 is not None and abs(new_mu1 - mu1) < tolerance and abs(new_mu2 - mu2) < tolerance:
            return threshold
        mu1, mu2 = new_mu1, new_mu2

def valley_threshold(histogram: np.ndarray) -> float:
    """ Valley point: start at the largest peak (the background peak) and walk towards the
        object peak until the histogram starts rising again. Falls back to inter-means
        if no valley exists (histogram not bi-modal).
            The object peak lies on the side of the background peak that holds more pixels
            (works even if both peaks are in the same half of the grey scale) """
    peak = int(np.argmax(histogram))
    left_mass = histogram[:peak].sum()
    right_mass = histogram[peak + 1:].sum()
    step = 1 if right_mass > left_mass else -1
    valley = peak
    while 0 < valley + step < 255 and histogram[valley + step] <= histogram[valley]:
        valley += step
    if not 0 < valley + step < 255:             # reached the end of the grey scale: no valley
        return inter_means_threshold(histogram)
    return float(valley)

def threshold_image(image: np.ndarray, threshold: float) -> np.ndarray:
    """ g(x,y) = 1 if f(x,y) > T else 0.  If the border turns out mostly 1 the object is darker
        than the background, so the result is inverted (object is always 1) """
    binary = (image > threshold).astype(np.uint8)
    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if border.mean() > 0.5:
        binary = 1 - binary
    return binary

# ---------------------------------------- Binary Clean-up ----------------------------------------

def connected_components(binary: np.ndarray) -> (np.ndarray, int):
    """ Connected component labelling by region growing: every unlabelled object pixel is a seed,
        and its 8-connected object neighbours get the same label """
    rows, cols = binary.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    count = 0
    for r, c in zip(*np.nonzero(binary)):
        if labels[r, c]:
            continue
        count += 1
        labels[r, c] = count
        stack = [(r, c)]
        while stack:
            y, x = stack.pop()
            for dy, dx in NEIGHBOURS_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and binary[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = count
                    stack.append((ny, nx))
    return labels, count

def largest_component(binary: np.ndarray) -> np.ndarray:
    """ Keep only the biggest region (the object); drops leftover stars and noise specks """
    labels, count = connected_components(binary)
    if count == 0:
        return binary
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0                                # label 0 is the background
    return (labels == np.argmax(sizes)).astype(np.uint8)

def fill_holes(binary: np.ndarray) -> np.ndarray:
    """ Region-grow the background from the image border (4-connected). Background pixels that
        cannot be reached are holes inside the object, so they are set to 1 """
    rows, cols = binary.shape
    outside = np.zeros((rows, cols), dtype=bool)
    border = [(r, c) for r in range(rows) for c in (0, cols - 1)] + \
             [(r, c) for c in range(cols) for r in (0, rows - 1)]
    stack = []
    for r, c in border:
        if not binary[r, c] and not outside[r, c]:
            outside[r, c] = True
            stack.append((r, c))
    while stack:
        y, x = stack.pop()
        for dy, dx in NEIGHBOURS_4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and not binary[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    return (~outside).astype(np.uint8)

# ------------------------------------- Segmentation Pipeline -------------------------------------

def segment_object(image: np.ndarray) -> (np.ndarray, np.ndarray, float):
    """ Segmentation: histogram -> smooth -> valley threshold -> binary image -> clean-up.
        Returns (object mask, raw thresholded image, threshold) """
    histogram = smooth_histogram(smooth_histogram(compute_histogram(image)))   # smoothed twice
    threshold = valley_threshold(histogram)
    binary = threshold_image(image, threshold)
    # Median filter on a binary image stays binary: removes specks and smooths ragged edges
    mask = median_filter(binary, kernel_size=7)
    mask = largest_component(mask)
    mask = fill_holes(mask)
    return mask, binary, threshold


# ===================================== Boundary Representation ===================================

# ------------------------------------------ Contour Tracking -------------------------------------

def trace_contour(mask: np.ndarray) -> np.ndarray:
    """ Contour tracking: 8-connected boundary pixels are traced in clockwise order.
            1. first boundary pixel = first object pixel found scanning from the top-left corner
            2. store it as the start point and set n = 1
            3. check direction m = (n - 2) mod 8
            4. if that pixel is object: move there, n = m, go to 3; otherwise m = (m + 1) mod 8
            5. stop when the start point is reached as the next pixel
        Returns the boundary pixels in order (clockwise) as an array of (row, col) """
    padded = np.pad(mask, 1)                    # neighbours of edge pixels always exist
    rows, cols = np.nonzero(padded)
    first = np.lexsort((cols, rows))[0]         # smallest row, then smallest column
    start = (rows[first], cols[first])

    boundary = [start]
    current, n = start, 1
    while True:
        m = (n - 2) % 8
        for _ in range(8):
            dy, dx = NEIGHBOURS_8[m]
            candidate = (current[0] + dy, current[1] + dx)
            if padded[candidate]:
                break
            m = (m + 1) % 8
        else:
            break                               # isolated single pixel: no neighbours
        if candidate == start:
            break
        boundary.append(candidate)
        current, n = candidate, m
    return np.array(boundary) - 1               # undo the padding offset

# --------------------------------------------- Chain Code ----------------------------------------

def direction_code(d_row: float, d_col: float) -> int:
    """ Freeman 8-direction code of a vector (Unit 7):  3 2 1 / 4 . 0 / 5 6 7
        i.e. 0 = East, then anti-clockwise in 45 degree steps. Image rows grow downwards,
        so the upward component is -d_row. The vector is rounded to the nearest direction. """
    angle = np.arctan2(-d_row, d_col)
    return int(np.round(angle / (np.pi / 4))) % 8

def pixel_chain_code(boundary: np.ndarray) -> list:
    """ Method 1 chain code: one direction per pair of neighbouring boundary pixels
        (the closed boundary, so the last pixel connects back to the first) """
    closed = np.vstack([boundary, boundary[:1]])
    steps = np.diff(closed, axis=0)
    return [direction_code(d_row, d_col) for d_row, d_col in steps]

def cumulative_perimeter(pixel_code: list) -> np.ndarray:
    """ Perimeter from the chain code: even codes add 1, odd (diagonal) codes add sqrt(2).
        Returns the running length along the boundary, starting at 0 """
    step_lengths = [1.0 if c % 2 == 0 else np.sqrt(2) for c in pixel_code]
    return np.concatenate([[0.0], np.cumsum(step_lengths)])

def subdivide_boundary(boundary: np.ndarray, segments: int = 32) -> np.ndarray:
    """ Method 3: divide the boundary into equal-length segments.
        Returns the segments + 1 end points (the last one equals the first: closed boundary) """
    arc = cumulative_perimeter(pixel_chain_code(boundary))
    closed = np.vstack([boundary, boundary[:1]])
    targets = np.arange(segments + 1) * arc[-1] / segments      # equally spaced lengths
    indices = np.searchsorted(arc, targets)                    # first pixel at/after each length
    indices = np.minimum(indices, len(closed) - 1)
    return closed[indices]

def segment_chain_code(points: np.ndarray) -> list:
    """ Join consecutive end points with straight lines and code each line's direction """
    steps = np.diff(points, axis=0)
    return [direction_code(d_row, d_col) for d_row, d_col in steps]

def boundary_chain_code(mask: np.ndarray, segments: int = 32) -> (list, np.ndarray, np.ndarray):
    """ Object mask -> boundary -> 32 equal segments -> 32-digit chain code.
        Returns (chain code, boundary pixels, segment end points) """
    boundary = trace_contour(mask)
    points = subdivide_boundary(boundary, segments)
    return segment_chain_code(points), boundary, points


# --------------------------------- Debugging / Visualization -------------------------------------

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

def save_histogram(name: str, image: np.ndarray, threshold: float, out_dir: Path):
    """Plot the raw and smoothed histogram with the selected threshold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = compute_histogram(image)
    smooth = smooth_histogram(smooth_histogram(raw))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(raw, color="lightgray", label="histogram")
    ax.plot(smooth, color="tab:blue", label="smoothed")
    ax.axvline(threshold, color="tab:red", linestyle="--", label=f"T = {threshold:.0f}")
    ax.set_xlabel("grey level")
    ax.set_ylabel("pixel count")
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(name).stem}_histogram.png", dpi=80)
    plt.close(fig)

def save_chain_code(name: str, image: np.ndarray, boundary: np.ndarray, points: np.ndarray,
                    code: list, out_dir: Path):
    """Draw the traced boundary, the 32 straight segments and each segment's code digit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.plot(boundary[:, 1], boundary[:, 0], color="yellow", linewidth=1, label="traced boundary")
    ax.plot(points[:, 1], points[:, 0], "-o", color="red", markersize=3, linewidth=1.5,
            label="32 segments")
    ax.plot(points[0, 1], points[0, 0], "s", color="cyan", markersize=8, label="start")
    for k, digit in enumerate(code):
        mid = (points[k] + points[k + 1]) / 2
        ax.text(mid[1], mid[0], str(digit), color="white", fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))
    ax.set_title(f"{name}: {''.join(map(str, code))}", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(name).stem}_chaincode.png", dpi=80)
    plt.close(fig)

# ========================================= Main Function =========================================

def main():
    """Main function: load images (grayscale) -> denoise -> segment -> chain code -> save debug images if requested. """
    parser = argparse.ArgumentParser(description="Group images of similar objects using RST-invariant chain codes.")
    parser.add_argument("input_folder", help="folder containing the .png / .jpg images")
    parser.add_argument("--output", default="output.txt", help="output text file")
    parser.add_argument("--debug", action="store_true", help="save intermediate images to ./debug")
    args = parser.parse_args()

    images = load_images(args.input_folder)
    print(f"Loaded {len(images)} images from {args.input_folder}")

    for name, image in images:
        denoised = reduce_noise(image)
        mask, binary, threshold = segment_object(denoised)
        code, boundary, points = boundary_chain_code(mask, segments=32)
        print(f"{name}: T = {threshold:.0f}, area = {int(mask.sum())} px, "
              f"boundary = {len(boundary)} px, chain code = {''.join(map(str, code))}")
        if args.debug:
            save_stages(name, [("Original", image), ("Denoised", denoised),
                               (f"Thresholded (T={threshold:.0f})", binary), ("Object mask", mask)],
                        Path("debug"))
            save_histogram(name, denoised, threshold, Path("debug"))
            save_chain_code(name, image, boundary, points, code, Path("debug"))

if __name__ == "__main__":
    main()
