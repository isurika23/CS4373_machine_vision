"""
    Run the image processing pipeline on a folder of images.
    python pipeline.py <input_folder> [--output output.txt] [--debug] [--threshold 10]
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
    """ Freeman 8-direction code of a vector:  3 2 1 | 4 . 0 | 5 6 7
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

def subdivide_boundary(arc: np.ndarray, segments: int = 32, start_length: float = 0.0) -> np.ndarray:
    """ Method 3: divide the boundary into equal-length segments.
        arc = cumulative_perimeter of the boundary (computed once per image and reused).
        start_length = distance along the boundary where the first cut is made (0 = first pixel).
        Returns the indices of the segments + 1 end points (last = first: closed boundary) """
    perimeter = arc[-1]
    boundary_length = len(arc) - 1                             # arc has one entry per pixel + closing step
    targets = (start_length + np.arange(segments + 1) * perimeter / segments) % perimeter
    indices = np.searchsorted(arc, targets)                    # first pixel at/after each length
    return indices % boundary_length                           # index boundary_length = start pixel

def segment_chain_code(points: np.ndarray) -> list:
    """ Join consecutive end points with straight lines and code each line's direction """
    steps = np.diff(points, axis=0)
    return [direction_code(d_row, d_col) for d_row, d_col in steps]

def boundary_chain_code(mask: np.ndarray, segments: int = 32) -> (list, np.ndarray, np.ndarray, np.ndarray):
    """ Object mask -> boundary -> 32 equal segments -> 32-digit chain code.
        Returns (chain code, boundary pixels, segment end points, cumulative perimeter);
        the cumulative perimeter is passed on to the invariance step so it is computed only once """
    boundary = trace_contour(mask)
    arc = cumulative_perimeter(pixel_chain_code(boundary))
    points = boundary[subdivide_boundary(arc, segments)]
    return segment_chain_code(points), boundary, points, arc


# ====================================== RST Invariance ===========================================

# ---------------------------------- Orientation from Image Moments -------------------------------

def raw_moment(mask: np.ndarray, p: int, q: int) -> float:
    """ m_pq = sum_x sum_y x^p y^q  over the object pixels (binary image: f(x,y) = 1) """
    rows, cols = np.nonzero(mask)
    x, y = cols.astype(np.float64), rows.astype(np.float64)
    return float(np.sum(x ** p * y ** q))

def central_moments(mask: np.ndarray) -> (float, float, float, float, float):
    """ Centroid x = m10/m00, y = m01/m00 and the 2nd-order central moments:
            mu11 = m11 - y*m10,  mu20 = m20 - x*m10,  mu02 = m02 - y*m01
        Central moments are measured about the centroid, so they are translation invariant """
    m00, m10, m01 = raw_moment(mask, 0, 0), raw_moment(mask, 1, 0), raw_moment(mask, 0, 1)
    x_bar, y_bar = m10 / m00, m01 / m00
    mu11 = raw_moment(mask, 1, 1) - y_bar * m10
    mu20 = raw_moment(mask, 2, 0) - x_bar * m10
    mu02 = raw_moment(mask, 0, 2) - y_bar * m01
    return x_bar, y_bar, mu20, mu02, mu11

def principal_axis_angle(mu20: float, mu02: float, mu11: float) -> float:
    """ Orientation of the major (longest) axis: theta = 1/2 * atan2(2 mu11, mu20 - mu02) """
    return 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

def align_boundary(boundary: np.ndarray, x_bar: float, y_bar: float, theta: float) -> np.ndarray:
    """ Move the centroid to the origin and rotate by -theta, so the major axis lies along x.
        After this every rotated copy of a shape has the same orientation (up to 180 degrees) """
    x = boundary[:, 1] - x_bar
    y = boundary[:, 0] - y_bar
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    return np.stack([y_rot, x_rot], axis=1)                    # (row, col) like the boundary

def major_axis_ends(aligned: np.ndarray) -> list:
    """ Boundary pixels at the two ends of the major axis (largest and smallest x after alignment).
        These are start points that do not depend on the object's rotation or position.
        Both are kept because the axis direction is ambiguous by 180 degrees """
    return [int(np.argmax(aligned[:, 1])), int(np.argmin(aligned[:, 1]))]

# ------------------------------------- Chain Code Normalisation ----------------------------------

def first_difference(code: list) -> list:
    """ Rotation normalisation: C'_i = (C_i - C_(i-1)) mod 8, code treated as circular.
        Counts the turns between segments instead of absolute directions """
    return [(code[i] - code[i - 1]) % 8 for i in range(len(code))]

def normalize_start_point(code: list) -> list:
    """ Starting-point normalisation: the circular shift forming the smallest integer """
    return min(code[i:] + code[:i] for i in range(len(code)))

def invariant_chain_codes(mask: np.ndarray, boundary: np.ndarray, arc: np.ndarray, segments: int = 32,
                          phases: int = 4) -> list:
    """ RST and starting-point invariant chain codes of one object.
            Translation : directions only + moments about the centroid
            Scale       : fixed number of equal segments
            Rotation    : boundary aligned to the major axis, then first difference
            Start point : cuts start at a major-axis end, then smallest-integer shift
        The cuts are also tried at `phases` offsets inside the first segment (0, 1/4, 1/2, 3/4),
        because a small shift of the start pixel moves all 32 cuts.
        arc = cumulative perimeter from boundary_chain_code (reused, not recomputed).
        Returns all candidate codes (2 axis ends x phases); the grouping step compares these sets """
    x_bar, y_bar, mu20, mu02, mu11 = central_moments(mask)
    aligned = align_boundary(boundary, x_bar, y_bar, principal_axis_angle(mu20, mu02, mu11))
    segment_length = arc[-1] / segments
    candidates = []
    for start in major_axis_ends(aligned):
        for p in range(phases):
            indices = subdivide_boundary(arc, segments, arc[start] + p * segment_length / phases)
            code = segment_chain_code(aligned[indices])         # directions after alignment
            candidates.append(normalize_start_point(first_difference(code)))
    return candidates

def canonical_code(candidates: list) -> list:
    """ One representative code per object: the smallest integer among the candidates
        (the same 'minimum magnitude' rule as the starting-point normalisation) """
    return min(candidates)


# =========================================== Grouping ============================================

# ----------------------------------------- Code Distance -----------------------------------------

def digit_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Circular difference between code digits: 0 and 7 are 1 step apart, not 7.
        d = min(|a - b|, 8 - |a - b|)  ->  0 (same turn) .. 4 (opposite turn) """
    d = np.abs(a - b) % 8
    return np.minimum(d, 8 - d)

def code_distance(code_a: list, code_b: list) -> int:
    """ Distance between two invariant codes: sum of digit differences, at the best circular
        alignment. All shifts are tried because one changed digit can move the
        smallest-integer start of an otherwise identical code """
    a = np.array(code_a)
    shifts = np.array([np.roll(code_b, s) for s in range(len(code_b))])   # every circular shift of b
    return int(digit_difference(a, shifts).sum(axis=1).min())

def chain_code_distance(candidates_a: list, candidates_b: list) -> int:
    """ Distance between two objects: the closest pair among their candidate codes
        (2 axis ends x 4 offsets each). Same shape -> one pair matches (almost) exactly """
    return min(code_distance(a, b) for a in candidates_a for b in candidates_b)

def distance_matrix(all_candidates: list) -> np.ndarray:
    """ Symmetric matrix D[i, j] = chain-code distance between image i and image j """
    n = len(all_candidates)
    D = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = chain_code_distance(all_candidates[i], all_candidates[j])
    return D

# ------------------------------------------ Grouping ---------------------------------------------

def find_root(parent: list, i: int) -> int:
    """ Follow parent links up to the group's representative (union-find) """
    while parent[i] != i:
        parent[i] = parent[parent[i]]           # path halving keeps the trees short
        i = parent[i]
    return i

def group_images(D: np.ndarray, threshold: int) -> list:
    """ Two images are linked if their distance <= threshold; groups are the connected
        components of these links. Returns a list of groups of image indices """
    n = len(D)
    parent = list(range(n))                     # every image starts in its own group
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= threshold:
                parent[find_root(parent, j)] = find_root(parent, i)    # merge the two groups
    groups = {}
    for i in range(n):
        groups.setdefault(find_root(parent, i), []).append(i)
    return list(groups.values())

# --------------------------------------------- Output --------------------------------------------

def write_output(path: Path, groups: list, results: list, threshold: int):
    """ Text file: the groups of image names, then each image's invariant chain code """
    lines = [f"Image groups (chain-code distance threshold = {threshold})", ""]
    for k, group in enumerate(groups, start=1):
        lines.append(f"Group {k}: {', '.join(results[i]['name'] for i in group)}")
    lines += ["", "RST and starting-point invariant chain codes (32 digits):"]
    width = max(len(r["name"]) for r in results)
    for r in results:
        lines.append(f"{r['name']:<{width}} : {''.join(map(str, r['invariant']))}")
    path.write_text("\n".join(lines) + "\n")


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

def make_grid(count: int, cell_width: float, cell_height: float, columns: int = 4):
    """Figure with `columns` plots per row and as many rows as needed; unused cells are hidden."""
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(cell_width * columns, cell_height * rows),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes[count:]:
        ax.axis("off")
    return fig, axes[:count]

def plot_histogram(ax, name: str, image: np.ndarray, threshold: float, show_legend: bool):
    """One panel: raw and smoothed histogram with the selected threshold."""
    raw = compute_histogram(image)
    smooth = smooth_histogram(smooth_histogram(raw))
    ax.plot(raw, color="lightgray", label="histogram")
    ax.plot(smooth, color="tab:blue", label="smoothed")
    ax.axvline(threshold, color="tab:red", linestyle="--", label="threshold")
    ax.set_title(f"{name}  (T = {threshold:.0f})", fontsize=10)
    ax.set_xlim(0, 255)
    ax.set_xlabel("grey level", fontsize=8)
    ax.set_ylabel("pixel count", fontsize=8)
    ax.tick_params(labelsize=7)
    if show_legend:
        ax.legend(fontsize=7)

def plot_chain_code(ax, name: str, image: np.ndarray, boundary: np.ndarray, points: np.ndarray,
                    code: list, show_legend: bool):
    """One panel: traced boundary, the 32 straight segments and each segment's code digit."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.plot(boundary[:, 1], boundary[:, 0], color="yellow", linewidth=1, label="traced boundary")
    ax.plot(points[:, 1], points[:, 0], "-o", color="red", markersize=2, linewidth=1.2,
            label="32 segments")
    ax.plot(points[0, 1], points[0, 0], "s", color="cyan", markersize=6, label="start")
    for k, digit in enumerate(code):
        mid = (points[k] + points[k + 1]) / 2
        ax.text(mid[1], mid[0], str(digit), color="white", fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6))
    ax.set_title(f"{name}\n{''.join(map(str, code))}", fontsize=8, family="monospace")
    ax.axis("off")
    if show_legend:
        ax.legend(loc="lower right", fontsize=6)

def save_histogram(result: dict, out_dir: Path):
    """One image's histogram as its own file: out_dir/<image name>.png"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_histogram(ax, result["name"], result["denoised"], result["threshold"], show_legend=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(result['name']).stem}_histogram.png", dpi=80)
    plt.close(fig)

def save_chain_code(result: dict, out_dir: Path):
    """One image's chain-code drawing as its own file: out_dir/<image name>.png"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_chain_code(ax, result["name"], result["image"], result["boundary"], result["points"],
                    result["code"], show_legend=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(result['name']).stem}_chain_code.png", dpi=100)
    plt.close(fig)

def save_histogram_grid(results: list, out_dir: Path):
    """All histograms in one figure (4 per row)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = make_grid(len(results), cell_width=4.5, cell_height=3.2)
    for k, (ax, r) in enumerate(zip(axes, results)):
        plot_histogram(ax, r["name"], r["denoised"], r["threshold"], show_legend=(k == 0))
    fig.suptitle("Histograms of the denoised images and selected thresholds")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "histograms.png", dpi=80)
    plt.close(fig)

def save_chain_code_grid(results: list, out_dir: Path):
    """All chain-code drawings in one figure (4 per row)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = make_grid(len(results), cell_width=4.5, cell_height=4.5)
    for k, (ax, r) in enumerate(zip(axes, results)):
        plot_chain_code(ax, r["name"], r["image"], r["boundary"], r["points"], r["code"],
                        show_legend=(k == 0))
    fig.suptitle("Traced boundaries and 32-segment chain codes")
    fig.tight_layout(rect=(0, 0, 1, 0.97))       # leave room for the title
    fig.savefig(out_dir / "chain_codes.png", dpi=120)   # higher dpi so the digits are readable
    plt.close(fig)

def plot_aligned(ax, name: str, mask: np.ndarray, boundary: np.ndarray, arc: np.ndarray,
                 invariant: list, show_legend: bool):
    """One panel: boundary after major-axis alignment, the axis, its two ends and the 32 cuts
    (first axis end, phase 0), titled with the canonical invariant code."""
    x_bar, y_bar, mu20, mu02, mu11 = central_moments(mask)
    aligned = align_boundary(boundary, x_bar, y_bar, principal_axis_angle(mu20, mu02, mu11))
    aligned = aligned / np.sqrt(mask.sum())                   # common scale for comparison
    ends = major_axis_ends(aligned)
    start = ends[0]
    cuts = aligned[subdivide_boundary(arc, 32, arc[start])]
    ax.plot(aligned[:, 1], aligned[:, 0], color="goldenrod", linewidth=1, label="aligned boundary")
    ax.plot(cuts[:, 1], cuts[:, 0], "-o", color="red", markersize=2, linewidth=1, label="32 segments")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="major axis")
    ax.plot(aligned[ends, 1], aligned[ends, 0], "s", color="tab:blue", markersize=6,
            label="axis ends (start points)")
    ax.plot(0, 0, "+", color="black", markersize=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()                                          # image convention: rows downwards
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(0.85, -0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{name}\n{''.join(map(str, invariant))}", fontsize=8, family="monospace")
    if show_legend:
        ax.legend(loc="lower right", fontsize=6)

def save_aligned(result: dict, out_dir: Path):
    """One image's aligned boundary as its own file: out_dir/<image name>.png"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_aligned(ax, result["name"], result["mask"], result["boundary"], result["arc"], result["invariant"],
                 show_legend=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(result['name']).stem}_aligned.png", dpi=100)
    plt.close(fig)

def save_distance_matrix(D: np.ndarray, results: list, groups: list, threshold: int, out_dir: Path):
    """Heat map of the chain-code distances; cells within the threshold are outlined."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [Path(r["name"]).stem for r in results]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(D, cmap="viridis_r")
    for i in range(len(D)):
        for j in range(len(D)):
            ax.text(j, i, str(D[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if D[i, j] > D.max() / 2 else "black")
            if i != j and D[i, j] <= threshold:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=1.5))
    ax.set_xticks(range(len(D)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(D)))
    ax.set_yticklabels(labels, fontsize=8)
    group_text = "   ".join(f"G{k}: " + ",".join(labels[i].replace("im-", "") for i in g)
                           for k, g in enumerate(groups, start=1))
    ax.set_title(f"Chain-code distances (red = within threshold {threshold})\n{group_text}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="distance")
    fig.tight_layout()
    fig.savefig(out_dir / "distance_matrix.png", dpi=100)
    plt.close(fig)

def save_aligned_grid(results: list, out_dir: Path):
    """All aligned boundaries in one figure (4 per row), same scale so shapes can be compared."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = make_grid(len(results), cell_width=4.0, cell_height=4.0)
    for k, (ax, r) in enumerate(zip(axes, results)):
        plot_aligned(ax, r["name"], r["mask"], r["boundary"], r["arc"], r["invariant"],
                     show_legend=(k == 0))
    fig.suptitle("Boundaries aligned to the major axis, with RST + start-point invariant codes")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "aligned.png", dpi=110)
    plt.close(fig)

# ========================================= Main Function =========================================

def main():
    """Main function: load images (grayscale) -> denoise -> segment -> chain code -> RST invariant code
                     -> group by chain-code distance -> write output file (+ debug images if requested). """
    parser = argparse.ArgumentParser(description="Group images of similar objects using RST-invariant chain codes.")
    parser.add_argument("input_folder", help="folder containing the .png / .jpg images")
    parser.add_argument("--output", default="output.txt", help="output text file")
    parser.add_argument("--debug", action="store_true", help="save intermediate images to ./debug")
    parser.add_argument("--threshold", type=int, default=6,
                        help="max chain-code distance for two images to be grouped (default 6)")
    args = parser.parse_args()

    images = load_images(args.input_folder)
    print(f"Loaded {len(images)} images from {args.input_folder}")

    results = []
    for name, image in images:
        denoised = reduce_noise(image)
        mask, binary, threshold = segment_object(denoised)
        code, boundary, points, arc = boundary_chain_code(mask, segments=32)
        candidates = invariant_chain_codes(mask, boundary, arc, segments=32, phases=4)
        invariant = canonical_code(candidates)
        print(f"{name}: T = {threshold:.0f}, area = {int(mask.sum())} px, boundary = {len(boundary)} px")
        print(f"    chain code     = {''.join(map(str, code))}")
        print(f"    invariant code = {''.join(map(str, invariant))}")
        result = {"name": name, "image": image, "denoised": denoised, "threshold": threshold,
                  "mask": mask, "boundary": boundary, "arc": arc, "points": points, "code": code,
                  "candidates": candidates, "invariant": invariant}
        results.append(result)
        if args.debug:
            save_stages(name, [("Original", image), ("Denoised", denoised),
                               (f"Thresholded (T={threshold:.0f})", binary), ("Object mask", mask)],
                        Path("debug") / "stages")                    # debug/stages/<name>_stages.png
            save_histogram(result, Path("debug") / "histogram")      # debug/histogram/<name>_histogram.png
            save_chain_code(result, Path("debug") / "chain_code")    # debug/chain_code/<name>_chain_code.png
            save_aligned(result, Path("debug") / "aligned")          # debug/aligned/<name>_aligned.png

    # Grouping: compare every pair of images by their invariant candidate codes
    D = distance_matrix([r["candidates"] for r in results])
    groups = group_images(D, args.threshold)
    write_output(Path(args.output), groups, results, args.threshold)
    print(f"\nGroups (threshold = {args.threshold}):")
    for k, group in enumerate(groups, start=1):
        print(f"    Group {k}: {', '.join(results[i]['name'] for i in group)}")
    print(f"Output written to {args.output}")

    if args.debug:
        save_histogram_grid(results, Path("debug"))                  # debug/histograms.png
        save_chain_code_grid(results, Path("debug"))                 # debug/chain_codes.png
        save_aligned_grid(results, Path("debug"))                    # debug/aligned.png
        save_distance_matrix(D, results, groups, args.threshold, Path("debug"))   # debug/distance_matrix.png

if __name__ == "__main__":
    main()
