# Machine Vision Assignments (In22-S7-CS4373)

Lab assignments for the Machine Vision module, by Gunasekara T I U (220192J). Every assignment implements the image-processing operations **from scratch with NumPy**, following the lecture notes; OpenCV is used mainly to read and write image files.

| # | Folder | Topic | Lecture unit | Main file |
|---|---|---|---|---|
| 1 | [`Assignment_2/`](Assignment_2) | Image resampling with bilinear interpolation | Unit 3, Input geometry | `code/resample.py` |
| 2 | [`noise_filter/`](noise_filter) | Noise generation and noise filtering | Unit 4, Noise filtering | `noise_filter.py`, `noise_generator.py` |
| 3 | [`canny_edge_detection/`](canny_edge_detection) | Canny edge detector | Unit 5, Gradients and edges | `canny.py` |
| 4 | [`Final_assignment/`](Final_assignment) | Grouping objects by RST-invariant chain codes | Units 4, 6, 7 | `pipeline.py` |

---

## Setup

```bash
python -m venv mvenv
mvenv\Scripts\activate            # Windows  (Linux/macOS: source mvenv/bin/activate)
pip install numpy opencv-python matplotlib scipy pillow
```

| Package | Used by |
|---|---|
| numpy | all assignments |
| opencv-python | image reading / writing (all), display windows (Canny) |
| matplotlib | figures and result plots |
| scipy | `ndimage.convolve` in the Canny assignment |
| pillow | reading and saving images in `noise_generator.py` |

Each script builds its paths from its own location, so it can be run from any folder.

---

## 1. Image resampling (`Assignment_2/`)

Resizes a grayscale image with **bilinear interpolation** implemented pixel by pixel, then measures how much detail is lost by scaling down and back up.

```bash
python Assignment_2/code/resample.py
```

- **`resample_image(image, scale)`**: for every output pixel, maps its centre back to the source with the centre-aligned formula `(i + 0.5) / scale − 0.5`, takes the four surrounding source pixels and blends them by their distances: `(1−a)(1−b)·p11 + a(1−b)·p21 + (1−a)b·p12 + ab·p22`.
- **`calculate_mse(a, b)`**: mean squared error between the original and the restored image.
- **Experiment:** `sample_images/test_bnw3.jpg` is scaled by 0.7, then by 1/0.7 back to its original size. Both results are saved to `output_images/` (`downscaled.jpg`, `upscaled.jpg`), the MSE is printed, and the three images are shown side by side.

## 2. Noise generation and filtering (`noise_filter/`)

Adds three kinds of noise to test images, then compares three spatial filters on each.

```bash
python noise_filter/noise_generator.py [--seed 42]   # images/original → images/noised
python noise_filter/noise_filter.py                  # images/noised   → images/filtered
```

**Noise models (`noise_generator.py`):**

| Suffix | Noise | How it is made |
|---|---|---|
| `_gau` | Gaussian | zero-mean normal noise, σ = 25 grey levels |
| `_snp` | Salt and pepper (outliers) | 5% of pixels set to 0 or 255 |
| `_str` | Structural (periodic) | a horizontal sine pattern with random frequency, phase and strength |

**Filters (`noise_filter.py`, 3×3 window, cyclic padding):**

| Filter | Output pixel | Works best on |
|---|---|---|
| `mean_filter()` | average of the window | Gaussian noise |
| `median_filter()` | median of the window | salt-and-pepper noise (outliers never become the median) |
| `mid_point_filter()` | (min + max) / 2 of the window | uniform-like noise; poor on outliers |

Three test images (`butterfly`, `eye`, `wave`) × 3 noise types × 3 filters give 27 filtered images in `images/filtered/`, named `<image>_<noise>_<filter>.jpg`. Each case is also shown as a 2×2 figure (noisy, mean, median, midpoint).

## 3. Canny edge detection (`canny_edge_detection/`)

The classic five-step Canny detector.

```bash
python canny_edge_detection/canny.py
```

| Step | Function | What it does |
|---|---|---|
| 1 | `gaussian_filter()` | smooths the image with a 5×5 Gaussian (σ = 1) to suppress noise |
| 2 | `sobel_filter()` | Sobel x / y gradients → magnitude √(Gx² + Gy²) and direction atan2(Gy, Gx) |
| 3 | `non_maximum_suppression()` | keeps a pixel only if it is the maximum along its gradient direction (rounded to 0°, 45°, 90°, 135°), thinning edges to one pixel |
| 4 | `double_threshold()` | high = 0.15 × max, low = 0.05 × high: strong (255), weak (75) or discarded |
| 5 | `hysteresis()` | a weak pixel becomes an edge only if one of its 8 neighbours is strong |

The input is read from `images/`, and the edge map is saved to `canny_edge_detection/edges/<image>_edges.png` (e.g. `puppy_edges.png`) and shown next to the original. Convolution uses `scipy.ndimage.convolve`.

## 4. Final assignment: RST-invariant shape grouping (`Final_assignment/`)

A complete pipeline that groups images of the same object even when it is rotated, scaled or moved. **Here OpenCV is used only for loading; everything else is from scratch.**

```bash
cd Final_assignment
python pipeline.py Samples [--threshold 6] [--debug]   # writes output.txt
python sample_grid.py Samples                          # sample overview figure
```

| Stage | Operations |
|---|---|
| 1. Noise reduction | median filter 5×5 (outliers) → separable Gaussian σ = 3 |
| 2. Segmentation | smoothed histogram → valley threshold (inter-means fallback) → binary clean-up: median 7×7, largest connected component, hole filling |
| 3. Boundary chain code | contour tracking → 32 equal-length segments (√2 perimeter rule) → 8-direction Freeman code |
| 4. RST invariance | moments → major-axis alignment → axis-end start points → first difference → smallest circular shift |
| 5. Grouping | circular chain-code distance → union-find grouping with a threshold → `output.txt` |

On the 12 samples it finds 7 groups: the asteroid in all its rotated and scaled versions (im-1, 3, 4, 5, 6, 7), its mirror image (im-2), and five different blobs (im-8 … im-12). The sample images come from `Samples.zip`; unzip it into `Final_assignment/Samples/` before running.

Full documentation: [`Final_assignment/doc/README.md`](Final_assignment/doc/README.md); report: [`Final_assignment/doc/Report.pdf`](Final_assignment/doc/Report.pdf).

---

## Folder layout

```
Assignments/
├── README.md                    ← this file
├── Assignment_2/                ← 1. bilinear resampling
│   ├── code/resample.py
│   ├── sample_images/
│   └── output_images/
├── noise_filter/                ← 2. noise generation and filtering
│   ├── noise_generator.py
│   ├── noise_filter.py
│   └── images/{original, noised, filtered}/
├── canny_edge_detection/        ← 3. Canny edge detector
│   ├── canny.py
│   ├── images/
│   └── edges/                   ← output of canny.py (<image>_edges.png)
└── Final_assignment/            ← 4. chain-code shape grouping
    ├── pipeline.py, sample_grid.py, output.txt
    ├── Samples/  (from Samples.zip)
    ├── debug/    (created with --debug)
    └── doc/      (README.md, Report.pdf, pipeline_diagram.drawio, Task.txt)
```

## Lecture topics covered

| Lecture unit | Assignment |
|---|---|
| Unit 3: Input geometry (resampling, interpolation) | 1 |
| Unit 4: Noise and noise filtering | 2, 4 |
| Unit 5: Gradients and edges | 3 |
| Unit 6: Segmentation (thresholding, region growing, contour tracking) | 4 |
| Unit 7: Representation (chain codes, normalisation, moments) | 4 |
