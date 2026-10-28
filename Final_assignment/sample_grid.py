"""
    Save all sample images in one figure (4 columns x 3 rows) for the report.
    python sample_grid.py <input_folder> [--output samples_grid.png]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import load_images        # same loader as the pipeline (OpenCV used only to read files)


def save_sample_grid(images: list, output: Path, columns: int = 4, rows: int = 3):
    """ Draw the images in a columns x rows grid, each titled with its file name """
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows))
    for ax, (name, image) in zip(axes.ravel(), images):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=12)
    for ax in axes.ravel():
        ax.axis("off")                  # also hides empty cells if there are fewer images
    fig.tight_layout()
    fig.savefig(output, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Show all sample images in one 4x3 grid.")
    parser.add_argument("input_folder", help="folder containing the .png / .jpg images")
    parser.add_argument("--output", default="samples_grid.png", help="output image file")
    args = parser.parse_args()

    images = load_images(args.input_folder)
    save_sample_grid(images[:12], Path(args.output))
    print(f"Saved {min(len(images), 12)} images to {args.output}")


if __name__ == "__main__":
    main()
