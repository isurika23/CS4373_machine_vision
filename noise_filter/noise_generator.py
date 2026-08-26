"""Generate Gaussian, salt-and-pepper, and structural noise images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def add_gaussian_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise and keep pixels in the valid range."""
    noise = rng.normal(0, 25, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
	image: np.ndarray, rng: np.random.Generator, amount: float = 0.05
) -> np.ndarray:
    """Randomly replace pixels with black or white pixels."""
    noisy = image.copy()
    pixel_count = image.shape[0] * image.shape[1]
    affected = rng.random(pixel_count) < amount
    salt = rng.random(pixel_count) < 0.5
    pixels = noisy.reshape(pixel_count, -1)
    pixels[affected] = np.where(salt[affected, None], 255, 0)
    return noisy


def add_structural_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a repeated sinusoidal pattern, also called periodic noise."""
    height, width = image.shape[:2]
    frequency = rng.uniform(0.04, 0.09)
    phase = rng.uniform(0, 2 * np.pi)
    strength = rng.uniform(25, 45)
    x = np.arange(width, dtype=np.float32)
    pattern = strength * np.sin(2 * np.pi * frequency * x + phase)
    pattern = np.broadcast_to(pattern, (height, width))
    if image.ndim == 3:
        pattern = pattern[..., None]
    return np.clip(image.astype(np.float32) + pattern, 0, 255).astype(np.uint8)


def generate_noised_images(
	source_dir: Path, destination_dir: Path, seed: int | None = None
) -> int:
    """Generate all three noise variants for every supported source image."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    image_paths = sorted(
        path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        with Image.open(image_path) as source:
            image = np.array(source)

        variants = {
            "gau": add_gaussian_noise(image, rng),
            "snp": add_salt_and_pepper_noise(image, rng),
            "str": add_structural_noise(image, rng),
        }
        for suffix, noisy_image in variants.items():
            output_path = destination_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
            Image.fromarray(noisy_image).save(output_path)

    return len(image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parent / "images" / "original",
        help="Directory containing the original images.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).parent / "images" / "noised",
        help="Directory in which to save generated images.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for repeatable noise.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    count = generate_noised_images(arguments.source, arguments.destination, arguments.seed)
    print(f"Generated {count * 3} noisy images from {count} source images.")
