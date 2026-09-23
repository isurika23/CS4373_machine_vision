import cv2
import numpy as np
from scipy import ndimage
from pathlib import Path

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    # Normalize so sum = 1
    return kernel / np.sum(kernel)

def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    kernel = gaussian_kernel(kernel_size, sigma)
    return ndimage.convolve(image, kernel, mode='constant')

def sobel_filter(image: np.ndarray) -> (np.ndarray, np.ndarray):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = ndimage.convolve(image, sobel_x)
    gradient_y = ndimage.convolve(image, sobel_y)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_angle = np.arctan2(gradient_y, gradient_x)
    
    return gradient_magnitude, gradient_angle

def non_maximum_suppression(gradient_magnitude: np.ndarray, gradient_angle: np.ndarray) -> np.ndarray:
    M, N = gradient_magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            angle = gradient_angle[i, j] * 180. / np.pi
            angle = (angle + 180) % 180  # Normalize angle to [0, 180)

            # Determine the two neighboring pixels to compare
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
            elif (22.5 <= angle < 67.5):
                neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
            elif (67.5 <= angle < 112.5):
                neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
            else:  # (112.5 <= angle < 157.5)
                neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]

            # Suppress non-maximum pixels
            if gradient_magnitude[i, j] >= max(neighbors):
                Z[i, j] = gradient_magnitude[i, j]
            else:
                Z[i, j] = 0

    return Z

def double_threshold(image: np.ndarray, low_threshold: float, high_threshold: float) -> (np.ndarray, float, float):
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.float32)

    strong = 255
    weak = 75

    # Get coordinates of strong and weak edges
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    # Set values
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result, weak, strong

def hysteresis(image: np.ndarray, weak: float = 75, strong: float = 255) -> np.ndarray:
    M, N = image.shape

    # Make a copy
    output = image.copy()

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if output[i, j] == weak:
                # Check 8-connected neighbors
                if ((output[i + 1, j - 1] == strong) or (output[i + 1, j] == strong) or
                    (output[i + 1, j + 1] == strong) or (output[i, j - 1] == strong) or
                    (output[i, j + 1] == strong) or (output[i - 1, j - 1] == strong) or
                    (output[i - 1, j] == strong) or (output[i - 1, j + 1] == strong)):
                    output[i, j] = strong
                else:
                    output[i, j] = 0

    return output

def canny(image: np.ndarray, low_ratio: float = 0.05, high_ratio: float = 0.15):
    # Step1: convert image to grayscale - already done in main function

    # Step2: Apply Gaussian filter to smooth the image
    blurred_image = gaussian_filter(image, kernel_size=5, sigma=1)

    # Step3: Compute gradients
    gradient_magnitude, gradient_angle = sobel_filter(blurred_image)

    # Step4: Non-maximum suppression
    suppressed_image = non_maximum_suppression(gradient_magnitude, gradient_angle)

    # Step5: Double thresholding
    high_threshold = suppressed_image.max() * high_ratio
    low_threshold = high_threshold * low_ratio
    thresholded_image, weak, strong = double_threshold(suppressed_image, low_threshold, high_threshold)

    # Step6: Edge tracking by hysteresis
    edge_image = hysteresis(thresholded_image, weak=weak, strong=strong)

    return edge_image

if __name__ == "__main__":

    dir = Path(__file__).resolve().parent/"images"/"puppy.jpg"
    image_path = str(dir)

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    data_int_image = image.astype(np.float32)
    canny_image = canny(data_int_image)

    # Save the edge image into ../edges/ relative to this script
    edges_dir = Path(__file__).resolve().parent.parent / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(edges_dir / "butterfly_edges.png")
    cv2.imwrite(output_path, canny_image)

    # Display the original image and the edges
    cv2.imshow("Original Image", image)
    cv2.imshow("Canny Edges", canny_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()