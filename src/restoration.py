import cv2
import numpy as np
from typing import Tuple, Union

def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    """
    Validates that input is a non-empty 3D RGB NumPy array of dtype uint8.
    Raises TypeError or ValueError if validation fails.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Input image must be 3-channel RGB with shape (H, W, 3). Got shape: {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Input image dimensions must be non-zero")
    if image.dtype != np.uint8:
        raise ValueError(f"Input image must be of dtype uint8. Got: {image.dtype}")
    return image.copy()

def median_denoise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Applies Median Filtering for noise reduction (especially salt-and-pepper noise).
    
    Args:
        image: Input RGB image array (H, W, 3) of dtype uint8.
        ksize: Aperture linear size; must be an odd positive integer >= 1.
        
    Returns:
        Denoised RGB image array of dtype uint8.
    """
    img = _validate_rgb_image(image)
    if not isinstance(ksize, int) or ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be an odd positive integer. Got: {ksize}")
    
    return cv2.medianBlur(img, ksize)

def gaussian_denoise(image: np.ndarray, ksize: int = 3, sigma: float = 0.0) -> np.ndarray:
    """
    Applies Gaussian Filtering to smooth image and reduce high-frequency Gaussian noise.
    
    Args:
        image: Input RGB image array (H, W, 3) of dtype uint8.
        ksize: Kernel size; must be an odd positive integer >= 1.
        sigma: Gaussian kernel standard deviation. If 0.0, calculated from ksize.
        
    Returns:
        Denoised RGB image array of dtype uint8.
    """
    img = _validate_rgb_image(image)
    if not isinstance(ksize, int) or ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be an odd positive integer. Got: {ksize}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative. Got: {sigma}")
    
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))

def bilateral_denoise(image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """
    Applies Bilateral Filtering to reduce noise while preserving sharp edges.
    
    Args:
        image: Input RGB image array (H, W, 3) of dtype uint8.
        d: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space. Larger values blend larger color differences.
        sigma_space: Filter sigma in the coordinate space. Larger values blend further pixels.
        
    Returns:
        Denoised RGB image array of dtype uint8.
    """
    img = _validate_rgb_image(image)
    if not isinstance(d, int) or d < 1:
        raise ValueError(f"d (diameter) must be a positive integer. Got: {d}")
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError(f"sigma_color and sigma_space must be positive floats. Got: sigma_color={sigma_color}, sigma_space={sigma_space}")
    
    return cv2.bilateralFilter(img, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Performs Global Histogram Equalization on the Luminance channel in LAB color space
    to enhance overall image contrast without causing unnatural color shifts.
    
    Args:
        image: Input RGB image array (H, W, 3) of dtype uint8.
        
    Returns:
        Contrast-enhanced RGB image array of dtype uint8.
    """
    img = _validate_rgb_image(image)
    
    # Convert RGB to LAB space to decouple lightness (L) from color information (A, B)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Equalize histogram on the L channel
    l_equalized = cv2.equalizeHist(l_channel)
    
    # Merge channels back and convert back to RGB
    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)

def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel in LAB color space.
    Enhances local contrast while avoiding over-amplification of noise.
    
    Args:
        image: Input RGB image array (H, W, 3) of dtype uint8.
        clip_limit: Threshold for contrast limiting. Higher values increase contrast.
        tile_grid_size: Size of grid for histogram equalization (height, width).
        
    Returns:
        Locally contrast-enhanced RGB image array of dtype uint8.
    """
    img = _validate_rgb_image(image)
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be a positive float. Got: {clip_limit}")
    if not isinstance(tile_grid_size, tuple) or len(tile_grid_size) != 2 or tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
        raise ValueError(f"tile_grid_size must be a tuple of 2 positive integers. Got: {tile_grid_size}")
    
    # Convert to LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)
    
    # Merge and convert back to RGB
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def unsharp_mask(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
    """
    Applies Unsharp Masking to sharpen image details by subtracting a blurred version from the original.
    Formula: sharpened = original + amount * (original - blurred)
    
    Args:
        image: Input RGB image array (H, W, 3) of dtype uint8.
        kernel_size: Tuple (width, height) of odd integers for Gaussian blur.
        sigma: Standard deviation for Gaussian blur.
        amount: Sharpening multiplier. Higher values create stronger sharpening.
        threshold: Minimum pixel difference required between image and blurred version to apply sharpening.
        
    Returns:
        Sharpened RGB image array of dtype uint8.
    """
    img = _validate_rgb_image(image)
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2 or kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0 or kernel_size[0] < 1 or kernel_size[1] < 1:
        raise ValueError(f"kernel_size must be a tuple of two positive odd integers. Got: {kernel_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be a positive float. Got: {sigma}")
    if amount < 0:
        raise ValueError(f"amount must be a non-negative float. Got: {amount}")
    if threshold < 0:
        raise ValueError(f"threshold must be a non-negative integer. Got: {threshold}")

    blurred = cv2.GaussianBlur(img, kernel_size, sigmaX=float(sigma), sigmaY=float(sigma))
    
    img_float = img.astype(np.float64)
    blurred_float = blurred.astype(np.float64)
    diff = img_float - blurred_float

    if threshold > 0:
        mask = np.abs(diff) >= threshold
        sharpened = img_float + amount * diff * mask
    else:
        sharpened = img_float + amount * diff
        
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def laplacian_sharpen(image: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Applies Laplacian Sharpening by extracting second-order derivative edges and adding them back to the image.
    Formula: sharpened = original - alpha * Laplacian(original)
    
    Args:
        image: Input RGB image array (H, W, 3) of dtype uint8.
        alpha: Weight of the Laplacian operator added to the image.
        
    Returns:
        Sharpened RGB image array of dtype uint8.
    """
    img = _validate_rgb_image(image)
    if alpha < 0:
        raise ValueError(f"alpha must be a non-negative float. Got: {alpha}")

    # Calculate Laplacian derivative across all channels
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # Subtract Laplacian (since central peak of standard 3x3 Laplacian mask is negative/positive depending on sign convention)
    sharpened = img.astype(np.float64) - alpha * laplacian
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)
