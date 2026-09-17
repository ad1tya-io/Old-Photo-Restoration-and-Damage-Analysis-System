import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

@dataclass
class DegradationReport:
    """
    Structured report containing quantitative metrics and heuristic indicators
    describing the visual degradation present in an analyzed photograph.
    """
    noise_score: float
    blur_score: float
    contrast_std: float
    dynamic_range: int
    mean_brightness: float
    artifact_ratio: float
    artifact_count: int
    image_dimensions: Tuple[int, int, int]
    is_noisy: bool
    is_blurry: bool
    is_low_contrast: bool
    has_artifacts: bool
    artifact_mask: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Converts report metrics to a clean dictionary (excluding heavy mask array)."""
        return {
            "noise_score": round(float(self.noise_score), 4),
            "blur_score": round(float(self.blur_score), 4),
            "contrast_std": round(float(self.contrast_std), 4),
            "dynamic_range": int(self.dynamic_range),
            "mean_brightness": round(float(self.mean_brightness), 4),
            "artifact_ratio": round(float(self.artifact_ratio), 6),
            "artifact_count": int(self.artifact_count),
            "image_dimensions": self.image_dimensions,
            "is_noisy": self.is_noisy,
            "is_blurry": self.is_blurry,
            "is_low_contrast": self.is_low_contrast,
            "has_artifacts": self.has_artifacts,
        }

def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    """
    Validates that input is a non-empty 3-channel RGB NumPy array of dtype uint8.
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
    return image

def estimate_noise(gray: np.ndarray) -> float:
    """
    Estimates standard deviation of Gaussian noise in an image using Immerkær's fast method.
    Kernel M = [[ 1, -2,  1],
                [-2,  4, -2],
                [ 1, -2,  1]]
    Formula: sigma = sqrt(pi / 2) * (1 / (6 * (W-2) * (H-2))) * sum(|I * M|)
    
    Args:
        gray: 2D grayscale NumPy array of dtype uint8 or float.
        
    Returns:
        Estimated noise standard deviation (higher score = more noise).
    """
    H, W = gray.shape
    if H < 3 or W < 3:
        return 0.0
    
    # 3x3 Laplacian-like kernel for noise estimation
    M = np.array([[1, -2, 1],
                  [-2, 4, -2],
                  [1, -2, 1]], dtype=np.float64)
    
    # Convolve with kernel
    conv = cv2.filter2D(gray.astype(np.float64), -1, M)
    
    # Compute sum of absolute values
    sigma = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6.0 * (W - 2) * (H - 2))
    return float(sigma)

def estimate_blur(gray: np.ndarray) -> float:
    """
    Estimates image blur using the variance of the 2D Laplacian.
    Sharp images contain clear high-frequency edges, producing a high Laplacian variance.
    Blurred images smooth out edges, yielding a lower Laplacian variance.
    
    Args:
        gray: 2D grayscale NumPy array.
        
    Returns:
        Variance of Laplacian (lower score = more blurry).
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())

def estimate_contrast(gray: np.ndarray) -> Tuple[float, int]:
    """
    Calculates contrast statistics:
    - Standard deviation of pixel intensities
    - Dynamic range (max intensity - min intensity)
    
    Returns:
        Tuple of (contrast_std, dynamic_range)
    """
    contrast_std = float(np.std(gray))
    dynamic_range = int(np.max(gray) - np.min(gray))
    return contrast_std, dynamic_range

def estimate_brightness(gray: np.ndarray) -> float:
    """
    Calculates mean brightness across grayscale intensity values [0, 255].
    """
    return float(np.mean(gray))

def detect_artifacts(gray: np.ndarray, 
                     kernel_size: int = 5, 
                     threshold_val: int = 30) -> Tuple[np.ndarray, float, int]:
    """
    Detects potential scratches, dust spots, and thin linear/isolated artifacts 
    using morphological Top-Hat & Black-Hat transformations.
    
    - Top-Hat isolates elements brighter than their surroundings (bright scratches/dust).
    - Black-Hat isolates elements darker than their surroundings (dark scratches/dust).
    
    Args:
        gray: 2D grayscale NumPy array of dtype uint8.
        kernel_size: Size of morphological structuring element.
        threshold_val: Difference threshold for binary artifact segmentation.
        
    Returns:
        Tuple of (artifact_mask, artifact_ratio, artifact_count)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Morphological Top-Hat and Black-Hat
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Combine scratch/dust candidate responses
    combined = cv2.add(top_hat, black_hat)
    
    # Threshold to create binary mask
    _, mask = cv2.threshold(combined, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Clean noise from mask using small opening
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
    
    # Calculate statistics
    total_pixels = mask.size
    artifact_pixels = int(np.count_nonzero(mask))
    artifact_ratio = float(artifact_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    # Count connected artifact components
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask)
    artifact_count = max(0, num_labels - 1)  # Subtract background component
    
    return mask, artifact_ratio, artifact_count


class DegradationAnalyzer:
    """
    Analyzer class for evaluating degradation characteristics of an image.
    
    Configurable Thresholds (Heuristics):
        noise_threshold: Immerkær noise score above which an image is considered noisy (default: 3.5).
        blur_threshold: Laplacian variance below which an image is considered blurry (default: 100.0).
        contrast_threshold: Intensity standard deviation below which contrast is considered low (default: 45.0).
        artifact_ratio_threshold: Artifact pixel ratio above which artifacts are detected (default: 0.005).
    """
    def __init__(self,
                 noise_threshold: float = 3.5,
                 blur_threshold: float = 100.0,
                 contrast_threshold: float = 45.0,
                 artifact_ratio_threshold: float = 0.005):
        self.noise_threshold = noise_threshold
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold
        self.artifact_ratio_threshold = artifact_ratio_threshold

    def analyze(self, image: np.ndarray) -> DegradationReport:
        """
        Analyzes an RGB image array and returns a structured DegradationReport.
        
        Args:
            image: 3-channel RGB image array (H, W, 3) of dtype uint8.
            
        Returns:
            DegradationReport instance containing metrics and heuristic flags.
        """
        _validate_rgb_image(image)
        
        # Convert RGB to Grayscale for classical intensity analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate degradation metrics
        noise_score = estimate_noise(gray)
        blur_score = estimate_blur(gray)
        contrast_std, dynamic_range = estimate_contrast(gray)
        mean_brightness = estimate_brightness(gray)
        artifact_mask, artifact_ratio, artifact_count = detect_artifacts(gray)
        
        # Evaluate heuristic indicator flags based on thresholds
        is_noisy = noise_score >= self.noise_threshold
        is_blurry = blur_score < self.blur_threshold
        is_low_contrast = contrast_std < self.contrast_threshold
        has_artifacts = artifact_ratio >= self.artifact_ratio_threshold
        
        return DegradationReport(
            noise_score=noise_score,
            blur_score=blur_score,
            contrast_std=contrast_std,
            dynamic_range=dynamic_range,
            mean_brightness=mean_brightness,
            artifact_ratio=artifact_ratio,
            artifact_count=artifact_count,
            image_dimensions=image.shape,
            is_noisy=is_noisy,
            is_blurry=is_blurry,
            is_low_contrast=is_low_contrast,
            has_artifacts=has_artifacts,
            artifact_mask=artifact_mask
        )
