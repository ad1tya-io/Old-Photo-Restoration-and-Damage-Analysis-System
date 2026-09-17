import cv2
import numpy as np
from typing import Tuple, Union

def _validate_image(image: np.ndarray) -> np.ndarray:
    """
    Validates input image array. Accepts 3-channel RGB (H, W, 3) or 2-channel Grayscale (H, W).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy.ndarray")
    if image.ndim not in [2, 3]:
        raise ValueError(f"Input image must be 2D grayscale or 3D RGB. Got shape: {image.shape}")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(f"Input 3D image must have 3 channels (RGB). Got shape: {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Input image dimensions must be non-zero")
    if image.dtype != np.uint8:
        raise ValueError(f"Input image must be of dtype uint8. Got: {image.dtype}")
    return image

def compute_fft(image: np.ndarray) -> np.ndarray:
    """
    Computes 2D Discrete Fourier Transform (DFT) and returns shifted magnitude spectrum.
    
    Computer Vision Context:
    - Spatial Domain: Pixels represent intensity values at spatial locations (x, y).
    - Frequency Domain: Frequencies represent rates of intensity change across spatial distance.
      High frequencies represent edges, noise, and sharp details. Low frequencies represent smooth regions.
    - Frequency Shifting: Shifts DC component (zero frequency) from top-left (0, 0) to center (H/2, W/2).
    - Log Magnitude: S = log(1 + |F_shifted|) compresses wide dynamic range of Fourier coefficients for visual rendering.
    
    Args:
        image: 3D RGB array (H, W, 3) or 2D Grayscale array (H, W).
        
    Returns:
        2D Float NumPy array representing log-magnitude spectrum.
    """
    img = _validate_image(image)
    
    # Extract luminance channel if RGB
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
        
    # Perform 2D Fast Fourier Transform
    fft_coeff = np.fft.fft2(gray.astype(np.float64))
    
    # Shift zero-frequency component to center of spectrum
    fft_shifted = np.fft.fftshift(fft_coeff)
    
    # Compute log magnitude spectrum: log(1 + |F|)
    magnitude_spectrum = np.log(1.0 + np.abs(fft_shifted))
    return magnitude_spectrum

def create_low_pass_mask(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Creates an ideal circular low-pass frequency domain mask.
    
    Computer Vision Context:
    - Low-pass filter allows frequencies inside cutoff radius D0 to pass (value = 1.0)
      and blocks high frequencies outside D0 (value = 0.0).
    - Low-pass filtering smooths images and suppresses high-frequency noise.
    
    Args:
        shape: Tuple of (height, width).
        cutoff: Cutoff radius in pixels (> 0).
        
    Returns:
        2D Float NumPy mask array of shape (H, W) with values in {0.0, 1.0}.
    """
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"shape must be a tuple of 2 positive integers. Got: {shape}")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be a positive number. Got: {cutoff}")
        
    H, W = shape
    cy, cx = H // 2, W // 2
    
    # Generate grid coordinates relative to frequency center
    y, x = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Circular binary mask
    mask = np.where(dist_from_center <= cutoff, 1.0, 0.0).astype(np.float64)
    return mask

def create_high_pass_mask(shape: Tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Creates an ideal circular high-pass frequency domain mask.
    
    Computer Vision Context:
    - High-pass filter blocks frequencies inside cutoff radius D0 (value = 0.0)
      and passes high-frequency edge details outside D0 (value = 1.0).
    
    Args:
        shape: Tuple of (height, width).
        cutoff: Cutoff radius in pixels (> 0).
        
    Returns:
        2D Float NumPy mask array of shape (H, W) with values in {0.0, 1.0}.
    """
    return 1.0 - create_low_pass_mask(shape, cutoff)

def apply_frequency_filter(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a frequency-domain mask to an image and reconstructs spatial domain output.
    
    Luminance-Based Design Decision:
    Filtering R, G, B color channels independently in frequency domain introduces phase/amplitude
    mismatches between color components, creating severe chromatic artifacts/fringing.
    We convert RGB -> YCrCb, apply frequency filtering to the Y (Luminance) channel only, 
    and recombine with original Cr, Cb (Chrominance) channels before converting back to RGB.
    
    Args:
        image: Input RGB array (H, W, 3) or 2D Grayscale (H, W) of dtype uint8.
        mask: 2D frequency mask array matching spatial shape (H, W).
        
    Returns:
        Filtered spatial domain array of same shape and dtype uint8.
    """
    img = _validate_image(image)
    H, W = img.shape[:2]
    
    if not isinstance(mask, np.ndarray) or mask.shape != (H, W):
        raise ValueError(f"Mask shape {getattr(mask, 'shape', None)} must match image spatial dimensions {(H, W)}")
        
    is_rgb = (img.ndim == 3)
    
    if is_rgb:
        # Convert RGB to YCrCb space
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        target_channel = y_channel
    else:
        target_channel = img.copy()

    # 1. 2D FFT
    fft_coeff = np.fft.fft2(target_channel.astype(np.float64))
    
    # 2. Shift zero-frequency to center
    fft_shifted = np.fft.fftshift(fft_coeff)
    
    # 3. Apply frequency mask
    filtered_fft_shifted = fft_shifted * mask
    
    # 4. Inverse shift
    filtered_fft = np.fft.ifftshift(filtered_fft_shifted)
    
    # 5. Inverse 2D FFT (reconstruct spatial domain)
    reconstructed_complex = np.fft.ifft2(filtered_fft)
    
    # Take real part and clip intensity values to [0, 255]
    reconstructed_real = np.real(reconstructed_complex)
    reconstructed_channel = np.clip(reconstructed_real, 0, 255).astype(np.uint8)
    
    if is_rgb:
        # Recombine filtered Luminance (Y) with original Chrominance (Cr, Cb)
        filtered_ycrcb = cv2.merge((reconstructed_channel, cr_channel, cb_channel))
        return cv2.cvtColor(filtered_ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        return reconstructed_channel

def frequency_denoise(image: np.ndarray, cutoff: float = 40.0) -> np.ndarray:
    """
    Denoises image by applying a low-pass Fourier filter to attenuate high-frequency noise.
    
    Args:
        image: Input RGB array (H, W, 3) or 2D Grayscale (H, W).
        cutoff: Low-pass frequency cutoff radius (> 0).
        
    Returns:
        Low-pass filtered uint8 image array.
    """
    img = _validate_image(image)
    shape = img.shape[:2]
    mask = create_low_pass_mask(shape, cutoff=cutoff)
    return apply_frequency_filter(img, mask)

def frequency_sharpen(image: np.ndarray, cutoff: float = 30.0, alpha: float = 0.5) -> np.ndarray:
    """
    Sharpens image details by enhancing high-frequency components in the frequency domain.
    Formula: Y_sharpened = Y_original + alpha * HighPassFilter(Y_original)
    
    Args:
        image: Input RGB array (H, W, 3) or 2D Grayscale (H, W).
        cutoff: High-pass frequency cutoff radius (> 0).
        alpha: Sharpening boost factor (>= 0).
        
    Returns:
        High-pass sharpened uint8 image array.
    """
    img = _validate_image(image)
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative. Got: {alpha}")
        
    shape = img.shape[:2]
    hp_mask = create_high_pass_mask(shape, cutoff=cutoff)
    
    is_rgb = (img.ndim == 3)
    if is_rgb:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        
        # High-pass filter the Y channel
        y_high_pass = apply_frequency_filter(y_channel, hp_mask).astype(np.float64)
        
        # Add high-frequency component to original Y channel
        y_sharpened = y_channel.astype(np.float64) + alpha * y_high_pass
        y_sharpened = np.clip(y_sharpened, 0, 255).astype(np.uint8)
        
        filtered_ycrcb = cv2.merge((y_sharpened, cr_channel, cb_channel))
        return cv2.cvtColor(filtered_ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        high_pass = apply_frequency_filter(img, hp_mask).astype(np.float64)
        sharpened = img.astype(np.float64) + alpha * high_pass
        return np.clip(sharpened, 0, 255).astype(np.uint8)
