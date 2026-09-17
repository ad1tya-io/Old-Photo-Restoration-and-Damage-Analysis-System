import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from typing import Dict

def _check_dimensions(img1: np.ndarray, img2: np.ndarray):
    """
    Ensures two images have the exact same dimensions and channels.
    Raises ValueError if they mismatch.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image dimensions mismatch. Image 1: {img1.shape}, Image 2: {img2.shape}. "
                         f"Metrics cannot be calculated on images of different shapes.")

def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculates Mean Squared Error (MSE) between two images.
    
    Mathematical meaning: MSE measures the average squared difference between the estimated values 
    and the actual value. A lower MSE indicates less error (better similarity).
    """
    _check_dimensions(img1, img2)
    return mean_squared_error(img1, img2)

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Mathematical meaning: PSNR represents the ratio between the maximum possible power of a signal 
    and the power of corrupting noise that affects the fidelity of its representation. 
    A higher PSNR generally indicates higher quality / better restoration.
    """
    _check_dimensions(img1, img2)
    return peak_signal_noise_ratio(img1, img2)

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculates Structural Similarity Index Measure (SSIM) between two images.
    
    Mathematical meaning: SSIM measures the similarity between two images considering 
    luminance, contrast, and structure. Value ranges from -1 to 1, where 1 indicates perfect similarity.
    Higher generally indicates greater structural similarity.
    """
    _check_dimensions(img1, img2)
    
    # Check if images are multichannel (e.g. RGB)
    multichannel = len(img1.shape) == 3 and img1.shape[2] in [3, 4]
    
    # scikit-image >= 0.19 uses channel_axis instead of multichannel
    # We will pass channel_axis=-1 for RGB images.
    if multichannel:
        return structural_similarity(img1, img2, channel_axis=-1)
    else:
        return structural_similarity(img1, img2)

def evaluate_restoration(degraded: np.ndarray, restored: np.ndarray, clean: np.ndarray) -> Dict[str, float]:
    """
    Evaluates a restoration attempt by comparing degraded vs clean and restored vs clean.
    
    Returns a dictionary containing:
    - degraded_vs_ground_truth MSE, PSNR, SSIM
    - restored_vs_ground_truth MSE, PSNR, SSIM
    - improvement values (restored - degraded)
    """
    
    deg_mse = calculate_mse(clean, degraded)
    deg_psnr = calculate_psnr(clean, degraded)
    deg_ssim = calculate_ssim(clean, degraded)
    
    rest_mse = calculate_mse(clean, restored)
    rest_psnr = calculate_psnr(clean, restored)
    rest_ssim = calculate_ssim(clean, restored)
    
    return {
        'degraded_vs_ground_truth_mse': deg_mse,
        'degraded_vs_ground_truth_psnr': deg_psnr,
        'degraded_vs_ground_truth_ssim': deg_ssim,
        
        'restored_vs_ground_truth_mse': rest_mse,
        'restored_vs_ground_truth_psnr': rest_psnr,
        'restored_vs_ground_truth_ssim': rest_ssim,
        
        'mse_improvement': deg_mse - rest_mse, # Positive means error decreased (improvement)
        'psnr_improvement': rest_psnr - deg_psnr, # Positive means PSNR increased (improvement)
        'ssim_improvement': rest_ssim - deg_ssim  # Positive means SSIM increased (improvement)
    }
