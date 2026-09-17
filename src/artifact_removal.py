import cv2
import numpy as np
from typing import Tuple

from src.restoration import _validate_rgb_image

def refine_artifact_mask(artifact_mask: np.ndarray, 
                         min_area: int = 5, 
                         max_area: int = 500) -> np.ndarray:
    """
    Refines a raw morphological artifact mask to reduce false positives.
    Filters out connected components that are too small (noise) or too large
    (natural scene structures, facial features, or heavy lighting gradients).
    
    Args:
        artifact_mask: 2D binary NumPy array (H, W) of dtype uint8.
        min_area: Minimum pixel area for a valid artifact.
        max_area: Maximum pixel area for a valid artifact.
        
    Returns:
        Refined binary artifact mask (H, W) of dtype uint8.
    """
    if not isinstance(artifact_mask, np.ndarray) or artifact_mask.ndim != 2:
        raise ValueError("artifact_mask must be a 2D NumPy array.")
    
    # Ensure binary format (0 or 255)
    binary_mask = (artifact_mask > 0).astype(np.uint8) * 255
    
    # Extract connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    refined_mask = np.zeros_like(binary_mask)
    
    # Loop over components (skip label 0, which is background)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if min_area <= area <= max_area:
            # Component passes size filter, add to refined mask
            refined_mask[labels == i] = 255
            
    return refined_mask

def remove_artifacts(image: np.ndarray, 
                     artifact_mask: np.ndarray, 
                     inpaint_radius: int = 3, 
                     method: int = cv2.INPAINT_TELEA) -> np.ndarray:
    """
    Fills in detected scratches/dust/artifacts using classical image inpainting.
    
    Design Choice:
    Defaults to cv2.INPAINT_TELEA (Fast Marching Method based on Alexandru Telea's algorithm).
    Telea is typically faster and works exceptionally well for thin, linear scratches 
    and small point-like dust spots common in old photographs compared to Navier-Stokes.
    
    Args:
        image: 3-channel RGB image (H, W, 3) of dtype uint8.
        artifact_mask: 2D binary artifact mask (H, W) of dtype uint8 (255 for artifacts).
        inpaint_radius: Radius of a circular neighborhood of each point inpainted.
        method: OpenCV inpainting algorithm flag (cv2.INPAINT_TELEA or cv2.INPAINT_NS).
        
    Returns:
        Inpainted RGB image (H, W, 3) of dtype uint8.
    """
    img = _validate_rgb_image(image)
    
    if not isinstance(artifact_mask, np.ndarray):
        raise TypeError("artifact_mask must be a NumPy array.")
    
    if artifact_mask.ndim != 2:
        raise ValueError(f"artifact_mask must be 2D. Got shape: {artifact_mask.shape}")
        
    if artifact_mask.shape != img.shape[:2]:
        raise ValueError(f"Mask dimensions {artifact_mask.shape} do not match image dimensions {img.shape[:2]}")
        
    if artifact_mask.dtype != np.uint8:
        artifact_mask = artifact_mask.astype(np.uint8)
        
    # Check if mask is empty
    if np.count_nonzero(artifact_mask) == 0:
        return img.copy()
        
    # Ensure binary mask values are exactly 0 or 255 for cv2.inpaint
    binary_mask = np.where(artifact_mask > 0, 255, 0).astype(np.uint8)
    
    # Perform OpenCV Inpainting
    restored = cv2.inpaint(img, binary_mask, inpaintRadius=inpaint_radius, flags=method)
    
    return restored
