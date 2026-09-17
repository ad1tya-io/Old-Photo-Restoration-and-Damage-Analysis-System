import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

from src.degradation_analysis import DegradationAnalyzer, DegradationReport
from src.restoration import (
    bilateral_denoise,
    clahe_enhancement,
    unsharp_mask,
    _validate_rgb_image
)
from src.artifact_removal import refine_artifact_mask, remove_artifacts

@dataclass
class RestorationResult:
    """
    Structured result of the restoration pipeline.
    """
    original_image: np.ndarray
    restored_image: np.ndarray
    degradation_report: DegradationReport
    operations_applied: List[str]
    intermediate_steps: List[Tuple[str, np.ndarray]] = field(default_factory=list)

class RestorationPipeline:
    """
    Integration layer that connects degradation analysis with restoration algorithms.
    Uses a rule-based decision system to selectively apply operations.
    
    Order of Operations Rationale:
    1. Denoising: Must be done first to avoid amplifying noise during enhancement.
    2. Contrast Enhancement: Operates best on denoised, clean structures.
    3. Artifact Removal: Inpaints dust and scratches using structural information from surrounding pixels.
    4. Sharpening: Done last to crispen edges without sharpening noise or artifacts.
    """
    
    def __init__(self,
                 analyzer: Optional[DegradationAnalyzer] = None,
                 denoise_d: int = 9,
                 denoise_sigma_color: float = 75.0,
                 denoise_sigma_space: float = 75.0,
                 clahe_clip_limit: float = 2.0,
                 sharpen_amount: float = 1.0,
                 artifact_min_area: int = 5,
                 artifact_max_area: int = 500,
                 artifact_inpaint_radius: int = 3):
        """
        Initializes the pipeline with configurable restoration parameters.
        """
        self.analyzer = analyzer or DegradationAnalyzer()
        
        # Parameters for restoration algorithms
        self.denoise_d = denoise_d
        self.denoise_sigma_color = denoise_sigma_color
        self.denoise_sigma_space = denoise_sigma_space
        self.clahe_clip_limit = clahe_clip_limit
        self.sharpen_amount = sharpen_amount
        self.artifact_min_area = artifact_min_area
        self.artifact_max_area = artifact_max_area
        self.artifact_inpaint_radius = artifact_inpaint_radius

    def process(self, image: np.ndarray, return_intermediates: bool = False) -> RestorationResult:
        """
        Processes an RGB image through the analysis and rule-based restoration pipeline.
        
        Args:
            image: 3-channel RGB image (H, W, 3) of dtype uint8.
            return_intermediates: Whether to store copies of intermediate steps (increases memory usage).
            
        Returns:
            RestorationResult containing the final image, report, and operation list.
        """
        # 1. Validation (raises errors if invalid)
        _validate_rgb_image(image)
        
        # We don't modify the original image
        current_img = image.copy()
        operations_applied = []
        intermediate_steps = []
        
        # 2. Degradation Analysis
        report = self.analyzer.analyze(current_img)
        
        # 3. Rule-Based Restoration Decision & Operations
        
        # RULE 1: Noise
        if report.is_noisy:
            current_img = bilateral_denoise(
                current_img, 
                d=self.denoise_d, 
                sigma_color=self.denoise_sigma_color, 
                sigma_space=self.denoise_sigma_space
            )
            operations_applied.append("bilateral_denoise")
            if return_intermediates:
                intermediate_steps.append(("bilateral_denoise", current_img.copy()))
                
        # RULE 2: Contrast
        if report.is_low_contrast:
            current_img = clahe_enhancement(current_img, clip_limit=self.clahe_clip_limit)
            operations_applied.append("clahe_enhancement")
            if return_intermediates:
                intermediate_steps.append(("clahe_enhancement", current_img.copy()))
                
        # RULE 3: Artifacts
        if report.has_artifacts:
            # Refine the raw morphological mask to prevent false positives from facial features/textures
            refined_mask = refine_artifact_mask(
                report.artifact_mask, 
                min_area=self.artifact_min_area, 
                max_area=self.artifact_max_area
            )
            
            # Only inpaint if valid scratch/dust regions remain after refinement
            if np.count_nonzero(refined_mask) > 0:
                current_img = remove_artifacts(
                    current_img, 
                    refined_mask, 
                    inpaint_radius=self.artifact_inpaint_radius
                )
                operations_applied.append("artifact_inpainting")
                if return_intermediates:
                    intermediate_steps.append(("artifact_inpainting", current_img.copy()))
            else:
                operations_applied.append("artifact_processing_required (skipped: mask refinement yielded no reliable regions)")
            
        # RULE 4: Blur
        if report.is_blurry:
            current_img = unsharp_mask(current_img, amount=self.sharpen_amount)
            operations_applied.append("unsharp_mask")
            if return_intermediates:
                intermediate_steps.append(("unsharp_mask", current_img.copy()))
                
        # If no operations were applied, record that fact
        if not operations_applied:
            operations_applied.append("none (image considered clean)")
            
        return RestorationResult(
            original_image=image.copy(),
            restored_image=current_img,
            degradation_report=report,
            operations_applied=operations_applied,
            intermediate_steps=intermediate_steps
        )
