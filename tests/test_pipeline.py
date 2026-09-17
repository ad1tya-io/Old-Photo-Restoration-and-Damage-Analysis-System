import pytest
import numpy as np
import cv2
from src.pipeline import RestorationPipeline, RestorationResult
from src.degradation_analysis import DegradationReport

@pytest.fixture
def sample_clean_rgb():
    # A perfectly clean, high-contrast, sharp image
    np.random.seed(42)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = 200
    img[:, 32:] = 50
    return img

def test_invalid_image_rejection():
    pipeline = RestorationPipeline()
    
    with pytest.raises((ValueError, TypeError)):
        pipeline.process("not_an_image")
        
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((64, 64), dtype=np.uint8))  # 2D grayscale
        
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((64, 64, 3), dtype=np.float32))  # float32

def test_valid_processing_and_output_structure(sample_clean_rgb):
    pipeline = RestorationPipeline()
    result = pipeline.process(sample_clean_rgb)
    
    # 3. Output structure
    assert isinstance(result, RestorationResult)
    
    # 7. Degradation report included
    assert isinstance(result.degradation_report, DegradationReport)
    
    # 8. Operations list populated
    assert isinstance(result.operations_applied, list)
    assert len(result.operations_applied) > 0
    
    # 4. Output dimensions
    assert result.restored_image.shape == sample_clean_rgb.shape
    
    # 5. uint8 output
    assert result.restored_image.dtype == np.uint8

def test_original_image_immutability(sample_clean_rgb):
    pipeline = RestorationPipeline()
    original = sample_clean_rgb.copy()
    
    result = pipeline.process(sample_clean_rgb)
    
    assert np.array_equal(sample_clean_rgb, original)
    # the returned original_image should also be a copy/unmodified
    assert np.array_equal(result.original_image, original)

def test_deterministic_behavior(sample_clean_rgb):
    pipeline = RestorationPipeline()
    result1 = pipeline.process(sample_clean_rgb)
    result2 = pipeline.process(sample_clean_rgb)
    
    assert np.array_equal(result1.restored_image, result2.restored_image)
    assert result1.operations_applied == result2.operations_applied

def test_rule_logic_clean_image(sample_clean_rgb):
    # A clean image with high contrast and sharp edge should trigger no operations
    pipeline = RestorationPipeline()
    result = pipeline.process(sample_clean_rgb)
    
    assert "none (image considered clean)" in result.operations_applied
    assert not result.degradation_report.is_noisy
    assert not result.degradation_report.is_blurry
    assert not result.degradation_report.is_low_contrast
    assert not result.degradation_report.has_artifacts

def test_rule_logic_noisy_image(sample_clean_rgb):
    # Add severe Gaussian noise
    np.random.seed(42)
    noise = np.random.normal(0, 50, sample_clean_rgb.shape)
    noisy_img = np.clip(sample_clean_rgb.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    pipeline = RestorationPipeline()
    result = pipeline.process(noisy_img)
    
    assert result.degradation_report.is_noisy
    assert "bilateral_denoise" in result.operations_applied

def test_rule_logic_low_contrast_image():
    # Very low contrast image
    low_contrast_img = np.full((64, 64, 3), 100, dtype=np.uint8)
    # Add tiny variation so std is not exactly zero, but very small
    low_contrast_img[:, :32] = 105
    
    pipeline = RestorationPipeline()
    result = pipeline.process(low_contrast_img)
    
    assert result.degradation_report.is_low_contrast
    assert "clahe_enhancement" in result.operations_applied

def test_rule_logic_blurred_image(sample_clean_rgb):
    # Extremely blurred image
    blurred_img = cv2.GaussianBlur(sample_clean_rgb, (15, 15), 10.0)
    
    pipeline = RestorationPipeline()
    result = pipeline.process(blurred_img)
    
    assert result.degradation_report.is_blurry
    assert "unsharp_mask" in result.operations_applied
    
def test_intermediate_steps(sample_clean_rgb):
    # Force some operations to see if intermediates are collected
    blurred_img = cv2.GaussianBlur(sample_clean_rgb, (15, 15), 10.0)
    pipeline = RestorationPipeline()
    result = pipeline.process(blurred_img, return_intermediates=True)
    
    assert len(result.intermediate_steps) > 0
    step_name, step_img = result.intermediate_steps[0]
    assert isinstance(step_name, str)
    assert isinstance(step_img, np.ndarray)
