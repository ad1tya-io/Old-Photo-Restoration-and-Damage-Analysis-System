import pytest
import numpy as np
import cv2
from src.degradation_analysis import (
    DegradationAnalyzer,
    DegradationReport,
    estimate_noise,
    estimate_blur,
    estimate_contrast,
    estimate_brightness,
    detect_artifacts,
)

@pytest.fixture
def clean_rgb_image():
    # Deterministic clean image with a sharp edge
    np.random.seed(42)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = 200
    img[:, 32:] = 50
    return img

def test_valid_rgb_image_accepted(clean_rgb_image):
    analyzer = DegradationAnalyzer()
    report = analyzer.analyze(clean_rgb_image)
    assert isinstance(report, DegradationReport)
    assert report.image_dimensions == (64, 64, 3)

def test_invalid_shape_rejected():
    analyzer = DegradationAnalyzer()
    
    # 2D grayscale
    with pytest.raises(ValueError):
        analyzer.analyze(np.zeros((64, 64), dtype=np.uint8))
        
    # 4D tensor
    with pytest.raises(ValueError):
        analyzer.analyze(np.zeros((1, 64, 64, 3), dtype=np.uint8))
        
    # RGBA
    with pytest.raises(ValueError):
        analyzer.analyze(np.zeros((64, 64, 4), dtype=np.uint8))

def test_invalid_dtype_rejected():
    analyzer = DegradationAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyze(np.zeros((64, 64, 3), dtype=np.float32))

def test_empty_image_rejected():
    analyzer = DegradationAnalyzer()
    with pytest.raises(ValueError):
        analyzer.analyze(np.zeros((0, 64, 3), dtype=np.uint8))

def test_degradation_report_structure(clean_rgb_image):
    analyzer = DegradationAnalyzer()
    report = analyzer.analyze(clean_rgb_image)
    
    assert hasattr(report, "noise_score")
    assert hasattr(report, "blur_score")
    assert hasattr(report, "contrast_std")
    assert hasattr(report, "dynamic_range")
    assert hasattr(report, "mean_brightness")
    assert hasattr(report, "artifact_ratio")
    assert hasattr(report, "artifact_count")
    assert hasattr(report, "is_noisy")
    assert hasattr(report, "is_blurry")
    assert hasattr(report, "is_low_contrast")
    assert hasattr(report, "has_artifacts")
    assert hasattr(report, "artifact_mask")
    
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "artifact_mask" not in d  # Dictionary summary excludes heavy numpy array

def test_determinism(clean_rgb_image):
    analyzer = DegradationAnalyzer()
    report1 = analyzer.analyze(clean_rgb_image)
    report2 = analyzer.analyze(clean_rgb_image)
    
    assert report1.noise_score == report2.noise_score
    assert report1.blur_score == report2.blur_score
    assert report1.contrast_std == report2.contrast_std
    assert report1.dynamic_range == report2.dynamic_range
    assert report1.mean_brightness == report2.mean_brightness
    assert report1.artifact_ratio == report2.artifact_ratio
    assert report1.artifact_count == report2.artifact_count
    assert np.array_equal(report1.artifact_mask, report2.artifact_mask)

def test_artifact_mask_dimensions(clean_rgb_image):
    analyzer = DegradationAnalyzer()
    report = analyzer.analyze(clean_rgb_image)
    assert report.artifact_mask.shape == (64, 64)
    assert report.artifact_mask.dtype == np.uint8

def test_synthetic_degradation_trends(clean_rgb_image):
    analyzer = DegradationAnalyzer()
    clean_report = analyzer.analyze(clean_rgb_image)
    
    # 1. Test Noise: Add random Gaussian noise -> noise_score should increase
    np.random.seed(42)
    noisy_img = clean_rgb_image.astype(np.float32) + np.random.normal(0, 25, clean_rgb_image.shape)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    noisy_report = analyzer.analyze(noisy_img)
    assert noisy_report.noise_score > clean_report.noise_score
    
    # 2. Test Blur: Apply strong Gaussian blur -> blur_score (Laplacian variance) should decrease
    blurred_img = cv2.GaussianBlur(clean_rgb_image, (15, 15), 5.0)
    blurred_report = analyzer.analyze(blurred_img)
    assert blurred_report.blur_score < clean_report.blur_score
    
    # 3. Test Artifacts: Draw bright scratches -> artifact_ratio should increase
    scratch_img = clean_rgb_image.copy()
    cv2.line(scratch_img, (0, 0), (63, 63), (255, 255, 255), 2)
    scratch_report = analyzer.analyze(scratch_img)
    assert scratch_report.artifact_ratio > clean_report.artifact_ratio
