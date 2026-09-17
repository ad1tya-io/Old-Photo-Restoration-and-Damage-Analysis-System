import pytest
import numpy as np
from src.metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    evaluate_restoration,
    _check_dimensions
)

@pytest.fixture
def clean_img():
    # 10x10 RGB image (all zeros)
    return np.zeros((10, 10, 3), dtype=np.uint8)

@pytest.fixture
def degraded_img():
    # 10x10 RGB image (all ones)
    return np.ones((10, 10, 3), dtype=np.uint8) * 100

@pytest.fixture
def restored_img():
    # 10x10 RGB image (all ones, but closer to clean)
    return np.ones((10, 10, 3), dtype=np.uint8) * 10

def test_check_dimensions():
    img1 = np.zeros((10, 10, 3))
    img2 = np.zeros((10, 10, 3))
    img3 = np.zeros((10, 10, 4))
    img4 = np.zeros((15, 15, 3))
    
    # Should not raise
    _check_dimensions(img1, img2)
    
    # Should raise
    with pytest.raises(ValueError):
        _check_dimensions(img1, img3)
        
    with pytest.raises(ValueError):
        _check_dimensions(img1, img4)

def test_calculate_mse(clean_img, degraded_img, restored_img):
    mse_deg = calculate_mse(clean_img, degraded_img)
    mse_rest = calculate_mse(clean_img, restored_img)
    
    assert mse_deg > 0
    assert mse_rest > 0
    assert mse_rest < mse_deg  # Restored is closer to clean (0) than degraded (100)
    
    assert calculate_mse(clean_img, clean_img) == 0.0

def test_calculate_psnr(clean_img, degraded_img, restored_img):
    # scikit-image peak_signal_noise_ratio returns inf if MSE is 0
    
    # Check that restored has better PSNR than degraded
    psnr_deg = calculate_psnr(clean_img, degraded_img)
    psnr_rest = calculate_psnr(clean_img, restored_img)
    
    assert psnr_rest > psnr_deg

def test_calculate_ssim(clean_img, degraded_img, restored_img):
    ssim_perfect = calculate_ssim(clean_img, clean_img)
    assert ssim_perfect == 1.0
    
    # Just checking it runs without error for RGB
    ssim_deg = calculate_ssim(clean_img, degraded_img)
    assert isinstance(ssim_deg, float)

def test_evaluate_restoration(clean_img, degraded_img, restored_img):
    results = evaluate_restoration(degraded_img, restored_img, clean_img)
    
    expected_keys = [
        'degraded_vs_ground_truth_mse',
        'degraded_vs_ground_truth_psnr',
        'degraded_vs_ground_truth_ssim',
        'restored_vs_ground_truth_mse',
        'restored_vs_ground_truth_psnr',
        'restored_vs_ground_truth_ssim',
        'mse_improvement',
        'psnr_improvement',
        'ssim_improvement'
    ]
    
    for k in expected_keys:
        assert k in results
        
    assert results['mse_improvement'] > 0
    assert results['psnr_improvement'] > 0
