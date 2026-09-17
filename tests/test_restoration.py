import pytest
import numpy as np
from src.restoration import (
    median_denoise,
    gaussian_denoise,
    bilateral_denoise,
    histogram_equalization,
    clahe_enhancement,
    unsharp_mask,
    laplacian_sharpen,
)

@pytest.fixture
def sample_rgb_image():
    # Create a deterministic 32x32 RGB image
    np.random.seed(42)
    return np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

def test_input_immutability(sample_rgb_image):
    original = sample_rgb_image.copy()
    
    _ = median_denoise(sample_rgb_image)
    _ = gaussian_denoise(sample_rgb_image)
    _ = bilateral_denoise(sample_rgb_image)
    _ = histogram_equalization(sample_rgb_image)
    _ = clahe_enhancement(sample_rgb_image)
    _ = unsharp_mask(sample_rgb_image)
    _ = laplacian_sharpen(sample_rgb_image)
    
    assert np.array_equal(sample_rgb_image, original), "Input image was modified in-place!"

def test_dimensions_and_dtype_preserved(sample_rgb_image):
    functions = [
        lambda img: median_denoise(img, ksize=3),
        lambda img: gaussian_denoise(img, ksize=3, sigma=1.0),
        lambda img: bilateral_denoise(img, d=5, sigma_color=50.0, sigma_space=50.0),
        lambda img: histogram_equalization(img),
        lambda img: clahe_enhancement(img, clip_limit=2.0),
        lambda img: unsharp_mask(img, kernel_size=(3, 3), amount=1.5),
        lambda img: laplacian_sharpen(img, alpha=0.1),
    ]
    
    for fn in functions:
        res = fn(sample_rgb_image)
        assert isinstance(res, np.ndarray)
        assert res.shape == sample_rgb_image.shape
        assert res.dtype == np.uint8

def test_invalid_input_shapes_and_types():
    invalid_inputs = [
        "not_an_image",
        np.zeros((32, 32), dtype=np.uint8),  # 2D grayscale
        np.zeros((32, 32, 4), dtype=np.uint8),  # RGBA
        np.zeros((32, 32, 3), dtype=np.float32),  # float instead of uint8
        np.zeros((0, 32, 3), dtype=np.uint8),  # 0 height
    ]
    
    functions = [
        median_denoise,
        gaussian_denoise,
        bilateral_denoise,
        histogram_equalization,
        clahe_enhancement,
        unsharp_mask,
        laplacian_sharpen,
    ]
    
    for inp in invalid_inputs:
        for fn in functions:
            with pytest.raises((ValueError, TypeError)):
                fn(inp)

def test_median_denoise_parameters(sample_rgb_image):
    # Even ksize or negative ksize should fail
    with pytest.raises(ValueError):
        median_denoise(sample_rgb_image, ksize=4)
    with pytest.raises(ValueError):
        median_denoise(sample_rgb_image, ksize=-1)

def test_gaussian_denoise_parameters(sample_rgb_image):
    with pytest.raises(ValueError):
        gaussian_denoise(sample_rgb_image, ksize=2)
    with pytest.raises(ValueError):
        gaussian_denoise(sample_rgb_image, ksize=3, sigma=-1.0)

def test_bilateral_denoise_parameters(sample_rgb_image):
    with pytest.raises(ValueError):
        bilateral_denoise(sample_rgb_image, d=0)
    with pytest.raises(ValueError):
        bilateral_denoise(sample_rgb_image, sigma_color=-10)

def test_clahe_parameters(sample_rgb_image):
    with pytest.raises(ValueError):
        clahe_enhancement(sample_rgb_image, clip_limit=-1.0)
    with pytest.raises(ValueError):
        clahe_enhancement(sample_rgb_image, tile_grid_size=(0, 8))

def test_unsharp_mask_parameters(sample_rgb_image):
    with pytest.raises(ValueError):
        unsharp_mask(sample_rgb_image, kernel_size=(4, 4))
    with pytest.raises(ValueError):
        unsharp_mask(sample_rgb_image, amount=-0.5)

def test_laplacian_sharpen_parameters(sample_rgb_image):
    with pytest.raises(ValueError):
        laplacian_sharpen(sample_rgb_image, alpha=-0.1)
