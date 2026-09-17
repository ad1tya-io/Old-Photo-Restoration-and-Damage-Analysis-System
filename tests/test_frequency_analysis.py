import pytest
import numpy as np
from src.frequency_analysis import (
    compute_fft,
    create_low_pass_mask,
    create_high_pass_mask,
    apply_frequency_filter,
    frequency_denoise,
    frequency_sharpen,
)

@pytest.fixture
def sample_rgb_image():
    np.random.seed(42)
    return np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

@pytest.fixture
def sample_gray_image():
    np.random.seed(42)
    return np.random.randint(0, 256, (32, 32), dtype=np.uint8)

def test_valid_image_input(sample_rgb_image, sample_gray_image):
    fft_rgb = compute_fft(sample_rgb_image)
    fft_gray = compute_fft(sample_gray_image)
    assert isinstance(fft_rgb, np.ndarray)
    assert isinstance(fft_gray, np.ndarray)

def test_invalid_image_input():
    invalid_inputs = [
        "not_an_image",
        np.zeros((32, 32, 4), dtype=np.uint8),  # RGBA
        np.zeros((32, 32, 3), dtype=np.float32),  # float dtype
        np.zeros((0, 32, 3), dtype=np.uint8),  # zero height
    ]
    for inp in invalid_inputs:
        with pytest.raises((ValueError, TypeError)):
            compute_fft(inp)

def test_fft_output_dimensions(sample_rgb_image):
    fft_spectrum = compute_fft(sample_rgb_image)
    assert fft_spectrum.shape == (32, 32)
    assert fft_spectrum.dtype in [np.float32, np.float64]

def test_low_pass_mask_dimensions_and_values():
    shape = (32, 32)
    mask = create_low_pass_mask(shape, cutoff=10.0)
    assert mask.shape == shape
    assert np.all((mask == 0.0) | (mask == 1.0))
    # Center pixel (16, 16) should be 1.0
    assert mask[16, 16] == 1.0
    # Corner pixel (0, 0) should be 0.0
    assert mask[0, 0] == 0.0

def test_high_pass_mask_dimensions_and_values():
    shape = (32, 32)
    lp_mask = create_low_pass_mask(shape, cutoff=10.0)
    hp_mask = create_high_pass_mask(shape, cutoff=10.0)
    assert hp_mask.shape == shape
    assert np.allclose(lp_mask + hp_mask, 1.0)

def test_filtered_output_dimensions_and_uint8(sample_rgb_image):
    mask = create_low_pass_mask((32, 32), cutoff=10.0)
    filtered = apply_frequency_filter(sample_rgb_image, mask)
    assert filtered.shape == sample_rgb_image.shape
    assert filtered.dtype == np.uint8

def test_original_input_remains_unchanged(sample_rgb_image):
    original = sample_rgb_image.copy()
    _ = compute_fft(sample_rgb_image)
    mask = create_low_pass_mask((32, 32), cutoff=10.0)
    _ = apply_frequency_filter(sample_rgb_image, mask)
    _ = frequency_denoise(sample_rgb_image, cutoff=15.0)
    _ = frequency_sharpen(sample_rgb_image, cutoff=15.0)
    assert np.array_equal(sample_rgb_image, original)

def test_invalid_cutoff_values_rejected(sample_rgb_image):
    with pytest.raises(ValueError):
        create_low_pass_mask((32, 32), cutoff=0.0)
    with pytest.raises(ValueError):
        create_low_pass_mask((32, 32), cutoff=-5.0)
    with pytest.raises(ValueError):
        create_high_pass_mask((32, 32), cutoff=-1.0)
    with pytest.raises(ValueError):
        frequency_denoise(sample_rgb_image, cutoff=-10.0)
    with pytest.raises(ValueError):
        frequency_sharpen(sample_rgb_image, cutoff=10.0, alpha=-0.5)
