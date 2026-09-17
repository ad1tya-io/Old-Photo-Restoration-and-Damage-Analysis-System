import pytest
import numpy as np
import cv2
from src.artifact_removal import refine_artifact_mask, remove_artifacts

@pytest.fixture
def clean_rgb_image():
    # 64x64 solid gray image
    return np.full((64, 64, 3), 128, dtype=np.uint8)

@pytest.fixture
def damaged_image(clean_rgb_image):
    # Add a bright red "scratch"
    img = clean_rgb_image.copy()
    cv2.line(img, (10, 10), (20, 20), (255, 0, 0), 2)
    return img

@pytest.fixture
def perfect_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.line(mask, (10, 10), (20, 20), 255, 2)
    return mask

def test_refine_artifact_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    # 1. Tiny component (area ~ 1 pixel) - should be removed
    mask[5, 5] = 255
    
    # 2. Huge component (area > 500) - should be removed
    cv2.rectangle(mask, (20, 20), (50, 50), 255, -1) # Area = 31*31 = 961
    
    # 3. Valid scratch (area ~ 10-20 pixels) - should be kept
    cv2.line(mask, (70, 70), (80, 70), 255, 2)
    
    refined = refine_artifact_mask(mask, min_area=5, max_area=500)
    
    # Tiny should be gone
    assert refined[5, 5] == 0
    
    # Huge should be gone
    assert refined[30, 30] == 0
    
    # Valid scratch should remain
    assert refined[70, 75] == 255
    
def test_remove_artifacts_valid_inputs(damaged_image, perfect_mask):
    restored = remove_artifacts(damaged_image, perfect_mask)
    
    # Check dimensions and type
    assert restored.shape == damaged_image.shape
    assert restored.dtype == np.uint8
    
    # Check that inpainting actually changed the red scratch pixels
    # Original damaged pixel was [255, 0, 0]. It should now be closer to [128, 128, 128]
    assert not np.array_equal(restored[15, 15], damaged_image[15, 15])

def test_original_image_unchanged(damaged_image, perfect_mask):
    original = damaged_image.copy()
    _ = remove_artifacts(damaged_image, perfect_mask)
    assert np.array_equal(damaged_image, original)
    
def test_empty_mask_behaves_safely(damaged_image):
    empty_mask = np.zeros((64, 64), dtype=np.uint8)
    restored = remove_artifacts(damaged_image, empty_mask)
    # Should just return copy of original
    assert np.array_equal(restored, damaged_image)

def test_invalid_mask_dimensions_rejected(damaged_image):
    bad_mask = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError):
        remove_artifacts(damaged_image, bad_mask)

def test_invalid_mask_type_rejected(damaged_image):
    with pytest.raises(TypeError):
        remove_artifacts(damaged_image, "not_a_mask")
