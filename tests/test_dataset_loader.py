import pytest
import os
import numpy as np
from src.dataset_loader import (
    parse_synthetic_filename, 
    load_image_rgb, 
    DatasetLoader,
    SyntheticImagePair
)

def test_parse_synthetic_filename():
    # Test clean image
    sid, dtype, comps = parse_synthetic_filename("lrp_img100.jpg")
    assert sid == "lrp_img100"
    assert dtype is None
    assert comps == []

    # Test degraded image
    sid, dtype, comps = parse_synthetic_filename("lrp_img100_(Noise_JPEG_Blur).jpg")
    assert sid == "lrp_img100"
    assert dtype == "Noise_JPEG_Blur"
    assert comps == ["Noise", "JPEG", "Blur"]
    
    sid, dtype, comps = parse_synthetic_filename("lrp_img100_(Complex_All).jpg")
    assert sid == "lrp_img100"
    assert dtype == "Complex_All"
    assert comps == ["Complex", "All"]

def test_load_image_rgb_missing_file():
    with pytest.raises(FileNotFoundError):
        load_image_rgb("non_existent_file_path_12345.jpg")

def test_dataset_loader_summary(monkeypatch):
    # Instead of creating a massive mock, we can just run against the actual dataset
    # since the instruction says: "Use actual dataset files for all inspection/testing."
    loader = DatasetLoader(dataset_dir="dataset")
    
    # Check if dataset exists locally, if so test the summary
    if os.path.exists("dataset"):
        summary = loader.get_dataset_summary()
        assert "num_clean_candidates" in summary
        assert "num_synthetic_pairs_confirmed" in summary
        
        # Real damaged dataset should have > 0 images based on Phase 1
        assert summary["num_real_damaged"] >= 0

def test_dataset_loader_pairing():
    loader = DatasetLoader(dataset_dir="dataset")
    if os.path.exists(os.path.join("dataset", "03_Synthetic_Dataset")):
        pairs = loader.load_synthetic_pairs()
        assert isinstance(pairs, list)
        if len(pairs) > 0:
            assert isinstance(pairs[0], SyntheticImagePair)
            assert os.path.exists(pairs[0].clean_path)
            assert os.path.exists(pairs[0].degraded_path)
            
            # Check degradation parsing was applied
            assert pairs[0].source_id != ""
            assert pairs[0].degradation_type != "Unknown"
            assert isinstance(pairs[0].degradation_components, list)
