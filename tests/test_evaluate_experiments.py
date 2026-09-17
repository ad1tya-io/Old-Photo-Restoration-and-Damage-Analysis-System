import pytest
import pandas as pd
from src.evaluate_experiments import aggregate_statistics

def test_aggregate_statistics_empty():
    df = pd.DataFrame()
    stats = aggregate_statistics(df, start_time=0.0, total_pairs=0, failed=0, mismatch=0, output_csv="dummy.csv")
    assert stats['evaluated_pairs'] == 0
    assert stats['runtime_seconds'] > 0
    assert 'count' not in stats.get('overall', {})

def test_aggregate_statistics_valid():
    data = [
        {
            'degradation_category': 'Noise_JPEG_Blur',
            'mse_before': 100.0, 'mse_after': 50.0, 'mse_delta': 50.0,
            'psnr_before': 20.0, 'psnr_after': 25.0, 'psnr_delta': 5.0,
            'ssim_before': 0.70, 'ssim_after': 0.85, 'ssim_delta': 0.15,
            'operations_applied': 'bilateral_denoise|unsharp_mask'
        },
        {
            'degradation_category': 'Noise_JPEG_Blur',
            'mse_before': 80.0, 'mse_after': 90.0, 'mse_delta': -10.0,
            'psnr_before': 22.0, 'psnr_after': 21.0, 'psnr_delta': -1.0,
            'ssim_before': 0.75, 'ssim_after': 0.70, 'ssim_delta': -0.05,
            'operations_applied': 'bilateral_denoise'
        }
    ]
    df = pd.DataFrame(data)
    stats = aggregate_statistics(df, start_time=0.0, total_pairs=2, failed=0, mismatch=0, output_csv="dummy.csv")
    
    # Check overall
    assert stats['evaluated_pairs'] == 2
    assert stats['overall']['count'] == 2
    assert stats['overall']['mean_mse_before'] == 90.0
    assert stats['overall']['percent_mse_improved'] == 50.0
    
    # Check operations
    assert stats['operations_count']['bilateral_denoise'] == 2
    assert stats['operations_count']['unsharp_mask'] == 1
    
    # Check category
    cat_stats = stats['by_category']['Noise_JPEG_Blur']
    assert cat_stats['count'] == 2
