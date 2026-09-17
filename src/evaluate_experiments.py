import os
import time
import pandas as pd
from typing import List, Dict, Any
from collections import Counter

from src.dataset_loader import DatasetLoader, load_image_rgb
from src.pipeline import RestorationPipeline
from src.metrics import evaluate_restoration

def run_evaluation(dataset_dir: str = 'dataset', output_csv: str = 'results/metrics/restoration_evaluation.csv') -> Dict[str, Any]:
    """
    Runs a quantitative evaluation experiment of the restoration pipeline on the synthetic dataset.
    
    Args:
        dataset_dir: Path to dataset directory.
        output_csv: Path to save detailed per-image metrics.
        
    Returns:
        A dictionary containing aggregated statistics.
    """
    start_time = time.time()
    
    loader = DatasetLoader(dataset_dir)
    pipeline = RestorationPipeline()
    
    # Load and sort pairs for deterministic evaluation
    pairs = loader.load_synthetic_pairs()
    pairs = sorted(pairs, key=lambda p: (p.source_id, p.degradation_type))
    
    total_pairs = len(pairs)
    print(f"Starting evaluation of {total_pairs} synthetic image pairs...")
    
    results = []
    failed_images = 0
    mismatch_images = 0
    
    for i, pair in enumerate(pairs):
        if (i + 1) % 50 == 0 or i == 0 or i == total_pairs - 1:
            print(f"Processing image {i + 1}/{total_pairs}...")
            
        try:
            # 1. & 2. Load images
            clean_img = load_image_rgb(pair.clean_path)
            degraded_img = load_image_rgb(pair.degraded_path)
            
            # Check dimensional compatibility
            if clean_img.shape != degraded_img.shape:
                print(f"Warning: Dimension mismatch for {pair.source_id}. Clean: {clean_img.shape}, Degraded: {degraded_img.shape}. Skipping.")
                mismatch_images += 1
                continue
                
            # 3. Run Pipeline
            pipeline_result = pipeline.process(degraded_img)
            restored_img = pipeline_result.restored_image
            
            # 5. Calculate Metrics
            metrics = evaluate_restoration(degraded_img, restored_img, clean_img)
            
            # Record results
            row = {
                'clean_filename': os.path.basename(pair.clean_path),
                'degraded_filename': os.path.basename(pair.degraded_path),
                'source_id': pair.source_id,
                'degradation_category': pair.degradation_type,
                'mse_before': metrics['degraded_vs_ground_truth_mse'],
                'psnr_before': metrics['degraded_vs_ground_truth_psnr'],
                'ssim_before': metrics['degraded_vs_ground_truth_ssim'],
                'mse_after': metrics['restored_vs_ground_truth_mse'],
                'psnr_after': metrics['restored_vs_ground_truth_psnr'],
                'ssim_after': metrics['restored_vs_ground_truth_ssim'],
                'mse_delta': metrics['mse_improvement'],
                'psnr_delta': metrics['psnr_improvement'],
                'ssim_delta': metrics['ssim_improvement'],
                'operations_applied': "|".join(pipeline_result.operations_applied)
            }
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {pair.degraded_path}: {e}")
            failed_images += 1
            
    # Save detailed CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(output_csv, index=False)
    else:
        print("No successful results to save.")
        
    # Aggregate statistics
    agg_stats = aggregate_statistics(df, start_time, total_pairs, failed_images, mismatch_images, output_csv)
    return agg_stats

def aggregate_statistics(df: pd.DataFrame, start_time: float, total_pairs: int, 
                         failed: int, mismatch: int, output_csv: str) -> Dict[str, Any]:
    """
    Aggregates overall and category-specific statistics from the results DataFrame.
    """
    end_time = time.time()
    runtime = end_time - start_time
    
    stats = {
        'total_pairs_discovered': total_pairs,
        'evaluated_pairs': len(df),
        'failed_images': failed,
        'dimension_mismatches': mismatch,
        'runtime_seconds': runtime,
        'csv_path': output_csv,
        'overall': {},
        'by_category': {},
        'operations_count': {}
    }
    
    if df.empty:
        return stats
        
    # Helper to calculate group stats
    def calc_group_stats(group_df):
        return {
            'count': len(group_df),
            'mean_mse_before': group_df['mse_before'].mean(),
            'mean_mse_after': group_df['mse_after'].mean(),
            'mean_psnr_before': group_df['psnr_before'].mean(),
            'mean_psnr_after': group_df['psnr_after'].mean(),
            'mean_ssim_before': group_df['ssim_before'].mean(),
            'mean_ssim_after': group_df['ssim_after'].mean(),
            'mean_mse_delta': group_df['mse_delta'].mean(),
            'mean_psnr_delta': group_df['psnr_delta'].mean(),
            'mean_ssim_delta': group_df['ssim_delta'].mean(),
            'percent_mse_improved': (group_df['mse_delta'] > 0).mean() * 100,
            'percent_psnr_improved': (group_df['psnr_delta'] > 0).mean() * 100,
            'percent_ssim_improved': (group_df['ssim_delta'] > 0).mean() * 100,
        }
        
    # Overall stats
    stats['overall'] = calc_group_stats(df)
    
    # By category
    for cat in df['degradation_category'].unique():
        cat_df = df[df['degradation_category'] == cat]
        stats['by_category'][cat] = calc_group_stats(cat_df)
        
    # Operations statistics
    all_ops = []
    for ops_str in df['operations_applied']:
        all_ops.extend(ops_str.split('|'))
    stats['operations_count'] = dict(Counter(all_ops))
    
    return stats

def print_report(stats: Dict[str, Any]):
    """Pretty prints the aggregate statistics report."""
    print("\n" + "="*50)
    print("RESTORATION EVALUATION REPORT")
    print("="*50)
    print(f"Total pairs discovered:  {stats['total_pairs_discovered']}")
    print(f"Pairs successfully eval: {stats['evaluated_pairs']}")
    print(f"Failed images:           {stats['failed_images']}")
    print(f"Dimension mismatches:    {stats['dimension_mismatches']}")
    print(f"Runtime:                 {stats['runtime_seconds']:.2f} seconds")
    print(f"Results CSV:             {stats['csv_path']}")
    
    if 'overall' not in stats or 'count' not in stats['overall']:
        return
        
    print("\n--- OVERALL STATISTICS ---")
    o = stats['overall']
    print(f"MSE:  {o['mean_mse_before']:.2f} -> {o['mean_mse_after']:.2f} (Delta: {o['mean_mse_delta']:+.2f}, Improved: {o['percent_mse_improved']:.1f}%)")
    print(f"PSNR: {o['mean_psnr_before']:.2f} -> {o['mean_psnr_after']:.2f} (Delta: {o['mean_psnr_delta']:+.2f}, Improved: {o['percent_psnr_improved']:.1f}%)")
    print(f"SSIM: {o['mean_ssim_before']:.4f} -> {o['mean_ssim_after']:.4f} (Delta: {o['mean_ssim_delta']:+.4f}, Improved: {o['percent_ssim_improved']:.1f}%)")
    
    print("\n--- BY DEGRADATION CATEGORY ---")
    for cat, c in stats['by_category'].items():
        print(f"\n{cat} ({c['count']} images):")
        print(f"  MSE:  {c['mean_mse_before']:.2f} -> {c['mean_mse_after']:.2f} (Improved: {c['percent_mse_improved']:.1f}%)")
        print(f"  PSNR: {c['mean_psnr_before']:.2f} -> {c['mean_psnr_after']:.2f} (Improved: {c['percent_psnr_improved']:.1f}%)")
        print(f"  SSIM: {c['mean_ssim_before']:.4f} -> {c['mean_ssim_after']:.4f} (Improved: {c['percent_ssim_improved']:.1f}%)")
        
    print("\n--- OPERATION USAGE ---")
    for op, count in stats['operations_count'].items():
        print(f"  {op}: {count} ({count/stats['evaluated_pairs']*100:.1f}%)")
    print("==================================================\n")

if __name__ == '__main__':
    stats = run_evaluation()
    print_report(stats)
