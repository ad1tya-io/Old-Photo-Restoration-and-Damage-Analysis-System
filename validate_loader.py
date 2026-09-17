import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.dataset_loader import DatasetLoader
import json

def run_validation():
    print("Initializing DatasetLoader...")
    loader = DatasetLoader(dataset_dir="dataset")
    
    print("\n--- Dataset Summary ---")
    summary = loader.get_dataset_summary()
    print(json.dumps(summary, indent=2))
    
    print("\n--- Example Synthetic Pairings ---")
    pairs = loader.load_synthetic_pairs()
    for i, pair in enumerate(pairs[:5]):
        print(f"\nPair {i+1}:")
        print(f"  Clean: {os.path.basename(pair.clean_path)}")
        print(f"  Degraded: {os.path.basename(pair.degraded_path)}")
        print(f"  Source ID: {pair.source_id}")
        print(f"  Degradation Type: {pair.degradation_type}")
        print(f"  Components: {pair.degradation_components}")

if __name__ == "__main__":
    run_validation()
