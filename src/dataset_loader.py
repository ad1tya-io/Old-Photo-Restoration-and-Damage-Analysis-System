import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SyntheticImagePair:
    clean_path: str
    degraded_path: str
    source_id: str
    degradation_type: str
    degradation_components: List[str]


def parse_synthetic_filename(filename: str) -> Tuple[str, Optional[str], List[str]]:
    """
    Parses a synthetic dataset filename to extract source ID and degradation information.
    Clean example: lrp_img100.jpg -> ('lrp_img100', None, [])
    Degraded example: lrp_img100_(Noise_JPEG_Blur).jpg -> ('lrp_img100', 'Noise_JPEG_Blur', ['Noise', 'JPEG', 'Blur'])
    """
    base_name = os.path.splitext(filename)[0]
    
    # Match pattern like: lrp_img100_(Complex_All)
    match = re.match(r'^(.*?)_\((.*?)\)$', base_name)
    if match:
        source_id = match.group(1)
        degradation_type = match.group(2)
        # Split by underscores to get components
        components = [c for c in degradation_type.split('_') if c]
        return source_id, degradation_type, components
    
    # If no degradation pattern is found, assume it's a clean image
    return base_name, None, []


def load_image_rgb(path: str) -> np.ndarray:
    """
    Safely loads an image from the given path in RGB format.
    Raises FileNotFoundError if path doesn't exist.
    Raises ValueError if image is unreadable or corrupt.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {path}")
        
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load or corrupted image at path: {path}")
        
    # Convert to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unexpected number of channels ({img.shape[2]}) in image: {path}")
    else:
        raise ValueError(f"Unexpected image shape ({img.shape}) at path: {path}")
        
    return img


class DatasetLoader:
    def __init__(self, dataset_dir: str = 'dataset'):
        self.dataset_dir = dataset_dir
        self.clean_candidates_dir = os.path.join(dataset_dir, '01_Clean_Candidates_GT')
        self.damaged_testing_dir = os.path.join(dataset_dir, '02_Damaged_Testing_Set')
        self.synthetic_dir = os.path.join(dataset_dir, '03_Synthetic_Dataset')
        self.synthetic_clean_dir = os.path.join(self.synthetic_dir, 'Train_GT_Clean')
        self.synthetic_degraded_dir = os.path.join(self.synthetic_dir, 'Train_Input_Degraded')

    def load_clean_candidates(self) -> List[str]:
        if not os.path.exists(self.clean_candidates_dir):
            return []
        return [os.path.join(self.clean_candidates_dir, f) for f in os.listdir(self.clean_candidates_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def load_real_damaged(self) -> List[str]:
        if not os.path.exists(self.damaged_testing_dir):
            return []
        return [os.path.join(self.damaged_testing_dir, f) for f in os.listdir(self.damaged_testing_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def load_synthetic_pairs(self) -> List[SyntheticImagePair]:
        pairs = []
        if not os.path.exists(self.synthetic_clean_dir) or not os.path.exists(self.synthetic_degraded_dir):
            logger.warning("Synthetic dataset directories not found.")
            return pairs

        clean_files = {parse_synthetic_filename(f)[0]: f for f in os.listdir(self.synthetic_clean_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}

        for degraded_file in os.listdir(self.synthetic_degraded_dir):
            if not degraded_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
                
            source_id, degradation_type, components = parse_synthetic_filename(degraded_file)
            
            if source_id in clean_files:
                clean_path = os.path.join(self.synthetic_clean_dir, clean_files[source_id])
                degraded_path = os.path.join(self.synthetic_degraded_dir, degraded_file)
                
                pairs.append(SyntheticImagePair(
                    clean_path=clean_path,
                    degraded_path=degraded_path,
                    source_id=source_id,
                    degradation_type=degradation_type or "Unknown",
                    degradation_components=components
                ))
            else:
                logger.warning(f"No clean pair found for degraded image: {degraded_file}")
                
        return pairs

    def get_dataset_summary(self) -> Dict:
        summary = {
            'num_clean_candidates': len(self.load_clean_candidates()),
            'num_real_damaged': len(self.load_real_damaged()),
            'num_synthetic_pairs_confirmed': 0,
            'num_synthetic_clean': 0,
            'num_synthetic_degraded': 0,
            'missing_pair_count': 0,
            'degradation_distribution': {}
        }
        
        if os.path.exists(self.synthetic_clean_dir):
            summary['num_synthetic_clean'] = len([f for f in os.listdir(self.synthetic_clean_dir) 
                                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                                                 
        if os.path.exists(self.synthetic_degraded_dir):
            degraded_files = [f for f in os.listdir(self.synthetic_degraded_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            summary['num_synthetic_degraded'] = len(degraded_files)
            
            pairs = self.load_synthetic_pairs()
            summary['num_synthetic_pairs_confirmed'] = len(pairs)
            summary['missing_pair_count'] = len(degraded_files) - len(pairs)
            
            dist = {}
            for p in pairs:
                dist[p.degradation_type] = dist.get(p.degradation_type, 0) + 1
            summary['degradation_distribution'] = dist
            
        return summary
