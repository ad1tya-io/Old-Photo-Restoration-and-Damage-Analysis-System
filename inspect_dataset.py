import os
import cv2
import pandas as pd
import json

def get_dir_stats(path):
    stats = {
        'num_files': 0,
        'extensions': set(),
        'total_size': 0,
        'channels': set(),
        'dtypes': set(),
        'corrupted': 0,
        'dimensions': set(),
        'examples': []
    }
    
    if not os.path.exists(path):
        stats['extensions'] = list(stats['extensions'])
        stats['channels'] = list(stats['channels'])
        stats['dtypes'] = list(stats['dtypes'])
        stats['dimensions'] = list(stats['dimensions'])
        return stats
        
    for root, _, files in os.walk(path):
        for f in files:
            stats['num_files'] += 1
            file_path = os.path.join(root, f)
            stats['total_size'] += os.path.getsize(file_path)
            ext = os.path.splitext(f)[1].lower()
            stats['extensions'].add(ext)
            
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    stats['corrupted'] += 1
                else:
                    h, w = img.shape[:2]
                    stats['dimensions'].add((h, w))
                    if len(img.shape) == 2:
                        stats['channels'].add('grayscale')
                    elif len(img.shape) == 3:
                        if img.shape[2] == 3:
                            stats['channels'].add('RGB')
                        elif img.shape[2] == 4:
                            stats['channels'].add('RGBA')
                    stats['dtypes'].add(str(img.dtype))
                    
                    if len(stats['examples']) < 5:
                        stats['examples'].append(f)
                        
    stats['extensions'] = list(stats['extensions'])
    stats['channels'] = list(stats['channels'])
    stats['dtypes'] = list(stats['dtypes'])
    stats['dimensions'] = list(stats['dimensions'])
    return stats

def main():
    dataset_path = 'dataset'
    
    res = {}
    
    # Try actual folders found
    actual_folders = []
    if os.path.exists(dataset_path):
        actual_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
    for folder in actual_folders:
        folder_path = os.path.join(dataset_path, folder)
        res[folder] = get_dir_stats(folder_path)
        
        # for synthetic, list subdirs
        if 'synthetic' in folder.lower():
            subdirs = []
            for d in os.listdir(folder_path):
                dp = os.path.join(folder_path, d)
                if os.path.isdir(dp):
                    subdirs.append(d)
                    # Let's get 3 example filenames from this subdir
                    sfiles = os.listdir(dp)
                    sfiles = [f for f in sfiles if os.path.isfile(os.path.join(dp, f))]
                    res[folder + "_" + d + "_examples"] = sfiles[:3]
            res[folder + '_subdirs'] = subdirs
            
    # Analyze CSV
    csv_path = os.path.join(dataset_path, 'image_classification_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        res['csv'] = {
            'rows': len(df),
            'cols': len(df.columns),
            'col_names': list(df.columns),
            'dtypes': {c: str(t) for c, t in df.dtypes.items()},
            'head': df.head(5).to_dict(orient='records')
        }
    
    print("--- DATASET STATS ---")
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
