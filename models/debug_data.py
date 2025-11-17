# models/debug_data.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils import load_image_data

def debug_data():
    splits = load_image_data()
    
    print("\n=== DATA DEBUG INFO ===")
    for split_name, split_data in splits.items():
        print(f"\n{split_name.upper()} SPLIT:")
        print(f"Total images: {len(split_data['images'])}")
        unique_labels = set(split_data['labels'])
        print(f"Unique labels: {unique_labels}")
        
        # Count each class
        from collections import Counter
        label_counts = Counter(split_data['labels'])
        for label, count in label_counts.items():
            print(f"  {label}: {count} images")

if __name__ == "__main__":
    debug_data()    