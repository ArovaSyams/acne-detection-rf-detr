"""Quick utility script to inspect split counts of the COCO dataset."""
import json
from pathlib import Path

dataset_dir = Path('dataset_coco')
splits = ['train', 'val', 'test']
for split in splits:
    anno_path = dataset_dir / 'annotations' / f'instances_{split}.json'
    if anno_path.exists():
        with open(anno_path) as f:
            d = json.load(f)
            print(f'{split.upper()}: {len(d["images"])} images, {len(d["annotations"])} boxes')
