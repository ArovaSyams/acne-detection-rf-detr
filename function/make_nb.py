import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """# COCO Dataset Summary
This notebook reports statistics for the generated COCO dataset, including image counts, bounding box annotations, and target classes."""

code = """import json
from pathlib import Path

dataset_dir = Path("dataset_coco")

splits = ["train", "val", "test"]
total_images = 0
total_annotations = 0

print(f"=== DATASET SUMMARY: {dataset_dir.name} ===")

for split in splits:
    anno_file = dataset_dir / "annotations" / f"instances_{split}.json"
    if not anno_file.exists():
        print(f"\\nSplit '{split}' not found.")
        continue
        
    with open(anno_file, 'r') as f:
        coco_data = json.load(f)
        
    num_images = len(coco_data['images'])
    num_anns = len(coco_data['annotations'])
    num_classes = len(coco_data['categories'])
    categories = [cat['name'] for cat in coco_data['categories']]
    
    total_images += num_images
    total_annotations += num_anns
    
    print(f"\\n[Split: {split.upper()}]")
    print(f"- Image Count       : {num_images}")
    print(f"- BBox Annotations  : {num_anns}")
    if split == "train":
        print(f"- Class Count       : {num_classes} ({', '.join(categories)})")

print("\\n=== OVERALL TOTALS ===")
print(f"Total Images      : {total_images}")
print(f"Total Annotations : {total_annotations}")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code)
]

nbf.write(nb, 'dataset_summary.ipynb')
print("Notebook created successfully.")
