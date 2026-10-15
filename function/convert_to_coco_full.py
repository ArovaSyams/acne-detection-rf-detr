import os
import json
import shutil
import cv2
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_coco_structure(output_dir):
    dirs = {
        "train": os.path.join(output_dir, "train"),
        "val": os.path.join(output_dir, "val"),
        "test": os.path.join(output_dir, "test"),
        "annotations": os.path.join(output_dir, "annotations"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def init_coco_dict():
    return {
        "info": {
            "description": "Acne04v3 Dataset Full Parents - Detection Only",
            "version": "3.0",
            "year": 2026,
            "contributor": "Converted",
            "date_created": "2026-07-28"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "acne",
                "supercategory": "lesion"
            }
        ]
    }

def main():
    base_dir = Path("/")
    dataset_dir = base_dir / "acne04v3" / "acne04v3_dataset"
    
    csv_path = dataset_dir / "manifests" / "all_records.csv"
    if not csv_path.exists():
        print(f"File {csv_path} not found.")
        return
        
    print("Reading CSV manifest...")
    df = pd.read_csv(csv_path)
    
    # Group by original full-resolution parent images
    parent_images = df['parent_path'].unique()
    
    print(f"Found {len(parent_images)} unique full parent images in all_records.csv")
    
    # Split dataset into train (70%), val (15%), test (15%)
    train_imgs, temp_imgs = train_test_split(parent_images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }
    
    output_dir = base_dir / "dataset_coco"
    print(f"Generating COCO dataset structure at {output_dir}")
    dirs = create_coco_structure(output_dir)
    
    annotation_id = 1
    
    for split_name, img_paths in splits.items():
        print(f"Processing split '{split_name}' ({len(img_paths)} images)...")
        coco_dict = init_coco_dict()
        
        for image_id, p_path in enumerate(tqdm(img_paths, desc=split_name)):
            src_img_path = dataset_dir / p_path
            
            if not src_img_path.exists():
                print(f"Warning: {src_img_path} not found!")
                continue
                
            img_filename = src_img_path.name
            dst_img_path = Path(dirs[split_name]) / img_filename
            
            # Copy image from source to target split directory
            shutil.copy2(src_img_path, dst_img_path)
            
            # Retrieve image dimensions
            img = cv2.imread(str(src_img_path))
            if img is None:
                continue
            height, width = img.shape[:2]
            
            coco_dict["images"].append({
                "id": image_id + 1,
                "file_name": img_filename,
                "width": width,
                "height": height
            })
            
            # Extract all acne bounding box annotations for this image
            annots = df[df['parent_path'] == p_path]
            for _, row in annots.iterrows():
                x = float(row['bbox_x'])
                y = float(row['bbox_y'])
                w = float(row['bbox_width'])
                h = float(row['bbox_height'])
                
                if w > 0 and h > 0:
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id + 1,
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1
                    
        # Save COCO JSON annotations
        json_path = os.path.join(dirs["annotations"], f"instances_{split_name}.json")
        with open(json_path, 'w') as f:
            json.dump(coco_dict, f)
            
    print("Full image COCO dataset conversion completed successfully!")

if __name__ == "__main__":
    main()
