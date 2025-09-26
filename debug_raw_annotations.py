#!/usr/bin/env python3
"""
Debug script to check raw annotation data before filtering.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_raw_annotations():
    """Check raw annotation data before any processing."""
    
    from mmengine import Config
    from mmyolo.registry import DATASETS
    import json
    
    # Load config
    config_path = 'test_minimal_config.py'
    cfg = Config.fromfile(config_path)
    
    print("=== Checking Raw Annotation Data ===")
    
    # Load the annotation file directly
    ann_file = os.path.join(cfg.data_root, cfg.train_dataloader.dataset.ann_file)
    with open(ann_file, 'r') as f:
        ann_data = json.load(f)
    
    print(f"Total images: {len(ann_data['images'])}")
    print(f"Total annotations: {len(ann_data['annotations'])}")
    print(f"Categories: {[cat['name'] for cat in ann_data['categories']]}")
    
    # Check first few annotations
    print("\n=== First 5 Annotations ===")
    for i, ann in enumerate(ann_data['annotations'][:5]):
        print(f"Annotation {i}:")
        print(f"  - ID: {ann['id']}")
        print(f"  - Image ID: {ann['image_id']}")
        print(f"  - Category ID: {ann['category_id']}")
        print(f"  - Bbox: {ann['bbox']}")
        print(f"  - Area: {ann['area']}")
        print(f"  - IsCrowd: {ann['iscrowd']}")
    
    # Check annotations for first few images
    print("\n=== Annotations per Image ===")
    image_ann_counts = {}
    for ann in ann_data['annotations']:
        img_id = ann['image_id']
        image_ann_counts[img_id] = image_ann_counts.get(img_id, 0) + 1
    
    # Show first 10 images and their annotation counts
    for i, img in enumerate(ann_data['images'][:10]):
        img_id = img['id']
        ann_count = image_ann_counts.get(img_id, 0)
        print(f"Image {i} (ID: {img_id}): {ann_count} annotations")
        if ann_count > 0:
            # Show annotations for this image
            img_anns = [ann for ann in ann_data['annotations'] if ann['image_id'] == img_id]
            for j, ann in enumerate(img_anns[:3]):  # Show first 3 annotations
                print(f"  - Ann {j}: cat={ann['category_id']}, bbox={ann['bbox']}, area={ann['area']}")
    
    # Check for potential filtering issues
    print("\n=== Checking for Filtering Issues ===")
    
    # Check bbox sizes
    bbox_areas = [ann['area'] for ann in ann_data['annotations']]
    print(f"Bbox areas - min: {min(bbox_areas):.2f}, max: {max(bbox_areas):.2f}, mean: {sum(bbox_areas)/len(bbox_areas):.2f}")
    
    # Check bbox coordinates
    bbox_widths = [ann['bbox'][2] for ann in ann_data['annotations']]
    bbox_heights = [ann['bbox'][3] for ann in ann_data['annotations']]
    print(f"Bbox widths - min: {min(bbox_widths):.2f}, max: {max(bbox_widths):.2f}")
    print(f"Bbox heights - min: {min(bbox_heights):.2f}, max: {max(bbox_heights):.2f}")
    
    # Check category IDs
    category_ids = [ann['category_id'] for ann in ann_data['annotations']]
    print(f"Category IDs: {set(category_ids)}")
    
    # Check for invalid bboxes
    invalid_bboxes = []
    for ann in ann_data['annotations']:
        bbox = ann['bbox']
        if bbox[2] <= 0 or bbox[3] <= 0:  # width or height <= 0
            invalid_bboxes.append(ann)
    
    print(f"Invalid bboxes (width or height <= 0): {len(invalid_bboxes)}")
    if invalid_bboxes:
        print("First few invalid bboxes:")
        for ann in invalid_bboxes[:3]:
            print(f"  - {ann}")

if __name__ == '__main__':
    debug_raw_annotations()
