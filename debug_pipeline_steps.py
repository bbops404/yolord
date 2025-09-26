#!/usr/bin/env python3
"""
Debug script to test pipeline steps individually to find where annotations are lost.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_pipeline_steps():
    """Test each pipeline step individually."""
    
    from mmengine import Config
    from mmyolo.registry import DATASETS
    from mmyolo.datasets import YOLOv5CocoDataset
    import json
    
    # Load config
    config_path = 'test_minimal_config.py'
    cfg = Config.fromfile(config_path)
    
    print("=== Testing Pipeline Steps Individually ===")
    
    # Create dataset without pipeline first
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset_cfg['pipeline'] = []  # No pipeline
    dataset = DATASETS.build(dataset_cfg)
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Get raw data for first sample
    print("\n=== Raw Data (No Pipeline) ===")
    raw_data = dataset[0]
    print(f"Raw data keys: {list(raw_data.keys())}")
    
    if 'gt_bboxes' in raw_data:
        print(f"Raw GT bboxes: {raw_data['gt_bboxes'].shape}")
        print(f"Raw GT labels: {raw_data['gt_bboxes_labels'].shape}")
    else:
        print("No gt_bboxes in raw data")
    
    # Test individual pipeline steps using registry
    from mmengine.registry import TRANSFORMS
    from mmyolo.datasets.transforms import LoadAnnotations
    
    print("\n=== Step 1: LoadImageFromFile ===")
    load_img = TRANSFORMS.build(dict(type='LoadImageFromFile', backend_args=None))
    data_after_img = load_img(raw_data.copy())
    print(f"After LoadImageFromFile keys: {list(data_after_img.keys())}")
    
    print("\n=== Step 2: LoadAnnotations ===")
    load_ann = LoadAnnotations(with_bbox=True)
    data_after_ann = load_ann(data_after_img.copy())
    print(f"After LoadAnnotations keys: {list(data_after_ann.keys())}")
    
    if 'gt_bboxes' in data_after_ann:
        print(f"GT bboxes after LoadAnnotations: {data_after_ann['gt_bboxes'].shape}")
        print(f"GT labels after LoadAnnotations: {data_after_ann['gt_bboxes_labels'].shape}")
        if len(data_after_ann['gt_bboxes']) > 0:
            print(f"First bbox: {data_after_ann['gt_bboxes'][0]}")
            print(f"First label: {data_after_ann['gt_bboxes_labels'][0]}")
    else:
        print("No gt_bboxes after LoadAnnotations!")
    
    print("\n=== Step 3: Resize ===")
    resize = TRANSFORMS.build(dict(type='mmdet.Resize', scale=cfg.img_scale, keep_ratio=True))
    data_after_resize = resize(data_after_ann.copy())
    print(f"After Resize keys: {list(data_after_resize.keys())}")
    
    if 'gt_bboxes' in data_after_resize:
        print(f"GT bboxes after Resize: {data_after_resize['gt_bboxes'].shape}")
        print(f"GT labels after Resize: {data_after_resize['gt_bboxes_labels'].shape}")
        if len(data_after_resize['gt_bboxes']) > 0:
            print(f"First bbox: {data_after_resize['gt_bboxes'][0]}")
            print(f"First label: {data_after_resize['gt_bboxes_labels'][0]}")
    else:
        print("No gt_bboxes after Resize!")
    
    print("\n=== Step 4: PackDetInputs ===")
    pack = TRANSFORMS.build(dict(type='mmdet.PackDetInputs', 
                                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')))
    data_after_pack = pack(data_after_resize.copy())
    print(f"After PackDetInputs keys: {list(data_after_pack.keys())}")
    
    if 'data_samples' in data_after_pack:
        data_sample = data_after_pack['data_samples']
        if hasattr(data_sample, 'gt_instances'):
            gt_instances = data_sample.gt_instances
            print(f"Final GT bboxes: {gt_instances.bboxes.shape}")
            print(f"Final GT labels: {gt_instances.labels.shape}")
            if len(gt_instances.bboxes) > 0:
                print(f"First final bbox: {gt_instances.bboxes[0]}")
                print(f"First final label: {gt_instances.labels[0]}")
        else:
            print("No gt_instances in final data_sample!")
    else:
        print("No data_samples in final data!")

if __name__ == '__main__':
    debug_pipeline_steps()
