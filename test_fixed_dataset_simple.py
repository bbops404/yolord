#!/usr/bin/env python3
"""
Test the fixed dataset with a simple approach.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fixed_dataset_simple():
    """Test the fixed dataset with a simple approach."""
    
    # Import the fixed dataset module
    sys.path.append(os.path.join(os.path.dirname(__file__), 'projects', 'yolo-rd'))
    from fixed_coco_dataset import FixedYOLOv5CocoDataset
    
    from mmengine.registry import TRANSFORMS
    from mmyolo.datasets.transforms import LoadAnnotations
    
    print("=== Testing Fixed Dataset (Simple) ===")
    
    # Create dataset directly
    dataset_cfg = {
        'type': 'FixedYOLOv5CocoDataset',
        'data_root': '/home/ubuntu/yolord/data/RDD2022_Japan/',
        'ann_file': 'annotations/train_fixed.json',
        'data_prefix': {'img': 'images/train/'},
        'filter_cfg': {'filter_empty_gt': False, 'min_size': 0},
        'pipeline': [
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
            dict(type='mmdet.PackDetInputs', 
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ],
        'test_mode': False,
        'lazy_init': False
    }
    
    dataset = FixedYOLOv5CocoDataset(**dataset_cfg)
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test the first sample
    print(f"\n=== Testing First Sample ===")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    
    if 'data_samples' in sample:
        data_sample = sample['data_samples']
        if hasattr(data_sample, 'gt_instances'):
            gt_instances = data_sample.gt_instances
            print(f"SUCCESS: GT bboxes shape: {gt_instances.bboxes.shape}")
            print(f"SUCCESS: GT labels shape: {gt_instances.labels.shape}")
            print(f"Number of annotations: {len(gt_instances.bboxes)}")
            
            if len(gt_instances.bboxes) > 0:
                print(f"First bbox: {gt_instances.bboxes[0]}")
                print(f"First label: {gt_instances.labels[0]}")
                print(f"✅ Dataset is working correctly! Losses should now be non-zero.")
            else:
                print(f"❌ Still no annotations found.")
        else:
            print(f"❌ No gt_instances in data_sample")
    else:
        print(f"❌ No data_samples in sample")
    
    # Test a few more samples
    print(f"\n=== Testing Multiple Samples ===")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        if 'data_samples' in sample:
            data_sample = sample['data_samples']
            if hasattr(data_sample, 'gt_instances'):
                gt_instances = data_sample.gt_instances
                print(f"Sample {i}: {len(gt_instances.bboxes)} annotations")
            else:
                print(f"Sample {i}: No gt_instances")
        else:
            print(f"Sample {i}: No data_samples")

if __name__ == '__main__':
    test_fixed_dataset_simple()
