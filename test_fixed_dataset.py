#!/usr/bin/env python3
"""
Test the fixed dataset to ensure annotations are loaded correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fixed_dataset():
    """Test the fixed dataset."""
    
    from mmengine import Config
    from mmyolo.registry import DATASETS
    from mmengine.registry import TRANSFORMS
    from mmyolo.datasets.transforms import LoadAnnotations
    
    # Load config
    config_path = 'test_minimal_config.py'
    cfg = Config.fromfile(config_path)
    
    print("=== Testing Fixed Dataset ===")
    
    # Create dataset with pipeline
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset = DATASETS.build(dataset_cfg)
    
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
    test_fixed_dataset()
