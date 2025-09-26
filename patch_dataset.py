#!/usr/bin/env python3
"""
Patch the existing dataset to fix the COCO loading issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def patch_dataset():
    """Patch the existing dataset to fix COCO loading."""
    
    from mmengine import Config
    from mmyolo.registry import DATASETS
    from pycocotools.coco import COCO
    
    # Load config
    config_path = 'test_minimal_config.py'
    cfg = Config.fromfile(config_path)
    
    print("=== Patching Dataset ===")
    
    # Create dataset without pipeline
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset_cfg['pipeline'] = []  # No pipeline
    dataset = DATASETS.build(dataset_cfg)
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Manually add COCO object and fix the data_list
    ann_file_path = os.path.join(cfg.data_root, cfg.train_dataloader.dataset.ann_file)
    print(f"Creating COCO object from: {ann_file_path}")
    
    try:
        coco = COCO(ann_file_path)
        dataset.coco = coco
        print(f"COCO object added successfully with {len(coco.anns)} annotations")
        
        # Get the raw data_list before filtering
        print(f"Getting raw data_list...")
        raw_data_list = dataset.load_data_list()
        print(f"Raw data_list length: {len(raw_data_list)}")
        
        # Fix the raw data_list by populating instances
        print(f"Fixing raw data_list...")
        for data_item in raw_data_list:
            if 'instances' not in data_item or not data_item['instances']:
                img_id = data_item['img_id']
                ann_ids = coco.getAnnIds(imgIds=[img_id])
                
                if ann_ids:
                    anns = coco.loadAnns(ann_ids)
                    instances = []
                    for ann in anns:
                        instance = {
                            'bbox': ann['bbox'],
                            'bbox_label': ann['category_id'],
                            'ignore_flag': ann.get('iscrowd', 0)
                        }
                        instances.append(instance)
                    data_item['instances'] = instances
        
        # Replace the dataset's data_list with our fixed version
        dataset.data_list = raw_data_list
        print(f"Replaced dataset data_list with fixed version")
        
        print(f"Data_list fixed! Now testing...")
        
        # Check if data_list is empty
        if len(dataset.data_list) == 0:
            print(f"❌ Data_list is empty! All samples were filtered out.")
            print(f"This is why the dataset has no samples.")
            return
        
        # Test the patched data_list directly
        print(f"\n=== Testing Patched Data List ===")
        first_item = dataset.data_list[0]
        print(f"First item keys: {list(first_item.keys())}")
        print(f"Instances: {len(first_item.get('instances', []))}")
        
        if first_item.get('instances'):
            print(f"SUCCESS: Data list has {len(first_item['instances'])} instances!")
            print(f"First instance: {first_item['instances'][0]}")
            
            # Test the pipeline with the patched data
            print(f"\n=== Testing Pipeline with Patched Data ===")
            from mmengine.registry import TRANSFORMS
            from mmyolo.datasets.transforms import LoadAnnotations
            
            # Load image
            load_img = TRANSFORMS.build(dict(type='LoadImageFromFile', backend_args=None))
            data_after_img = load_img(first_item.copy())
            
            # Load annotations
            load_ann = LoadAnnotations(with_bbox=True)
            data_after_ann = load_ann(data_after_img.copy())
            
            print(f"After LoadAnnotations keys: {list(data_after_ann.keys())}")
            
            if 'gt_bboxes' in data_after_ann:
                print(f"SUCCESS: GT bboxes shape: {data_after_ann['gt_bboxes'].shape}")
                print(f"SUCCESS: GT labels shape: {data_after_ann['gt_bboxes_labels'].shape}")
                print(f"✅ Pipeline is working! Losses should now be non-zero.")
                
                # Now test the dataset directly
                print(f"\n=== Testing Dataset with Patched Data ===")
                # Override the dataset's get_data_info method to use our patched data
                original_get_data_info = dataset.get_data_info
                def patched_get_data_info(idx):
                    return dataset.data_list[idx]
                dataset.get_data_info = patched_get_data_info
                
                # Test the dataset
                sample = dataset[0]
                print(f"Dataset sample keys: {list(sample.keys())}")
                print(f"Dataset sample instances: {len(sample.get('instances', []))}")
                
                if sample.get('instances'):
                    print(f"✅ Dataset is now working correctly!")
                else:
                    print(f"❌ Dataset still not working")
            else:
                print(f"❌ Still no gt_bboxes after LoadAnnotations!")
        else:
            print(f"❌ Still no instances in data_list")
            
    except Exception as e:
        print(f"Error patching dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    patch_dataset()
