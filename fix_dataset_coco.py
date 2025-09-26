#!/usr/bin/env python3
"""
Fix the dataset by manually adding the COCO object.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def fix_dataset_coco():
    """Fix the dataset by manually adding the COCO object."""
    
    from mmengine import Config
    from mmyolo.registry import DATASETS
    from pycocotools.coco import COCO
    
    # Load config
    config_path = 'test_minimal_config.py'
    cfg = Config.fromfile(config_path)
    
    print("=== Fixing Dataset COCO Object ===")
    
    # Create dataset without pipeline
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset_cfg['pipeline'] = []  # No pipeline
    dataset = DATASETS.build(dataset_cfg)
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Manually add COCO object
    ann_file_path = os.path.join(cfg.data_root, cfg.train_dataloader.dataset.ann_file)
    print(f"Creating COCO object from: {ann_file_path}")
    
    try:
        coco = COCO(ann_file_path)
        dataset.coco = coco
        print(f"COCO object added successfully with {len(coco.anns)} annotations")
        
        # Now let's test if the dataset works correctly
        print(f"\n=== Testing Fixed Dataset ===")
        
        # Get raw data
        raw_data = dataset[0]
        print(f"Raw data keys: {list(raw_data.keys())}")
        print(f"Instances: {len(raw_data.get('instances', []))}")
        
        if raw_data.get('instances'):
            print(f"SUCCESS: Dataset now has instances!")
            print(f"First instance: {raw_data['instances'][0]}")
        else:
            print(f"Still no instances. Let's check what's happening...")
            
            # Check if the dataset's load_data_list method is working
            print(f"Calling load_data_list again...")
            raw_data_list = dataset.load_data_list()
            print(f"Raw data_list length: {len(raw_data_list)}")
            
            if raw_data_list:
                first_item = raw_data_list[0]
                print(f"First item instances: {len(first_item.get('instances', []))}")
                
                if first_item.get('instances'):
                    print(f"SUCCESS: load_data_list now returns instances!")
                    print(f"First instance: {first_item['instances'][0]}")
                else:
                    print(f"load_data_list still returns empty instances")
                    
                    # Let's manually populate the instances
                    print(f"Manually populating instances...")
                    img_id = first_item['img_id']
                    ann_ids = coco.getAnnIds(imgIds=[img_id])
                    
                    if ann_ids:
                        anns = coco.loadAnns(ann_ids)
                        print(f"Found {len(anns)} annotations for image {img_id}")
                        
                        # Convert COCO annotations to instances format
                        instances = []
                        for ann in anns:
                            instance = {
                                'bbox': ann['bbox'],
                                'bbox_label': ann['category_id'],
                                'ignore_flag': ann.get('iscrowd', 0)
                            }
                            instances.append(instance)
                        
                        first_item['instances'] = instances
                        print(f"Manually added {len(instances)} instances")
                        print(f"First instance: {instances[0]}")
                        
                        # Now test the pipeline
                        print(f"\n=== Testing Pipeline with Fixed Data ===")
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
                            print(f"GT bboxes: {data_after_ann['gt_bboxes']}")
                        else:
                            print(f"Still no gt_bboxes after LoadAnnotations!")
                            
    except Exception as e:
        print(f"Error creating COCO object: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    fix_dataset_coco()
