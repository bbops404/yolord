#!/usr/bin/env python3
"""
Debug script to check dataset loading process in detail.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_dataset_loading_deep():
    """Debug the dataset loading process in detail."""
    
    from mmengine import Config
    from mmyolo.registry import DATASETS
    import json
    
    # Load config
    config_path = 'test_minimal_config.py'
    cfg = Config.fromfile(config_path)
    
    print("=== Debugging Dataset Loading Process ===")
    
    # Create dataset without pipeline
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset_cfg['pipeline'] = []  # No pipeline
    
    print(f"Dataset config: {dataset_cfg}")
    
    # Let's check what happens during dataset creation
    print(f"\n=== Creating Dataset ===")
    dataset = DATASETS.build(dataset_cfg)
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Check the data_list directly
    print(f"\n=== Checking data_list ===")
    print(f"Number of items in data_list: {len(dataset.data_list)}")
    
    if dataset.data_list:
        first_item = dataset.data_list[0]
        print(f"First data_list item keys: {list(first_item.keys())}")
        print(f"First data_list item: {first_item}")
    else:
        print("ERROR: data_list is empty! All samples were filtered out.")
    
    # Check if there's a filter_data method and what it does
    print(f"\n=== Checking filter_data method ===")
    if hasattr(dataset, 'filter_data'):
        print(f"Dataset has filter_data method")
        # Let's see what the original data_list looks like before filtering
        if hasattr(dataset, '_data_list_before_filter'):
            print(f"Data before filtering: {len(dataset._data_list_before_filter)} items")
        else:
            print("No _data_list_before_filter attribute")
    
    # Let's manually check the annotation file
    print(f"\n=== Manual Annotation Check ===")
    ann_file = os.path.join(cfg.data_root, cfg.train_dataloader.dataset.ann_file)
    with open(ann_file, 'r') as f:
        ann_data = json.load(f)
    
    print(f"Annotation file has {len(ann_data['images'])} images and {len(ann_data['annotations'])} annotations")
    
    # Check the first image in the annotation file
    first_img = ann_data['images'][0]
    first_img_id = first_img['id']
    print(f"First image in annotation file: ID={first_img_id}, file={first_img['file_name']}")
    
    # Find annotations for this image
    img_anns = [ann for ann in ann_data['annotations'] if ann['image_id'] == first_img_id]
    print(f"Annotations for first image: {len(img_anns)}")
    for i, ann in enumerate(img_anns[:3]):
        print(f"  Annotation {i}: {ann}")
    
    # Let's check if there's a mismatch in image IDs
    print(f"\n=== Checking Image ID Mismatch ===")
    if dataset.data_list:
        dataset_img_ids = set(item['img_id'] for item in dataset.data_list[:10])
    else:
        dataset_img_ids = set()
    ann_img_ids = set(img['id'] for img in ann_data['images'][:10])
    
    print(f"First 10 dataset image IDs: {sorted(dataset_img_ids)}")
    print(f"First 10 annotation image IDs: {sorted(ann_img_ids)}")
    print(f"Common IDs: {dataset_img_ids.intersection(ann_img_ids)}")
    
    # Check if the dataset is using a different annotation file
    print(f"\n=== Checking Dataset Annotation File ===")
    if hasattr(dataset, 'ann_file'):
        print(f"Dataset ann_file: {dataset.ann_file}")
    if hasattr(dataset, 'data_root'):
        print(f"Dataset data_root: {dataset.data_root}")
    
    # Let's check the dataset's internal annotation loading
    print(f"\n=== Checking Dataset Internal State ===")
    if hasattr(dataset, 'coco'):
        print(f"Dataset has coco object: {type(dataset.coco)}")
        if hasattr(dataset.coco, 'anns'):
            print(f"Number of annotations in coco: {len(dataset.coco.anns)}")
    
    # Check if there's a load_data_list method
    print(f"\n=== Checking load_data_list ===")
    if hasattr(dataset, 'load_data_list'):
        print(f"Dataset has load_data_list method")
        # Let's see what it returns
        try:
            raw_data_list = dataset.load_data_list()
            print(f"Raw data_list length: {len(raw_data_list)}")
            if raw_data_list:
                first_raw = raw_data_list[0]
                print(f"First raw item keys: {list(first_raw.keys())}")
                print(f"First raw item instances: {len(first_raw.get('instances', []))}")
                if first_raw.get('instances'):
                    print(f"First instance: {first_raw['instances'][0]}")
        except Exception as e:
            print(f"Error calling load_data_list: {e}")
    
    # Let's try to understand why filtering is happening
    print(f"\n=== Debugging Filtering Issue ===")
    
    # Check the filter_cfg
    print(f"Filter config: {dataset_cfg.get('filter_cfg', 'No filter_cfg')}")
    
    # Let's try to manually call load_data_list and see what happens
    if hasattr(dataset, 'load_data_list'):
        try:
            print(f"Calling load_data_list manually...")
            raw_data = dataset.load_data_list()
            print(f"Raw data length: {len(raw_data)}")
            
            if raw_data:
                print(f"First raw item: {raw_data[0]}")
                
                # Check why instances is empty
                print(f"\n=== Investigating Empty Instances ===")
                first_item = raw_data[0]
                print(f"First item instances: {first_item.get('instances', 'No instances key')}")
                
                # Let's check if the dataset has a coco object and if it can find annotations
                if hasattr(dataset, 'coco') and dataset.coco:
                    print(f"COCO object has {len(dataset.coco.anns)} annotations")
                    
                    # Try to get annotations for the first image
                    img_id = first_item['img_id']
                    ann_ids = dataset.coco.getAnnIds(imgIds=[img_id])
                    print(f"Annotation IDs for image {img_id}: {ann_ids}")
                    
                    if ann_ids:
                        anns = dataset.coco.loadAnns(ann_ids)
                        print(f"Loaded annotations: {len(anns)}")
                        print(f"First annotation: {anns[0] if anns else 'None'}")
                        
                        # Let's see what the dataset's load_data_list method should be doing
                        print(f"\n=== Checking Dataset's load_data_list Implementation ===")
                        print(f"Dataset type: {type(dataset)}")
                        print(f"Dataset MRO: {[cls.__name__ for cls in type(dataset).__mro__]}")
                        
                        # Check if there's a method that converts COCO annotations to instances
                        if hasattr(dataset, '_parse_ann_info'):
                            print(f"Dataset has _parse_ann_info method")
                            try:
                                # Try to parse the first annotation
                                parsed_ann = dataset._parse_ann_info(anns[0])
                                print(f"Parsed annotation: {parsed_ann}")
                            except Exception as e:
                                print(f"Error parsing annotation: {e}")
                        else:
                            print(f"Dataset does not have _parse_ann_info method")
                            
                        # Check if there's a method that loads instances
                        if hasattr(dataset, '_load_instances'):
                            print(f"Dataset has _load_instances method")
                        else:
                            print(f"Dataset does not have _load_instances method")
                            
                    else:
                        print(f"No annotation IDs found for image {img_id}")
                else:
                    print(f"Dataset does not have COCO object")
                    
                    # Let's check what attributes the dataset has
                    print(f"\n=== Dataset Attributes ===")
                    print(f"Dataset attributes: {[attr for attr in dir(dataset) if not attr.startswith('_')]}")
                    
                    # Check if there's a method to load COCO
                    if hasattr(dataset, 'load_annotations'):
                        print(f"Dataset has load_annotations method")
                    else:
                        print(f"Dataset does not have load_annotations method")
                        
                    if hasattr(dataset, '_load_annotations'):
                        print(f"Dataset has _load_annotations method")
                    else:
                        print(f"Dataset does not have _load_annotations method")
                        
                    # Check the dataset's initialization
                    print(f"\n=== Dataset Initialization Check ===")
                    print(f"Dataset ann_file: {getattr(dataset, 'ann_file', 'No ann_file')}")
                    print(f"Dataset data_root: {getattr(dataset, 'data_root', 'No data_root')}")
                    print(f"Dataset test_mode: {getattr(dataset, 'test_mode', 'No test_mode')}")
                    
                    # Let's try to manually create a COCO object
                    print(f"\n=== Manual COCO Object Creation ===")
                    try:
                        from pycocotools.coco import COCO
                        ann_file_path = os.path.join(cfg.data_root, cfg.train_dataloader.dataset.ann_file)
                        print(f"Trying to create COCO object from: {ann_file_path}")
                        coco = COCO(ann_file_path)
                        print(f"COCO object created successfully with {len(coco.anns)} annotations")
                        
                        # Check if we can get annotations for the first image
                        img_id = first_item['img_id']
                        ann_ids = coco.getAnnIds(imgIds=[img_id])
                        print(f"Annotation IDs for image {img_id}: {ann_ids}")
                        
                        if ann_ids:
                            anns = coco.loadAnns(ann_ids)
                            print(f"Loaded annotations: {len(anns)}")
                            print(f"First annotation: {anns[0] if anns else 'None'}")
                        else:
                            print(f"No annotation IDs found for image {img_id}")
                            
                    except Exception as e:
                        print(f"Error creating COCO object: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Now let's see what filter_data does to this
                print(f"\n=== Testing filter_data ===")
                # filter_data is a method that doesn't take arguments
                dataset.data_list = raw_data  # Set the data_list first
                filtered_data = dataset.filter_data()
                print(f"Filtered data length: {len(filtered_data)}")
                
                if len(filtered_data) == 0:
                    print("ERROR: filter_data removed all samples!")
                    print("This is why data_list is empty.")
                else:
                    print(f"Filtered data first item: {filtered_data[0]}")
                    
        except Exception as e:
            print(f"Error in manual filtering test: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    debug_dataset_loading_deep()
