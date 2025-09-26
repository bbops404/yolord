#!/usr/bin/env python3
"""
Apply the dataset fix to the main codebase.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def apply_dataset_fix():
    """Apply the dataset fix to the main codebase."""
    
    print("=== Applying Dataset Fix to Main Codebase ===")
    
    # Path to the YOLOv5CocoDataset file
    dataset_file = 'mmyolo/datasets/yolov5_coco.py'
    
    print(f"Modifying {dataset_file}...")
    
    # Read the current file
    with open(dataset_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'def _ensure_coco_loaded(self):' in content:
        print("✅ Dataset is already patched!")
        return
    
    # Create the patch
    patch = '''
    def _ensure_coco_loaded(self):
        """Ensure COCO object is loaded and instances are populated."""
        if not hasattr(self, 'coco') or self.coco is None:
            # Load COCO object
            ann_file_path = os.path.join(self.data_root, self.ann_file)
            self.coco = COCO(ann_file_path)
            print(f"Loaded COCO object with {len(self.coco.anns)} annotations")

    def load_data_list(self):
        """Load data list and ensure instances are populated."""
        # Call parent method
        data_list = super().load_data_list()
        
        # Ensure COCO object is loaded
        self._ensure_coco_loaded()
        
        # Populate instances for each data item
        for data_item in data_list:
            if 'instances' not in data_item or not data_item['instances']:
                img_id = data_item['img_id']
                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                
                if ann_ids:
                    anns = self.coco.loadAnns(ann_ids)
                    instances = []
                    for ann in anns:
                        instance = {
                            'bbox': ann['bbox'],
                            'bbox_label': ann['category_id'],
                            'ignore_flag': ann.get('iscrowd', 0)
                        }
                        instances.append(instance)
                    data_item['instances'] = instances
        
        return data_list
'''
    
    # Add the necessary imports
    if 'from pycocotools.coco import COCO' not in content:
        content = content.replace(
            'from typing import Any, Optional',
            'from typing import Any, Optional\nfrom pycocotools.coco import COCO\nimport os'
        )
    
    # Add the patch before the class definition
    content = content.replace(
        '@DATASETS.register_module()\nclass YOLOv5CocoDataset(BatchShapePolicyDataset, CocoDataset):',
        f'@DATASETS.register_module()\nclass YOLOv5CocoDataset(BatchShapePolicyDataset, CocoDataset):{patch}'
    )
    
    # Write the patched file
    with open(dataset_file, 'w') as f:
        f.write(content)
    
    print("✅ Dataset fix applied successfully!")
    print("The YOLOv5CocoDataset class now properly loads COCO annotations.")
    print("Your training should now work with non-zero losses!")

if __name__ == '__main__':
    apply_dataset_fix()
