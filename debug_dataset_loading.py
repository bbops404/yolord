#!/usr/bin/env python3
"""
Simple debug script to test if the training works with the simplified pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """Test if the config loads correctly."""
    
    from mmengine import Config
    
    # Load config
    config_path = 'projects/yolo-rd/yolord_s_coco_config.py'
    cfg = Config.fromfile(config_path)
    
    print("=== Dataset Configuration ===")
    print(f"Data root: {cfg.data_root}")
    print(f"Ann file: {cfg.train_dataloader.dataset.ann_file}")
    print(f"Data prefix: {cfg.train_dataloader.dataset.data_prefix}")
    print(f"Filter config: {cfg.train_dataloader.dataset.filter_cfg}")
    
    print("\n=== Training Pipeline ===")
    pipeline = cfg.train_dataloader.dataset.pipeline
    for i, step in enumerate(pipeline):
        print(f"Step {i}: {step['type']}")
    
    print("\n=== Config looks good! ===")
    print("The simplified pipeline should work. Try running training now:")
    print("python tools/train.py projects/yolo-rd/yolord_s_coco_config.py --work-dir work_dirs/yolord_s_coco --resume")

if __name__ == '__main__':
    test_config_loading()
