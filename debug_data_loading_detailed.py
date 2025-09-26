import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_data_loading():
    """Test data loading with minimal pipeline."""
    
    from mmengine import Config
    from mmyolo.registry import DATASETS
    import json
    
    # Load the minimal config
    config_path = 'test_minimal_config.py'
    cfg = Config.fromfile(config_path)
    
    print("=== Testing with Minimal Config ===")
    print(f"Data root: {cfg.data_root}")
    print(f"Ann file: {cfg.train_dataloader.dataset.ann_file}")
    print(f"Pipeline: {[step['type'] for step in cfg.train_dataloader.dataset.pipeline]}")
    
    try:
        # Build dataset
        dataset_cfg = cfg.train_dataloader.dataset.copy()
        dataset = DATASETS.build(dataset_cfg)
        print(f"✓ Dataset built successfully with {len(dataset)} samples")
        
        # Test loading first few samples
        print("\n=== Testing Sample Loading ===")
        for i in range(min(3, len(dataset))):
            print(f"\n--- Sample {i} ---")
            try:
                sample = dataset[i]
                print(f"Sample keys: {list(sample.keys())}")
                
                # Check for ground truth data in data_samples
                if 'data_samples' in sample:
                    data_sample = sample['data_samples']
                    print(f"Data sample type: {type(data_sample)}")
                    
                    if hasattr(data_sample, 'gt_instances'):
                        gt_instances = data_sample.gt_instances
                        print(f"GT instances type: {type(gt_instances)}")
                        
                        if hasattr(gt_instances, 'bboxes'):
                            bboxes = gt_instances.bboxes
                            print(f"✓ GT bboxes: {bboxes.shape}")
                            if len(bboxes) > 0:
                                print(f"  - First bbox: {bboxes[0]}")
                            else:
                                print(f"  - NO BBOXES!")
                        else:
                            print(f"✗ No bboxes attribute")
                            
                        if hasattr(gt_instances, 'labels'):
                            labels = gt_instances.labels
                            print(f"✓ GT labels: {labels.shape}")
                            if len(labels) > 0:
                                print(f"  - First label: {labels[0]}")
                            else:
                                print(f"  - NO LABELS!")
                        else:
                            print(f"✗ No labels attribute")
                    else:
                        print(f"✗ No gt_instances in data_sample")
                else:
                    print(f"✗ No data_samples key")
                    
            except Exception as e:
                print(f"✗ Error loading sample {i}: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"✗ Error building dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_simple_data_loading()
