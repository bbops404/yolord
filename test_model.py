#!/usr/bin/env python3
"""
Standalone test script for YOLO-RD model components.
This script tests the model without requiring full MMYOLO installation.
"""

import sys
import os
import torch
import torch.nn as nn

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_som_backbone():
    """Test the SOM backbone component."""
    print("Testing SOM Backbone...")
    
    try:
        # Import the backbone
        import importlib.util
        spec = importlib.util.spec_from_file_location("som_backbone", "projects/yolo-rd/som_backbone.py")
        som_backbone_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(som_backbone_module)
        
        SOM_YOLOv8CSPDarknet = som_backbone_module.SOM_YOLOv8CSPDarknet
        
        # Create backbone
        backbone = SOM_YOLOv8CSPDarknet()
        
        # Test with dummy input
        x = torch.randn(1, 3, 640, 640)
        print(f"Input shape: {x.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = backbone(x)
        
        print(f"Backbone outputs: {[o.shape for o in outputs]}")
        print("✅ SOM Backbone test passed!")
        return True
        
    except Exception as e:
        print(f"❌ SOM Backbone test failed: {e}")
        return False

def test_maf_neck():
    """Test the MAF neck component."""
    print("\nTesting MAF Neck...")
    
    try:
        from projects.yolo_rd.mafpafpn import MAFPAFPN
        
        # Create neck
        neck = MAFPAFPN()
        
        # Test with dummy inputs (4 feature maps from backbone)
        inputs = [
            torch.randn(1, 64, 160, 160),   # P2
            torch.randn(1, 128, 80, 80),    # P3
            torch.randn(1, 256, 40, 40),    # P4
            torch.randn(1, 512, 20, 20)     # P5
        ]
        
        print(f"Neck inputs: {[x.shape for x in inputs]}")
        
        # Forward pass
        with torch.no_grad():
            outputs = neck(inputs)
        
        print(f"Neck outputs: {[o.shape for o in outputs]}")
        print("✅ MAF Neck test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MAF Neck test failed: {e}")
        return False

def test_wt_head():
    """Test the WT head component."""
    print("\nTesting WT Head...")
    
    try:
        from projects.yolo_rd.wt_head import WTCHead
        
        # Create head
        head = WTCHead(
            num_classes=4,
            in_channels=[128, 256, 512],
            featmap_strides=[8, 16, 32]
        )
        
        # Test with dummy inputs (3 feature maps from neck)
        inputs = [
            torch.randn(1, 128, 80, 80),    # P3
            torch.randn(1, 256, 40, 40),    # P4
            torch.randn(1, 512, 20, 20)     # P5
        ]
        
        print(f"Head inputs: {[x.shape for x in inputs]}")
        
        # Forward pass
        with torch.no_grad():
            outputs = head(inputs)
        
        print(f"Head outputs: {[o.shape for o in outputs]}")
        print("✅ WT Head test passed!")
        return True
        
    except Exception as e:
        print(f"❌ WT Head test failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline."""
    print("\nTesting Full Pipeline...")
    
    try:
        from projects.yolo_rd.som_backbone import SOM_YOLOv8CSPDarknet
        from projects.yolo_rd.mafpafpn import MAFPAFPN
        from projects.yolo_rd.wt_head import WTCHead
        
        # Create all components
        backbone = SOM_YOLOv8CSPDarknet()
        neck = MAFPAFPN()
        head = WTCHead(
            num_classes=4,
            in_channels=[128, 256, 512],
            featmap_strides=[8, 16, 32]
        )
        
        # Test with dummy input
        x = torch.randn(1, 3, 640, 640)
        print(f"Input: {x.shape}")
        
        # Forward pass through entire pipeline
        with torch.no_grad():
            # Backbone
            backbone_outputs = backbone(x)
            print(f"Backbone outputs: {[o.shape for o in backbone_outputs]}")
            
            # Neck
            neck_outputs = neck(backbone_outputs)
            print(f"Neck outputs: {[o.shape for o in neck_outputs]}")
            
            # Head
            head_outputs = head(neck_outputs)
            print(f"Head outputs: {[o.shape for o in head_outputs]}")
        
        print("✅ Full pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing YOLO-RD Model Components")
    print("=" * 50)
    
    # Test individual components
    backbone_ok = test_som_backbone()
    neck_ok = test_maf_neck()
    head_ok = test_wt_head()
    
    # Test full pipeline if all components work
    if backbone_ok and neck_ok and head_ok:
        pipeline_ok = test_full_pipeline()
    else:
        pipeline_ok = False
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Backbone: {'✅ PASS' if backbone_ok else '❌ FAIL'}")
    print(f"Neck:     {'✅ PASS' if neck_ok else '❌ FAIL'}")
    print(f"Head:     {'✅ PASS' if head_ok else '❌ FAIL'}")
    print(f"Pipeline: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")
    
    if pipeline_ok:
        print("\n🎉 All tests passed! Your YOLO-RD model is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
