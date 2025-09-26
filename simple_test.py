#!/usr/bin/env python3
"""
Simple test for YOLO-RD model components.
"""

import sys
import os
import torch

# Add the projects directory to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'projects'))

def test_imports():
    """Test if we can import the modules."""
    print("Testing imports...")
    
    try:
        # Test individual imports
        from yolo_rd.som_backbone import SOM_YOLOv8CSPDarknet
        print("✅ SOM Backbone imported")
        
        from yolo_rd.mafpafpn import MAFPAFPN
        print("✅ MAF-PAFPN imported")
        
        from yolo_rd.wt_head import WTCHead
        print("✅ WT Head imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_backbone():
    """Test the backbone."""
    print("\nTesting SOM Backbone...")
    
    try:
        from yolo_rd.som_backbone import SOM_YOLOv8CSPDarknet
        
        # Create backbone
        backbone = SOM_YOLOv8CSPDarknet()
        
        # Test input
        x = torch.randn(1, 3, 640, 640)
        print(f"Input shape: {x.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = backbone(x)
        
        print(f"Output shapes: {[o.shape for o in outputs]}")
        print("✅ Backbone test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Backbone test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Simple YOLO-RD Test")
    print("=" * 30)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test backbone
        backbone_ok = test_backbone()
        
        if backbone_ok:
            print("\n🎉 Basic test passed! Your model components are working!")
        else:
            print("\n⚠️  Backbone test failed.")
    else:
        print("\n❌ Import test failed. Check dependencies.")
