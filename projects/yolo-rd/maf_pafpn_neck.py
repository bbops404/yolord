import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from mmdet.models.utils import multi_apply
from .arm import AttentionRefinementModule  # Importing ARM Module

@MODELS.register_module()
class MultiDimensionalAuxiliaryFusion(nn.Module):
    """
    Multi-dimensional Auxiliary Fusion (MAF) module for fusing multi-scale features.
    This module performs feature fusion by combining low-level and high-level features
    and applies the Attention Refinement Module (ARM) to refine the features for small crack detection.
    """
    def __init__(self, in_channels, out_channels, num_levels=4):
        """
        Initialize the MAF module.
        
        Args:
        - in_channels (list): List of input channels for each level.
        - out_channels (list): List of output channels for each level.
        - num_levels (int): Number of levels to process (default is 4).
        """
        super().__init__()
        self.num_levels = num_levels  # Number of feature levels to process (4 levels for Stage 1, Stage 2, Stage 3, Stage 4)
        self.in_channels = in_channels  # Input channels for each stage
        self.out_channels = out_channels  # Output channels for each stage

        # Initialize MAF layers to fuse multi-scale features from different levels
        self.maf_layers = nn.ModuleList()
        for i in range(num_levels):
            # Apply convolution with kernel size 1 to reduce dimensionality, followed by softmax attention
            maf_layer = nn.Sequential(
                ConvModule(in_channels[i], out_channels[i], kernel_size=1, stride=1, padding=0),  # Convolution to map input to output channels
                nn.Softmax(dim=1)  # Softmax across the channel dimension for attention-based feature refinement
            )
            self.maf_layers.append(maf_layer)

        print("--- INFO: MultiDimensionalAuxiliaryFusion (MAF) - Feature Fusion Module ---")

        # Initialize the ARM (Attention Refinement Module) for spatial feature refinement
        # ARM operates on the first stage output after feature fusion
        self.arm = AttentionRefinementModule(in_channels=out_channels[0])  # ARM operates on the output of Stage 1

    def forward(self, x):
        """
        Forward pass for the MultiDimensionalAuxiliaryFusion module.
        
        Args:
        - x (list): A list of feature maps from different stages (Stage 1, Stage 2, etc.)
        
        Returns:
        - Processed feature map after fusion and attention refinement by ARM
        """
        assert len(x) == self.num_levels  # Ensure the number of input levels matches the expected number
        features = multi_apply(self.forward_single, x)  # Apply fusion to each feature map at each level
        return self.arm(features[0])  # Apply ARM refinement to the first stage feature map

    def forward_single(self, x):
        """
        Forward pass for a single scale level. It applies the MAF fusion for the current level.
        
        Args:
        - x (Tensor): The input feature map for a single stage (Level 1, Level 2, etc.)
        
        Returns:
        - out (Tensor): The refined feature map after applying the fusion and attention refinement
        """
        out = self.maf_layers[0](x)  # Apply the fusion layer to the input feature map
        return out

# Example of how to instantiate this MAF module with 4 stages
num_levels = 4  # 4 stages (Stage 1, Stage 2, Stage 3, Stage 4)
in_channels = [64, 128, 256, 512]  # Input channels for each stage (low-level to high-level features)
out_channels = [128, 256, 512, 1024]  # Output channels after fusion (increasing as the model progresses)

maf_module = MultiDimensionalAuxiliaryFusion(in_channels, out_channels, num_levels)
