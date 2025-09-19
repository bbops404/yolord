import torch
import torch.nn as nn
from mmcv.ops import DeformConv2dPack

class AttentionRefinementModule(nn.Module):
    """
    Attention Refinement Module (ARM) to enhance focus on important regions (e.g., cracks).
    This module refines the feature maps from different stages by applying spatial adjustments
    to focus on critical regions such as small cracks, and suppress irrelevant background features.
    """
    def __init__(self, in_channels):
        """
        Initialize the ARM module.
        
        Args:
        - in_channels (int): Number of input channels for the feature map.
        """
        super().__init__()
        # sa loob nito meron na siyang offsetconv + deform_conv 
        self.deform_conv = DeformConv2dPack(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )
        # Sigmoid activation to normalize the feature map after refinement
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fstagei):
        """
        Forward pass to apply spatial refinement and attention to the feature map.
        
        Args:
        - Fstagei (Tensor): Feature map from a specific stage (e.g., Stage 2, Stage 3, Stage 4).
        
        Returns:
        - FARM (Tensor): The refined feature map after applying deformable convolution and attention.
        """
        # Apply deformable convolution which internally estimates offsets
        Frefined = self.deform_conv(Fstagei)
        # Apply Sigmoid activation to normalize the output feature map, focusing on important regions
        FARM = self.sigmoid(Frefined)  # FARM = sigmoid(deform_conv(Fstagei + Δpk))
        return FARM
