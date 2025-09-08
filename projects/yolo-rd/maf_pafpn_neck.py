import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from mmdet.models.utils import multi_apply
import math

@MODELS.register_module()
class MAFPAFPN(nn.Module):
    """
    Multi-dimensional Auxiliary Fusion (MAF) and Path Aggregation Feature Pyramid Network (PAFPN)
    Neck module for YOLO-RD, adapted for MMYOLO.
    """
    def __init__(self, in_channels, out_channels, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.maf_layers = nn.ModuleList()
        for i in range(num_levels):
            # Define layers for multi-scale feature fusion (MAF)
            maf_layer = nn.Sequential(
                ConvModule(in_channels[i], out_channels[i], kernel_size=1, stride=1, padding=0),
                nn.Softmax(dim=1)  # Adding attention through Softmax (for feature refinement)
            )
            self.maf_layers.append(maf_layer)

        print("--- INFO: MAFPAFPN - Multi-dimensional Auxiliary Fusion with Path Aggregation ---")

    def forward(self, x):
        """
        Forward pass for the MAF-PAFPN neck.
        x: List of feature maps from different levels (e.g., backbone output).
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x)

    def forward_single(self, x):
        """
        Forward pass for a single scale level.
        """
        # Apply the MAF fusion for this level
        out = self.maf_layers[0](x)  # Fusion and attention refinement
        return out


class AttentionRefinementModule(nn.Module):
    """
    Attention Refinement Module (ARM) for better focusing on important features.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = ConvModule(in_channels, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass to apply attention.
        """
        attention_map = self.conv(x)
        return x * self.sigmoid(attention_map)  # Apply attention to enhance key areas


class MAFModule(nn.Module):
    """
    MAF Module (Multi-dimensional Auxiliary Fusion) for combining low-level and high-level features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.low_conv = ConvModule(in_channels[0], out_channels, kernel_size=3, padding=1)
        self.high_conv = ConvModule(in_channels[1], out_channels, kernel_size=3, padding=1)

    def forward(self, low_features, high_features):
        """
        Forward pass to fuse low-level and high-level features.
        """
        low_features = self.low_conv(low_features)
        high_features = self.high_conv(high_features)
        return torch.cat([low_features, high_features], dim=1)  # Combine features from both levels
