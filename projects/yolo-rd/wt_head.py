import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8Head
from pytorch_wavelets import DWTForward, DWTInverse


# Helper Module: The core Wavelet Convolution block
class WaveletConvModule(nn.Module):
    """
    Implements the Wavelet Transform Convolution (WTC) block from the YOLO-RD paper.
    
    This module decomposes the input into frequency bands, applies convolutions
    on each band, reconstructs the feature map, and adds a residual connection.
    This is a reusable building block for the WTC Head.
    
    Reference: Figure 5 in the YOLO-RD paper.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 wavelet: str = 'db1', # db1 is the Haar wavelet, as used in the paper
                 levels: int = 2):  # Changed from 1 to 2 for multi-level decomposition
        super().__init__()
        self.levels = levels
        
        # Wavelet decomposition and reconstruction layers
        self.dwt = DWTForward(J=levels, wave=wavelet, mode='symmetric')
        self.iwt = DWTInverse(wave=wavelet, mode='symmetric')

        # Convolutions for each frequency band at each level
        # For multi-level decomposition, we need convolutions for each level
        self.conv_modules = nn.ModuleList()
        
        for level in range(levels):
            level_convs = nn.ModuleDict({
                'll': ConvModule(in_channels, in_channels, 3, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')),
                'lh': ConvModule(in_channels, in_channels, 3, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')),
                'hl': ConvModule(in_channels, in_channels, 3, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')),
                'hh': ConvModule(in_channels, in_channels, 3, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
            })
            self.conv_modules.append(level_convs)

        # Residual connection path (Conv(X) in Figure 5)
        self.residual_conv = ConvModule(in_channels, out_channels, 1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        # Final convolution to match output channels after adding residual
        self.out_conv = ConvModule(out_channels, out_channels, 1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Residual path (Conv(X) in the diagram)
        residual = self.residual_conv(x)

        # 2. Recursive Wavelet Decomposition and Reconstruction
        # This implements the true recursive decomposition as shown in the diagram:
        # X → X^(1) → X^(2) → ... → X^(levels) then Z^(levels) → Z^(levels-1) → ... → Z^(1) → Z^(0)
        
        # Start with the input X
        current_x = x
        processed_bands = []  # Store processed bands for each level
        
        # Recursive decomposition: X → X^(1) → X^(2) → ... → X^(levels)
        for level in range(self.levels):
            # Decompose current X into frequency bands
            yl, yh = self.dwt(current_x)
            
            # Process high-frequency bands (LH, HL, HH) for this level
            lh_band, hl_band, hh_band = yh[0].unbind(dim=2)  # yh[0] contains the bands for this level
            
            lh_proc = self.conv_modules[level]['lh'](lh_band)
            hl_proc = self.conv_modules[level]['hl'](hl_band)
            hh_proc = self.conv_modules[level]['hh'](hh_band)
            
            # Store processed high-frequency bands for this level
            processed_bands.append(torch.stack([lh_proc, hl_proc, hh_proc], dim=2))
            
            # Process low-frequency band (LL) for this level
            ll_proc = self.conv_modules[level]['ll'](yl)
            
            # For the next level, use the processed low-frequency component
            current_x = ll_proc
        
        # 3. Bottom-up Reconstruction: Z^(levels) → Z^(levels-1) → ... → Z^(1) → Z^(0)
        # Start reconstruction from the deepest level
        current_z = current_x  # This is the final processed X_LL^(levels)
        
        # Reconstruct level by level from deepest to shallowest
        for level in range(self.levels - 1, -1, -1):
            # Get the processed high-frequency bands for this level
            processed_yh = [processed_bands[level]]
            
            # Reconstruct this level using IWT
            current_z = self.iwt((current_z, processed_yh))
        
        # The final reconstructed feature map
        reconstructed = current_z
        
        # 4. Add residual to the reconstructed feature map
        # Ensure the size matches if padding/striding caused a mismatch
        if reconstructed.shape != residual.shape:
             reconstructed = F.interpolate(reconstructed, size=residual.shape[2:], mode='bilinear', align_corners=False)
        
        output_fused = reconstructed + residual
        
        # Apply final conv to integrate features
        output = self.out_conv(output_fused)

        return output


# Specialized Head for Low-Frequency Features (Classification)
class LowFrequencyHead(nn.Module):
    """
    Specialized head that capitalizes on low-frequency features (X_LL) to effectively 
    discern regions containing cracks by leveraging the global structural context 
    encoded within these features.
    """
    def __init__(self, in_channels: int, num_classes: int, levels: int = 2):
        super().__init__()
        self.levels = levels
        
        # Wavelet decomposition for extracting low-frequency components
        self.dwt = DWTForward(J=levels, wave='db1', mode='symmetric')
        
        # Convolutions specifically for low-frequency components at each level
        self.ll_convs = nn.ModuleList()
        for level in range(levels):
            self.ll_convs.append(
                ConvModule(in_channels, in_channels, 3, padding=1, 
                         norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
            )
        
        # Global context processing for crack region identification
        self.global_context = nn.Sequential(
            ConvModule(in_channels, in_channels, 3, padding=1, 
                      norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')),
            ConvModule(in_channels, in_channels, 1, 
                      norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        )
        
        # Final classification layer
        self.cls_conv = ConvModule(in_channels, num_classes, 1, act_cfg=None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classification Head: Uses low-frequency features (X_LL) for global structural context
        # to discern regions containing cracks
        
        current_x = x
        ll_features = []
        
        # Recursive decomposition to extract low-frequency components only
        for level in range(self.levels):
            yl, _ = self.dwt(current_x)  # Only extract low-frequency (yl)
            ll_processed = self.ll_convs[level](yl)
            ll_features.append(ll_processed)
            current_x = ll_processed  # Only X_LL goes to next level
        
        # Use the deepest low-frequency component for global structural context
        # This captures the overall crack region information
        global_features = self.global_context(ll_features[-1])
        
        # Classification prediction for crack regions
        cls_pred = self.cls_conv(global_features)
        
        return cls_pred


# Specialized Head for High-Frequency Features (Regression)
class HighFrequencyHead(nn.Module):
    """
    Specialized head that harnesses high-frequency components (X_LH, X_HL, X_HH) 
    to achieve precise delineation of crack boundaries, ensuring the accurate 
    capture of intricate edge details and local textures.
    """
    def __init__(self, in_channels: int, reg_out_channels: int, levels: int = 2):
        super().__init__()
        self.levels = levels
        
        # Wavelet decomposition for extracting high-frequency components
        self.dwt = DWTForward(J=levels, wave='db1', mode='symmetric')
        self.iwt = DWTInverse(wave='db1', mode='symmetric')
        
        # Convolutions for all frequency components at each level
        self.hf_convs = nn.ModuleList()
        for level in range(levels):
            level_convs = nn.ModuleDict({
                'll': ConvModule(in_channels, in_channels, 3, padding=1, 
                               norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')),
                'lh': ConvModule(in_channels, in_channels, 3, padding=1, 
                               norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')),
                'hl': ConvModule(in_channels, in_channels, 3, padding=1, 
                               norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')),
                'hh': ConvModule(in_channels, in_channels, 3, padding=1, 
                               norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
            })
            self.hf_convs.append(level_convs)
        
        # Edge enhancement for precise boundary delineation
        # Input channels = in_channels * 3 * levels (3 HF bands × levels)
        hf_channels = in_channels * 3 * levels
        self.edge_enhancement = nn.Sequential(
            ConvModule(hf_channels, in_channels, 3, padding=1, 
                      norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')),
            ConvModule(in_channels, in_channels, 1, 
                      norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        )
        
        # Final regression layer
        self.reg_conv = ConvModule(in_channels, reg_out_channels, 1, act_cfg=None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regression Head: Uses high-frequency features (X_LH, X_HL, X_HH) for precise 
        # delineation of crack boundaries and capture of intricate edge details
        
        # Step 1: Extract high-frequency components from all decomposition levels
        current_x = x
        hf_features = []  # Store high-frequency features from all levels
        
        # Recursive decomposition: Only X_LL is decomposed, but we collect HF at each level
        for level in range(self.levels):
            yl, yh = self.dwt(current_x)
            
            # Process high-frequency bands (LH, HL, HH) for this level
            lh_band, hl_band, hh_band = yh[0].unbind(dim=2)
            
            lh_proc = self.hf_convs[level]['lh'](lh_band)
            hl_proc = self.hf_convs[level]['hl'](hl_band)
            hh_proc = self.hf_convs[level]['hh'](hh_band)
            
            # Store high-frequency features from this level
            hf_features.append({
                'lh': lh_proc,  # Horizontal edges
                'hl': hl_proc,  # Vertical edges  
                'hh': hh_proc   # Diagonal edges/corners
            })
            
            # Only X_LL goes to next level decomposition
            current_x = yl
        
        # Step 2: Combine high-frequency features from all levels
        # This captures edge details at multiple scales for precise boundary delineation
        combined_hf = []
        for level in range(self.levels):
            # Combine all high-frequency components from this level
            level_hf = torch.cat([
                hf_features[level]['lh'],
                hf_features[level]['hl'], 
                hf_features[level]['hh']
            ], dim=1)  # Concatenate along channel dimension
            combined_hf.append(level_hf)
        
        # Fuse high-frequency features from all levels
        # This provides multi-scale edge information for precise boundaries
        fused_hf = torch.cat(combined_hf, dim=1)
        
        # Edge enhancement for precise boundary delineation
        enhanced_features = self.edge_enhancement(fused_hf)
        
        # Regression prediction for precise crack boundaries
        reg_pred = self.reg_conv(enhanced_features)
        
        return reg_pred


# Main Module: The complete Detection Head
@MODELS.register_module()
class WTCHead(YOLOv8Head):
    """
    YOLO-RD Wavelet Transform Convolution (WTC) Head.
    
    This head replaces the standard convolution layers in the YOLOv8 head's
    prediction branches with specialized frequency-based heads:
    - Classification Head: Uses low-frequency features (X_LL) for global structural context
    - Regression Head: Uses high-frequency features (X_LH, X_HL, X_HH) for precise boundaries
    """
    
    def __init__(self, *args, levels: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = levels

    def _init_layers(self):
        """Override the parent class's layer initialization to use specialized frequency-based WTC."""
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        # In YOLOv8, reg_out_channels = 4 (for bbox) * self.reg_max (for DFL)
        reg_out_channels = 4 * self.reg_max
        
        # Build specialized frequency-based branches for each feature map level from the neck
        for in_channel in self.in_channels:
            # Classification Branch - Specialized for Low-Frequency Features (X_LL)
            # Uses low-frequency components for global structural context to identify crack regions
            self.cls_preds.append(
                LowFrequencyHead(in_channel, self.num_classes, levels=self.levels)
            )
            
            # Regression Branch - Specialized for High-Frequency Features (X_LH, X_HL, X_HH)
            # Uses high-frequency components for precise boundary delineation and edge details
            self.reg_preds.append(
                HighFrequencyHead(in_channel, reg_out_channels, levels=self.levels)
            )

        # The DFL (Distribution Focal Loss) layers and stride tensor from the
        # parent YOLOv8Head are initialized automatically and do not need to be changed.