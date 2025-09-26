import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8Head
from mmyolo.models.dense_heads.yolov8_head import YOLOv8HeadModule
from pytorch_wavelets import DWTForward, DWTInverse
from mmengine.model import BaseModule
from mmdet.models.utils import multi_apply
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


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
        
        # ULTRA AGGRESSIVE CLIPPING - Apply immediately after conv
        output = torch.clamp(output, min=-0.01, max=0.01)  # Extremely tight clipping
        
        # Add numerical stability by clipping extreme values
        output = torch.clamp(output, min=-0.01, max=0.01)  # Extremely tight clipping
        
        # Additional stability: check for NaN or Inf values
        if torch.isnan(output).any():
            print("WARNING: NaN detected in WaveletConvModule output, replacing with zeros")
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        if torch.isinf(output).any():
            print("WARNING: Inf detected in WaveletConvModule output, replacing with zeros")
            output = torch.where(torch.isinf(output), torch.zeros_like(output), output)
        
        # Check for extremely large values that might cause gradient explosion
        if output.abs().max() > 1.0:
            print(f"WARNING: Large values in WaveletConvModule output: max={output.abs().max():.2f}, clipping to [-1, 1]")
            output = torch.clamp(output, min=-1.0, max=1.0)

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
        
        # Debug: Check values before clipping
        if cls_pred.abs().max() > 0.01:
            print(f"🚨 BEFORE CLIPPING - cls_pred max: {cls_pred.abs().max():.4f}, min: {cls_pred.min():.4f}")
        
        # ULTRA AGGRESSIVE CLIPPING - Apply immediately after conv
        cls_pred = torch.clamp(cls_pred, min=-0.01, max=0.01)  # Extremely tight clipping
        
        # Upsample to match original spatial dimensions
        target_size = (x.shape[2], x.shape[3])  # Original H, W
        if cls_pred.shape[2:] != target_size:
            cls_pred = F.interpolate(cls_pred, size=target_size, mode='bilinear', align_corners=False)
        
        # Add numerical stability by clipping extreme values
        cls_pred = torch.clamp(cls_pred, min=-0.01, max=0.01)  # Extremely tight clipping
        
        # Additional stability: check for NaN or Inf values
        if torch.isnan(cls_pred).any():
            print("WARNING: NaN detected in cls_pred, replacing with zeros")
            cls_pred = torch.where(torch.isnan(cls_pred), torch.zeros_like(cls_pred), cls_pred)
        if torch.isinf(cls_pred).any():
            print("WARNING: Inf detected in cls_pred, replacing with zeros")
            cls_pred = torch.where(torch.isinf(cls_pred), torch.zeros_like(cls_pred), cls_pred)
        
        # Check for extremely large values that might cause gradient explosion
        if cls_pred.abs().max() > 1.0:
            print(f"WARNING: Large values in cls_pred: max={cls_pred.abs().max():.2f}, clipping to [-1, 1]")
            cls_pred = torch.clamp(cls_pred, min=-1.0, max=1.0)
        
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
        # Input channels = in_channels * 3 (3 HF bands for first level only)
        hf_channels = in_channels * 3
        # More aggressive memory reduction
        intermediate_channels = min(in_channels, 128)  # Further reduce to 128 channels
        self.edge_enhancement = nn.Sequential(
            ConvModule(hf_channels, intermediate_channels, 3, padding=1, 
                      norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')),
            ConvModule(intermediate_channels, in_channels, 1, 
                      norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'))
        )
        
        # Final regression layer
        self.reg_conv = ConvModule(in_channels, reg_out_channels, 1, act_cfg=None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Regression Head: Uses high-frequency features (X_LH, X_HL, X_HH) for precise 
        # delineation of crack boundaries and capture of intricate edge details
        
        # Ultra memory-efficient approach: Process only the first level to save memory
        # This trades some accuracy for memory efficiency
        current_x = x
        target_size = x.shape[-2:]
        
        # Process only the first level to save memory
        yl, yh = self.dwt(current_x)
        
        # Process high-frequency bands (LH, HL, HH) for the first level only
        lh_band, hl_band, hh_band = yh[0].unbind(dim=2)
        
        lh_proc = self.hf_convs[0]['lh'](lh_band)
        hl_proc = self.hf_convs[0]['hl'](hl_band)
        hh_proc = self.hf_convs[0]['hh'](hh_band)
        
        # Combine and resize to save memory
        level_hf = torch.cat([lh_proc, hl_proc, hh_proc], dim=1)
        
        # Resize to target spatial size
        if level_hf.shape[-2:] != target_size:
            level_hf = F.interpolate(level_hf, size=target_size, mode='bilinear', align_corners=False)
        
        # Edge enhancement for precise boundary delineation
        enhanced_features = self.edge_enhancement(level_hf)
        
        # Regression prediction for precise crack boundaries
        reg_pred = self.reg_conv(enhanced_features)
        
        # ULTRA AGGRESSIVE CLIPPING - Apply immediately after conv
        reg_pred = torch.clamp(reg_pred, min=-0.01, max=0.01)  # Extremely tight clipping
        
        # Add numerical stability by clipping extreme values
        reg_pred = torch.clamp(reg_pred, min=-0.01, max=0.01)  # Extremely tight clipping
        
        # Additional stability: check for NaN or Inf values
        if torch.isnan(reg_pred).any():
            print("WARNING: NaN detected in reg_pred, replacing with zeros")
            reg_pred = torch.where(torch.isnan(reg_pred), torch.zeros_like(reg_pred), reg_pred)
        if torch.isinf(reg_pred).any():
            print("WARNING: Inf detected in reg_pred, replacing with zeros")
            reg_pred = torch.where(torch.isinf(reg_pred), torch.zeros_like(reg_pred), reg_pred)
        
        # Check for extremely large values that might cause gradient explosion
        if reg_pred.abs().max() > 1.0:
            print(f"WARNING: Large values in reg_pred: max={reg_pred.abs().max():.2f}, clipping to [-1, 1]")
            reg_pred = torch.clamp(reg_pred, min=-1.0, max=1.0)
        
        return reg_pred


# Custom Head Module that integrates with YOLOv8Head
@MODELS.register_module()
class WTCHeadModule(BaseModule):
    """
    Custom head module that uses frequency-based heads and integrates properly with YOLOv8Head.
    This replaces the standard YOLOv8HeadModule with our wavelet-based implementation.
    """
    
    def __init__(self,
                 num_classes: int,
                 in_channels: list,
                 widen_factor: float = 1.0,
                 reg_max: int = 16,
                 norm_cfg: dict = None,
                 act_cfg: dict = None,
                 featmap_strides: list = [8, 16, 32],
                 levels: int = 2,
                 init_cfg: dict = None):
        super().__init__(init_cfg=init_cfg)
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.widen_factor = widen_factor
        self.reg_max = reg_max
        self.norm_cfg = norm_cfg or dict(type='BN', momentum=0.03, eps=0.001)
        self.act_cfg = act_cfg or dict(type='SiLU', inplace=True)
        self.featmap_strides = featmap_strides
        self.levels = levels
        self.num_levels = len(in_channels)
        
        # print(f"DEBUG WTCHeadModule: Initializing with num_classes={num_classes}, in_channels={in_channels}")
        # print(f"DEBUG WTCHeadModule: reg_max={reg_max}, levels={levels}")
        
        self._init_layers()
    
    def _init_layers(self):
        """Initialize our custom frequency-based heads."""
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        # In YOLOv8, reg_out_channels = 4 (for bbox) * self.reg_max (for DFL)
        reg_out_channels = 4 * self.reg_max
        
        # Initialize the proj buffer for DFL (Distribution Focal Loss)
        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)
        
        # Build specialized frequency-based branches for each feature map level
        for in_channel in self.in_channels:
            # Classification Branch - Specialized for Low-Frequency Features (X_LL)
            self.cls_preds.append(
                LowFrequencyHead(in_channel, self.num_classes, levels=self.levels)
            )
            
            # Regression Branch - Specialized for High-Frequency Features (X_LH, X_HL, X_HH)
            self.reg_preds.append(
                HighFrequencyHead(in_channel, reg_out_channels, levels=self.levels)
            )
    
    def forward(self, x):
        """Forward features from the upstream network."""
        assert len(x) == self.num_levels
        result = multi_apply(self.forward_single, x, self.cls_preds, self.reg_preds)
        return result
    
    def forward_single(self, x, cls_pred, reg_pred):
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        
        # Use our custom frequency-based heads
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)  # Raw distribution predictions
        
        # ULTRA AGGRESSIVE CLIPPING - Apply before any further processing
        cls_logit = torch.clamp(cls_logit, min=-0.01, max=0.01)  # Extremely tight clipping
        bbox_dist_preds = torch.clamp(bbox_dist_preds, min=-0.01, max=0.01)  # Extremely tight clipping
        
        # Check for extreme values and warn
        if cls_logit.abs().max() > 0.01:
            print(f"🚨 EXTREME CLS VALUES: max={cls_logit.abs().max():.4f}, clipping to [-0.01, 0.01]")
            cls_logit = torch.clamp(cls_logit, min=-0.01, max=0.01)
        if bbox_dist_preds.abs().max() > 0.01:
            print(f"🚨 EXTREME REG VALUES: max={bbox_dist_preds.abs().max():.4f}, clipping to [-0.01, 0.01]")
            bbox_dist_preds = torch.clamp(bbox_dist_preds, min=-0.01, max=0.01)
        
        # Process bbox predictions like YOLOv8HeadModule does
        if self.reg_max > 1:
            # Reshape from (B, 4*reg_max, H, W) to (B, H*W, 4, reg_max)
            bbox_dist_preds_reshaped = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
            
            # Apply softmax and matrix multiplication to get final bbox predictions
            bbox_preds = bbox_dist_preds_reshaped.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        
        # Final clipping of all outputs
        cls_logit = torch.clamp(cls_logit, min=-0.01, max=0.01)
        bbox_preds = torch.clamp(bbox_preds, min=-0.01, max=0.01)
        bbox_dist_preds = torch.clamp(bbox_dist_preds, min=-0.01, max=0.01)
        
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


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
    
    def __init__(self, 
                 head_module: dict,
                 prior_generator: dict = None,
                 bbox_coder: dict = None,
                 loss_cls: dict = None,
                 loss_bbox: dict = None,
                 loss_dfl: dict = None,
                 train_cfg: dict = None,
                 test_cfg: dict = None,
                 init_cfg: dict = None,
                 levels: int = 2,
                 **kwargs):
        # Debug: Check input parameters
        # print(f"DEBUG: train_cfg parameter: {train_cfg}")
        # print(f"DEBUG: test_cfg parameter: {test_cfg}")
        
        # Store levels parameter first
        self.levels = levels
        
        # Extract parameters from head_module for our custom implementation
        self.num_classes = head_module.get('num_classes', 80)
        self.in_channels = head_module.get('in_channels', [256, 512, 1024])
        self.widen_factor = head_module.get('widen_factor', 1.0)
        self.reg_max = head_module.get('reg_max', 16)
        self.norm_cfg = head_module.get('norm_cfg', dict(type='BN', momentum=0.03, eps=0.001))
        self.act_cfg = head_module.get('act_cfg', dict(type='SiLU', inplace=True))
        self.featmap_strides = head_module.get('featmap_strides', [8, 16, 32])
        
        # Create our custom head_module for YOLOv8Head
        compatible_head_module = dict(
            type='WTCHeadModule',
            num_classes=self.num_classes,
            in_channels=self.in_channels,
            widen_factor=self.widen_factor,
            reg_max=self.reg_max,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            featmap_strides=self.featmap_strides,
            levels=self.levels
        )
        
        # Store original parameters for later use
        self.original_head_module = head_module
        
        # Call parent class initialization
        super().__init__(
            head_module=compatible_head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)
        
        # Add missing attributes for YOLOv8HeadModule compatibility
        self.featmap_sizes_train = None
        
        # Debug: Check if assigner exists
        # print(f"DEBUG: hasattr(self, 'assigner'): {hasattr(self, 'assigner')}")
        # if hasattr(self, 'assigner'):
        #     print(f"DEBUG: self.assigner: {self.assigner}")
        # print(f"DEBUG: self.train_cfg: {self.train_cfg}")
        
        # Ensure assigner is properly initialized
        if not hasattr(self, 'assigner') or self.assigner is None:
            if self.train_cfg and 'assigner' in self.train_cfg:
                from mmdet.registry import TASK_UTILS
                self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
                # print(f"DEBUG: Created assigner: {self.assigner}")
            else:
                # print("DEBUG: No train_cfg or assigner in train_cfg")
                # Create a default assigner
                from mmdet.registry import TASK_UTILS
                default_assigner_cfg = dict(
                    type='BatchTaskAlignedAssigner',
                    num_classes=self.num_classes,
                    topk=26,  # More permissive
                    alpha=0.5,  # Less strict
                    beta=3,  # Less strict
                    eps=1e-9
                )
                self.assigner = TASK_UTILS.build(default_assigner_cfg)
                # print(f"DEBUG: Created default assigner: {self.assigner}")
        
        # Override assigner forward to add debug information
        # original_forward = self.assigner.forward
        # def debug_forward(pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes, pad_bbox_flag):
        #     print(f"DEBUG Assigner: pred_bboxes shape: {pred_bboxes.shape}")
        #     print(f"DEBUG Assigner: pred_scores shape: {pred_scores.shape}")
        #     print(f"DEBUG Assigner: priors shape: {priors.shape}")
        #     print(f"DEBUG Assigner: gt_labels shape: {gt_labels.shape}")
        #     print(f"DEBUG Assigner: gt_bboxes shape: {gt_bboxes.shape}")
        #     print(f"DEBUG Assigner: pad_bbox_flag shape: {pad_bbox_flag.shape}")
        #     print(f"DEBUG Assigner: pad_bbox_flag sum: {pad_bbox_flag.sum()}")
        #     
        #     result = original_forward(pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes, pad_bbox_flag)
        #     
        #     print(f"DEBUG Assigner: fg_mask_pre_prior sum: {result['fg_mask_pre_prior'].sum()}")
        #     print(f"DEBUG Assigner: assigned_scores sum: {result['assigned_scores'].sum()}")
        #     
        #     return result
        # 
        # self.assigner.forward = debug_forward
        


    def forward(self, x):
        """Forward pass using the head_module (which we'll replace with our custom implementation)."""
        # Use the parent class's forward method, which will call self.head_module(x)
        # We'll replace head_module with our custom implementation
        result = self.head_module(x)
        return result
    
    def loss_by_feat(self, cls_scores, bbox_preds, bbox_dist_preds, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore=None):
        """Override loss computation to add comprehensive debug information."""
        
        # Call parent's loss computation
        result = super().loss_by_feat(cls_scores, bbox_preds, bbox_dist_preds, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        
        # Debug zero losses
        if result['loss_bbox'] == 0 or result['loss_dfl'] == 0:
            print(f"⚠️  ZERO LOSS DEBUG: loss_bbox={result['loss_bbox']:.4f}, loss_dfl={result['loss_dfl']:.4f}")
            print(f"⚠️  This means no positive assignments found by assigner")
            print(f"⚠️  GT instances type: {type(batch_gt_instances)}")
            if hasattr(batch_gt_instances, 'shape'):
                print(f"⚠️  GT instances shape: {batch_gt_instances.shape}")
                print(f"⚠️  GT sample values: {batch_gt_instances[:3] if len(batch_gt_instances) > 0 else 'Empty'}")
                print(f"⚠️  GT value ranges: min={batch_gt_instances.min():.4f}, max={batch_gt_instances.max():.4f}")
        
        # Debug extremely high losses
        if result['loss_cls'] > 1000:
            print(f"🚨 HIGH LOSS DEBUG: loss_cls={result['loss_cls']:.2f} (extremely high!)")
            print(f"🚨 This indicates numerical instability in the model")
        
        return result

