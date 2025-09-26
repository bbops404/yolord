# projects/yolo-rd/som_backbone.py

import torch.nn as nn
from mmcv.cnn import ConvModule

from mmyolo.registry import MODELS
from mmyolo.models.backbones.csp_darknet import YOLOv8CSPDarknet
from mmdet.models.backbones.csp_darknet import CSPLayer as YOLOv8CSPLayer
from mmyolo.models.layers.yolo_bricks import SPPFBottleneck
from mmyolo.models.utils import make_divisible
from .star_operation_module import StarOperationModule


@MODELS.register_module()
class SOM_YOLOv8CSPDarknet(YOLOv8CSPDarknet):
    """
    YOLO-RD Backbone with StarOperationModule (SOM).

    This class meticulously reconstructs the backbone from the YOLO-RD paper
    (Figure 2), ensuring the feature map dimensions at each stage are accurate.
    SOM is integrated into the third and fourth stages to enhance feature
    extraction for small objects, as described in the paper.
    """

    def __init__(self,
                 arch='P5',
                 last_stage_out_channels=1024,
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 input_channels=3,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU', inplace=True),
                 use_depthwise=False,
                 **kwargs):
        
        # We call the parent's init to properly initialize the base backbone
        super().__init__(
            arch=arch,
            last_stage_out_channels=last_stage_out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        # Store architecture scaling factors
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.input_channels = input_channels

        # Define the channel progression for each stage output
        # Based on a medium-sized model like YOLOv8-s/m
        channels_settings = {
            'P5': [64, 128, 256, 512, 512]
        }
        self.channels = [
            int(make_divisible(c, widen_factor)) for c in channels_settings[arch]
        ]
        
        # Define block counts for CSPLayers
        num_blocks_settings = {
            'P5': [3, 6, 6, 3]
        }
        self.num_blocks = [
            round(n * deepen_factor) for n in num_blocks_settings[arch]
        ]

        # --- Build Network Stage-by-Stage to Match Paper's Figure 2 ---

        # 1. Stem Layer (initial convolution)
        # Input: 640x640x3
        self.stem = ConvModule(
            input_channels,
            self.channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # Output: 320x320x64

        # 2. Stage 1 (Standard YOLOv8 Block)
        self.stage1 = nn.Sequential(
            ConvModule(
                self.channels[0], self.channels[0], 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            YOLOv8CSPLayer(
                self.channels[0], self.channels[0], num_blocks=self.num_blocks[0], add_identity=True, use_depthwise=use_depthwise, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        # Output: 160x160x64
        
        # 3. Stage 2 (Standard YOLOv8 Block)
        self.stage2 = nn.Sequential(
            ConvModule(
                self.channels[0], self.channels[1], 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            YOLOv8CSPLayer(
                self.channels[1], self.channels[1], num_blocks=self.num_blocks[1], add_identity=True, use_depthwise=use_depthwise, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        # Output: 80x80x128 (This is the first feature map sent to the neck)

        # 4. Stage 3 (CUSTOM STAGE WITH SOM)
        # According to Figure 2, this stage contains 8 SOM modules and a CSPLayer
        self.stage3 = nn.Sequential(
            # First, a downsampling convolution
            ConvModule(
                self.channels[1], self.channels[2], 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            # Then, 8 Star Operation Modules (×8)
            *[StarOperationModule(
                in_channels=self.channels[2],
                out_channels=self.channels[2],
            ) for _ in range(8)],
            # Followed by a standard CSPLayer
            YOLOv8CSPLayer(
                self.channels[2], self.channels[2], num_blocks=self.num_blocks[2], add_identity=True, use_depthwise=use_depthwise, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )
        # Output: 40x40x256 (This is the second feature map sent to the neck)

        # 5. Stage 4 (CUSTOM STAGE WITH SOM)
        # This stage contains 5 SOM modules, a CSPLayer, and SPPF
        self.stage4 = nn.Sequential(
             # First, a downsampling convolution
            ConvModule(
                self.channels[2], self.channels[3], 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            # Then, 5 Star Operation Modules (×5)
            *[StarOperationModule(
                in_channels=self.channels[3],
                out_channels=self.channels[3],
            ) for _ in range(5)],
            # Followed by a standard CSPLayer
            YOLOv8CSPLayer(
                self.channels[3], self.channels[3], num_blocks=self.num_blocks[3], add_identity=True, use_depthwise=use_depthwise, norm_cfg=norm_cfg, act_cfg=act_cfg),
            # Finally, SPPF module
            SPPFBottleneck(
                in_channels=self.channels[3],
                out_channels=self.channels[3],
                kernel_sizes=5,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
        )
        # Output: 20x20x512 (This is the third feature map sent to the neck)

        # We don't use the original _build_stem_layer or _build_stage_layer methods
        # because we have defined the full architecture manually above.

    def forward(self, x):
        """
        Forward pass through the SOM-Backbone.
        Returns a tuple of feature maps for the neck.
        """
        x = self.stem(x)
        out0 = self.stage1(x)    # Output for Neck (P2), size: 160x160x64
        out1 = self.stage2(out0) # Output for Neck (P3), size: 80x80x128
        out2 = self.stage3(out1) # Output for Neck (P4), size: 40x40x256
        out3 = self.stage4(out2) # Output for Neck (P5), size: 20x20x512
        
        return (out0, out1, out2, out3)