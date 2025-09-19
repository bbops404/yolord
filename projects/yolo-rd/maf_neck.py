import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from .arm import AttentionRefinementModule
@MODELS.register_module()
class MAFNeck(nn.Module):
    """
    Multi-dimensional Auxiliary Fusion (MAF) Neck with stride-2 convolutions
    for downsampling instead of MaxPool.
    """

    def __init__(self, in_channels):
        """
        Args:
            in_channels (list[int]): [Stage1, Stage2, Stage3, Stage4]
                                     Expected: [64, 128, 256, 512]
        """
        super().__init__()
        assert len(in_channels) == 4, "Expected in_channels = [64, 128, 256, 512]"
        c1, c2, c3, c4 = in_channels

        # notes:
        # kapag convmodule -> stride=2 kaya halved (pero the the channels are untouched)
        # kapag conv1x1 -> channels are reduced pero the height and width are untouched

        # --------------------
        # Stage 1 → 80x80 downsample conv (160x160 -> 80x80)
        # --------------------
        self.s1_down = ConvModule(
            c1, 64, 3, stride=2, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')
        )

        # --------------------
        # Attention Refinement Modules
        # --------------------
        self.arm2 = AttentionRefinementModule(c2)
        self.arm3 = AttentionRefinementModule(c3)
        self.arm4 = AttentionRefinementModule(c4)

        # --------------------
        # Stage 1&2 Fusion (80x80)
        # Concat(Stage1_down=64 + ARM(s2)=128) → 192 → 128
        # --------------------
        self.conv2_out = ConvModule(
            c2 + 64, 128, 1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')
        )

        # --------------------
        # Stage 2&3 Fusion (40x40)
        # s2_cat (192) ↓ stride-2 conv → (192, 40x40) → 1x1 → (64, 40x40)
        # Concat(64 + ARM(s3)=256) → 320 → 256
        # --------------------
        self.reduce2_spatial = ConvModule(
            c2 + 64, c2 + 64, 3, stride=2, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')
        )
        self.reduce2 = ConvModule(
            c2 + 64, 64, 1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')
        )
        self.conv3_out = ConvModule(
            c3 + 64, 256, 1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')
        )

        # --------------------
        # Stage 3&4 Fusion (20x20)
        # s3_cat (c3+64=320) ↓ stride-2 conv → (320, 20x20) → 1x1 → (64, 20x20)
        # Concat(64 + ARM(s4)=512) → 576 → 512
        # --------------------
        self.reduce3_spatial = ConvModule(
            c3 + 64, c3 + 64, 3, stride=2, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')
        )
        self.reduce3 = ConvModule(
            c3 + 64, 64, 1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')
        )
        self.conv4_out = ConvModule(
            c4 + 64, 512, 1, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU')
        )

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): [Stage1, Stage2, Stage3, Stage4]
                - Stage 1: (B, 64, 160, 160)
                - Stage 2: (B, 128, 80, 80)
                - Stage 3: (B, 256, 40, 40)
                - Stage 4: (B, 512, 20, 20)

        Returns:
            tuple[Tensor]: [P3, P4, P5]
                - P3: (B, 128, 80, 80)
                - P4: (B, 256, 40, 40)
                - P5: (B, 512, 20, 20)
        """
        s1, s2, s3, s4 = inputs

        # --------------------
        # Stage 1 → downsample
        # --------------------
        s1_down = self.s1_down(s1)  # (B, 64, 80, 80)

        # --------------------
        # Stage 2 → P3
        # --------------------
        s2_arm = self.arm2(s2)  # (B, 128, 80, 80)
        s2_cat = torch.cat([s1_down, s2_arm], dim=1)  # (B, 192, 80, 80)
        p3 = self.conv2_out(s2_cat)  # (B, 128, 80, 80)
        p3_red = self.reduce2(self.reduce2_spatial(s2_cat))  # (B, 64, 40, 40)

        # --------------------
        # Stage 3 → P4
        # --------------------
        s3_arm = self.arm3(s3)  # (B, 256, 40, 40)
        s3_cat = torch.cat([s3_arm, p3_red], dim=1)  # (B, 320, 40, 40)
        p4 = self.conv3_out(s3_cat)  # (B, 256, 40, 40)
        p4_red = self.reduce3(self.reduce3_spatial(s3_cat))  # (B, 64, 20, 20)

        # --------------------
        # Stage 4 → P5
        # --------------------
        s4_arm = self.arm4(s4)  # (B, 512, 20, 20)
        s4_cat = torch.cat([s4_arm, p4_red], dim=1)  # (B, 576, 20, 20)
        p5 = self.conv4_out(s4_cat)  # (B, 512, 20, 20)

        return p3, p4, p5