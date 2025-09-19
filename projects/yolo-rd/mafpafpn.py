from typing import List

import torch.nn as nn

from mmyolo.registry import MODELS
from mmyolo.models.necks import YOLOv8PAFPN
from .maf_neck import MAFNeck


@MODELS.register_module()
class MAFPAFPN(nn.Module):
    """Composite neck: MAFNeck → YOLOv8PAFPN.

    Expects backbone outputs with channels [64, 128, 256, 512] and returns
    three feature maps suitable for YOLO heads.
    """

    def __init__(self,
                 maf_in_channels: List[int] = [64, 128, 256, 512],
                 pafpn_out_channels: List[int] = [128, 256, 512],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3):
        super().__init__()
        self.maf = MAFNeck(in_channels=maf_in_channels)
        self.pafpn = YOLOv8PAFPN(
            in_channels=pafpn_out_channels,
            out_channels=pafpn_out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
        )

    def forward(self, inputs):
        # inputs: tuple of 4 tensors from backbone
        p3, p4, p5 = self.maf(inputs)
        p3, p4, p5 = self.pafpn([p3, p4, p5])
        return p3, p4, p5


