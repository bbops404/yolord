import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8Head, YOLOv8HeadModule
from mmdet.models.utils import multi_apply

try:
    from pytorch_wavelets import DWTForward
except ImportError:
    raise ImportError('Please run "pip install pytorch_wavelets" to install the required library.')

@MODELS.register_module()
class WTHeadModule(YOLOv8HeadModule):
    def _init_layers(self):
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.stems = nn.ModuleList() # Stems are not used in this custom head

        for i in range(self.num_levels):
            in_c = self.in_channels[i]
            reg_in_c = in_c * 3
            self.cls_preds.append(nn.Sequential(nn.Conv2d(in_c, self.num_classes, 1)))
            self.reg_preds.append(nn.Sequential(nn.Conv2d(reg_in_c, 4 * self.reg_max, 1)))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)
        print("--- INFO: Using Custom WTHeadModule ---")

    def forward(self, x):
        return multi_apply(self.forward_single, x, self.cls_preds, self.reg_preds)

    def forward_single(self, x, cls_pred_layer, reg_pred_layer):
        b, _, h, w = x.shape
        yl, yh = self.dwt(x)
        cls_logit = cls_pred_layer(yl)
        yh0 = yh[0]
        lh, hl, hh = yh0[:, :, 0, :, :], yh0[:, :, 1, :, :], yh0[:, :, 2, :, :]
        yh_cat = torch.cat([lh, hl, hh], dim=1)
        bbox_dist_pred = reg_pred_layer(yh_cat)
        cls_logit = F.interpolate(cls_logit, size=(h, w), mode='bilinear', align_corners=False)
        bbox_dist_pred = F.interpolate(bbox_dist_pred, size=(h, w), mode='bilinear', align_corners=False)

        if self.training:
            return cls_logit, bbox_dist_pred
        else:
            bbox_pred = self.decode_bbox(bbox_dist_pred, self.proj)
            return cls_logit, bbox_pred

@MODELS.register_module()
class WTHead(YOLOv8Head):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the head_module with our custom one
        self.head_module = MODELS.build(self.head_module)