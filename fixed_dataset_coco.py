# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional

from mmdet.datasets import BaseDetDataset, CocoDataset
from pycocotools.coco import COCO
import os

from mmyolo.registry import DATASETS, TASK_UTILS


class BatchShapePolicyDataset(BaseDetDataset):
    """Dataset with the batch shape policy that makes paddings with least
    pixels during batch inference process, which does not require the image
    scales of all batches to be the same throughout validation."""

    def __init__(self,
                 *args,
                 batch_shapes_cfg: Optional[dict] = None,
                 **kwargs):
        self.batch_shapes_cfg = batch_shapes_cfg
        super().__init__(*args, **kwargs)

    def full_init(self):
        """rewrite full_init() to be compatible with serialize_data in
        BatchShapePolicy."""
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()

        # batch_shapes_cfg
        if self.batch_shapes_cfg:
            batch_shapes_policy = TASK_UTILS.build(self.batch_shapes_cfg)
            self.data_list = batch_shapes_policy(self.data_list)
            del batch_shapes_policy

        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def prepare_data(self, idx: int) -> Any:
        """
        data augmentation, such as Mosaic and MixUp."""
        if self.test_mode is False:
            data_info = self.get_data_info(idx)
            data_info['dataset'] = self

        else:
            return super().prepare_data(idx)


@DATASETS.register_module()
class FixedYOLOv5CocoDataset(BatchShapePolicyDataset, CocoDataset):
    """Fixed YOLOv5 COCO Dataset that properly loads COCO annotations.
    
    This fixes the issue where the original YOLOv5CocoDataset doesn't load
    the COCO object properly, resulting in empty instances.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure COCO object is loaded
        self._ensure_coco_loaded()

    def _ensure_coco_loaded(self):
        """Ensure COCO object is loaded and instances are populated."""
        if not hasattr(self, 'coco') or self.coco is None:
            # Load COCO object
            ann_file_path = os.path.join(self.data_root, self.ann_file)
            self.coco = COCO(ann_file_path)
            print(f"Loaded COCO object with {len(self.coco.anns)} annotations")

    def load_data_list(self):
        """Load data list and ensure instances are populated."""
        # Call parent method
        data_list = super().load_data_list()
        
        # Ensure COCO object is loaded
        self._ensure_coco_loaded()
        
        # Populate instances for each data item
        for data_item in data_list:
            if 'instances' not in data_item or not data_item['instances']:
                img_id = data_item['img_id']
                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                
                if ann_ids:
                    anns = self.coco.loadAnns(ann_ids)
                    instances = []
                    for ann in anns:
                        instance = {
                            'bbox': ann['bbox'],
                            'bbox_label': ann['category_id'],
                            'ignore_flag': ann.get('iscrowd', 0)
                        }
                        instances.append(instance)
                    data_item['instances'] = instances
        
        return data_list
