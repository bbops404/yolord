# projects/yolo-rd/yolord_s_coco_config.py

# This tells MMYOLO to load your custom modules before building the model
custom_imports = dict(
    imports=[
        'projects.yolo-rd.star_operation_module',
        'projects.yolo-rd.maf_pafpn_neck',
        'projects.yolo-rd.wt_head'
    ],
    allow_failed_imports=False)

_base_ = [
    '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/det_p5_tta.py'
]

# ======================== Data Config ========================

data_root = 'C:/Users/Christian Buenagua/Downloads/thesis_start/DataSci28/RDD2022_Japan/'
class_name = ('D00', 'D10', 'D20', 'D40',)
dataset_type = 'YOLOv5CocoDataset'
num_classes = len(class_name)
img_scale = (640, 640)

# ======================== Model Config =======================
deepen_factor = 0.33
widen_factor = 0.5
last_stage_out_channels = 512
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
act_cfg = dict(type='SiLU', inplace=True)

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),

    # --- YOLO-RD Backbone with SOM ---
    # NOTE: To integrate SOM, the YOLOv8CSPDarknet backbone in MMYOLO would need
    # to be modified to accept plugins. For now, we define a standard backbone,
    # and a more advanced user would create a new 'SOM_YOLOv8CSPDarknet' class.
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),

    # --- YOLO-RD Neck (MAF-PAFPN) ---
    neck=dict(
        type='MAFPAFPN', # Using our custom neck
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),

    # --- YOLO-RD Head (WT-Head) ---
    bbox_head=dict(
        type='WTHead', # Using our custom head container
        head_module=dict(
            type='WTHeadModule', # Using our custom head module
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            featmap_strides=[8, 16, 32]),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=1.5 / 4)),

    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

# ====================== Training Config ======================
# Based on paper's implementation details
max_epochs = 200
base_lr = 0.02
train_batch_size_per_gpu = 16
train_num_workers = 8

# ... (The rest of the config for dataloaders, optimizers, hooks, etc.)
# The following is mostly standard from YOLOv8 configs
affine_scale = 0.5
close_mosaic_epochs = 10
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train_fixed.json',
         data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

# ... (Val/Test dataloaders and other hooks)
# The rest of the file can be copied from a standard yolov8-s config
# as the key changes are in the model definition above.
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val_fixed.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='YOLOv5KeepRatioResize', scale=img_scale),
            dict(
                type='LetterResize',
                scale=img_scale,
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=None))
test_dataloader = val_dataloader
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/val_fixed.json',
    metric='bbox')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='auto',
        max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]