# Minimal test config to debug data loading
custom_imports = dict(
    imports=[
        'projects.yolo-rd.som_backbone',
        'projects.yolo-rd.mafpafpn',
        'projects.yolo-rd.wt_head',
        'projects.yolo-rd.fixed_coco_dataset'
    ],
    allow_failed_imports=True)

# Data config
data_root = '/home/ubuntu/yolord/data/RDD2022_Japan/'
class_name = ('D00', 'D10', 'D20', 'D40',)
dataset_type = 'YOLOv5CocoDataset'
num_classes = len(class_name)
img_scale = (640, 640)

# Model config
deepen_factor = 0.33
widen_factor = 1.0
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

    backbone=dict(
        type='SOM_YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),

    neck=dict(
        type='MAFPAFPN',
        maf_in_channels=[64, 128, 256, 512],
        pafpn_out_channels=[128, 256, 512],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        num_csp_blocks=3),

    bbox_head=dict(
        type='WTCHead',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[128, 256, 512],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            featmap_strides=[8, 16, 32]
        ),
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
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            topk=26,
            alpha=0.5,
            beta=3,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

# Minimal pipeline - load, resize, and pack
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.PackDetInputs', 
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,  # Small batch for testing
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train_fixed.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline,
        test_mode=False,
        lazy_init=False))

# Minimal training config
max_epochs = 1
base_lr = 0.001

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=2),
    constructor='YOLOv5OptimizerConstructor')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(max_epochs - 10, 1)])

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=1,
        log_metric_by_epoch=False),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=False)
