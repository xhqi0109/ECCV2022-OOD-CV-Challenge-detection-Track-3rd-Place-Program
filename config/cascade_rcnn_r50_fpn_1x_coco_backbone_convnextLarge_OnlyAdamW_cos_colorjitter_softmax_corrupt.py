model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth',
            prefix='backbone.')),
    neck=[
        dict(
            type='FPN',
            in_channels=[192, 384, 768, 1536],
            out_channels=256,
            num_outs=5),
        dict(type='BFP', in_channels=256, num_levels=5)
    ],
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 1.5, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.7, method='linear'),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(
                type='soft_nms',
                iou_threshold=0.4,
                method='linear',
                min_score=0.01),
            max_per_img=100)))
dataset_type = 'CocoDataset'
data_root = './data/'
classes = ('aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair',
           'diningtable', 'motorbike', 'sofa', 'train')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Occlusion',
        screen_imgDirPath='./data/occlusion/screen_img_black_final'),
    dict(
        type='Resize',
        img_scale=[(640, 1333), (800, 1600)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Corrupt',
            'corruption': 'frost'
        }, {
            'type': 'Corrupt',
            'corruption': 'fog'
        }, {
            'type': 'Corrupt',
            'corruption': 'snow'
        }, {
            'type': 'Corrupt',
            'corruption': 'brightness'
        }, {
            'type': 'Corrupt',
            'corruption': 'contrast'
        }, {
            'type': 'Corrupt',
            'corruption': 'elastic_transform'
        }, {
            'type': 'Corrupt',
            'corruption': 'pixelate'
        }, {
            'type': 'Corrupt',
            'corruption': 'jpeg_compression'
        }, {
            'type': 'Corrupt',
            'corruption': 'spatter'
        }, {
            'type': 'Corrupt',
            'corruption': 'saturate'
        }]]),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
                always_apply=False,
                p=0.4)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='coco',
            label_fields=['gt_labels'],
            min_area=100,
            min_visibility=0.1,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 640), (1333, 800), (1600, 1000), (1850, 1120),
                   (2000, 1200)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', direction=['horizontal', 'vertical']),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file=
        './data/train/train.json',
        # img_prefix=
        # '/home/data/lkd/ECCV2022_OOD/object-detection/ROBINv1.1-det/ROBINv1.1-det-new/train/Images/',
        img_prefix=
        './data/train/Images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Occlusion',
                screen_imgDirPath='./data/occlusion/screen_img_black_final'
            ),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type': 'Corrupt',
                    'corruption': 'frost'
                }, {
                    'type': 'Corrupt',
                    'corruption': 'fog'
                }, {
                    'type': 'Corrupt',
                    'corruption': 'snow'
                }]]),
            dict(
                type='Resize',
                img_scale=[(640, 1333), (800, 1600)],
                keep_ratio=True),
            dict(
                type='RandomFlip',
                flip_ratio=0.5,
                direction=['horizontal', 'vertical']),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair',
                 'diningtable', 'motorbike', 'sofa', 'train')),
    val=dict(
        type='CocoDataset',
        ann_file=
        './data/val/iid_test.json',
        # img_prefix=
        # '/home/data/lkd/ECCV2022_OOD/object-detection/ROBINv1.1-det/ROBINv1.1-det-new/nuisances/iid_test/Images/',
        img_prefix=
        './data/val/Images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair',
                 'diningtable', 'motorbike', 'sofa', 'train')),
    test=dict(
        type='CocoDataset',
        ann_file=
        './data/test/phase2-det.json',
        # img_prefix=
        # '/home/data/qxh/dataset/dataset_ECCV2022_OOD/final/final/phase2-det-new/images/',
        img_prefix=
        './data/test/Images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1333, 640), (1333, 800), (1600, 1000),
                           (1850, 1120), (2000, 1200)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomFlip',
                        direction=['horizontal', 'vertical']),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair',
                 'diningtable', 'motorbike', 'sofa', 'train')))
evaluation = dict(interval=50, metric='bbox')
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(decay_rate=0.7, decay_type='layer_wise', num_layers=12))
optimizer_config = dict(grad_clip=None)
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth'
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.1,
    min_lr=0,
    warmup_by_epoch=True)
albu_train_transforms = [
    dict(
        type='ColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2,
        always_apply=False,
        p=0.4)
]
work_dir = './work_dirs'
auto_resume = False
gpu_ids = range(0, 4)

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh /home/data/qxh/code/ECCV2022_OOD/det_ood_code/config/cascade_rcnn_r50_fpn_1x_coco_backbone_convnextLarge_OnlyAdamW_cos_colorjitter_softmax_corrupt.py 8 --seed 0 --deterministic --work-dir ./work_dirs/ 
'''
