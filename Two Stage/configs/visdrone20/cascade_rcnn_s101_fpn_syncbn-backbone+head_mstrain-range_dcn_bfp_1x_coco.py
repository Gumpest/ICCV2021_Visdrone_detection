_base_ = './cascade_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py'

# optimizer 
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) # change lr 0.02->0.01
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnest101',
    # pretrained=None,
    backbone=dict(
        stem_channels=128, 
        depth=101,
        # groups=32, # add groups
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
        ),
    neck=[
        dict(
            type='PAFPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')
        ]
    )


# data
dataset_type = 'CocoDataset'
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')


img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True) # use ResNeSt img_norm
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=False,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(416, 416), (832, 832)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2, # batchsize
    workers_per_gpu=1,
    train=dict(
        img_prefix='/home/zy123/VisDrone2019-DET-train-test-split/images/',
        classes=classes,
        ann_file='/home/zy123/VisDrone2019-DET-train-test-split/VisDrone2019-DET_train-test_coco.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix='/home/lxc/visdrone/VisDrone2019-DET-valsplit/images/',
        classes=classes,
        ann_file='/home/zy123/mmdetection/mmdet/data/VisDrone2019-DET_val_coco.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='/home/lxc/visdrone/VisDrone2019-DET-valsplit/images/',
        classes=classes,
        ann_file='/home/zy123/mmdetection/mmdet/data/VisDrone2019-DET_val_coco.json',
        pipeline=test_pipeline))

work_dir = './work_dirs/Cascade_rcnn_resnest101_albu'