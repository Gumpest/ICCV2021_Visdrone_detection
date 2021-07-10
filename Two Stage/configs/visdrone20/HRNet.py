_base_ = '../hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
# fp16
fp16 = dict(loss_scale=512.)
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
runner = dict(type='EpochBasedRunner', max_epochs=30)

norm_cfg = dict(type='BN', requires_grad=True)
# model settings
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w40',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage2=dict(num_channels=(40, 80)),
            stage3=dict(num_channels=(40, 80, 160)),
            stage4=dict(num_channels=(40, 80, 160, 320)))),
    # neck=dict(type='HRFPN', in_channels=[40, 80, 160, 320], out_channels=256)
    neck=[
        dict(
            type='PAFPN',
            in_channels=[40, 80, 160, 320],
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
# use ResNeSt img_norm
img_norm_cfg = dict(
    mean=[95.79, 97.242, 93.648], std=[59.189, 59.07, 61.002], to_rgb=True)

# add albu
albu_train_transforms = [
    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=0,
    #     interpolation=1,
    #     p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.2),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='RGBShift',
    #             r_shift_limit=10,
    #             g_shift_limit=10,
    #             b_shift_limit=10,
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0)
    #     ],
    #     p=0.1),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='CLAHE', clip_limit=2),
            dict(type='Sharpen'),
            dict(type='Emboss'),
            dict(type='RandomBrightnessContrast'),
        ],
        p=0.3),
]

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
    # Albu has order
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
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
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

dataset_type = 'CocoDataset'
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
data = dict(
    samples_per_gpu=1, # batchsize
    workers_per_gpu=1,
    train=dict(
        img_prefix='/home/zy123/VisDrone2019-DET-train-test-split/images/',
        classes=classes,
        ann_file='/home/zy123/VisDrone2019-DET-train-test-split/VisDrone2019-DET_train-test_coco.json'),
    val=dict(
        img_prefix='/home/lxc/visdrone/VisDrone2019-DET-valsplit/images/',
        classes=classes,
        ann_file='/home/zy123/mmdetection/mmdet/data/VisDrone2019-DET_val_coco.json'),
    test=dict(
        img_prefix='/home/lxc/visdrone/VisDrone2019-DET-valsplit/images/',
        classes=classes,
        ann_file='/home/zy123/mmdetection/mmdet/data/VisDrone2019-DET_val_coco.json'))
work_dir = './work_dirs/Cascade_rcnn_hrnetw40_albu'

#load_from = '/home/zy123/mmdetection/work_dirs/hrNet/latest.pth'
#resume_from = '/home/zy123/mmdetection/work_dirs/hrNet/latest.pth'

