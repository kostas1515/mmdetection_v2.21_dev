_base_ = [
    '../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

model = dict(
    backbone=dict(
        type='SE_ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/imagenet_full_se_r50_cosine_scheduler_mixup_aug_se_gumbel.pth'),
        use_gumbel=True),
    roi_head=dict(bbox_head=dict(loss_cls=dict(
                                                type='DropLoss',
                                                use_sigmoid=True,
                                                loss_weight=1.0,
                                                lambda_=0.0011,
                                                version='v1',
                                                use_classif='gumbel'),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-2.0, override=dict(name='fc_cls'))),
                 mask_head=dict(type='FCNMaskHead',predictor_cfg=dict(type='NormedConv2d',learnable_temp=True))),
    test_cfg = dict(rcnn=dict(nms=dict(type='nms', iou_threshold=0.3))))

evaluation = dict(metric=['bbox', 'segm'], interval=12)
optimizer = dict(
    _delete_=True
    type='AdamW',
    lr=0.0002,
    weight_decay=0.05,
    paramwise_cfg=dict(norm_decay_mult=0.0, bypass_duplicate=True))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)


work_dir='./experiments/test/'
