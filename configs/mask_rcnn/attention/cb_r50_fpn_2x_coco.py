_base_ = '../mask_rcnn_r50_fpn_2x_coco.py'

model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='CBAM_ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='./pretrained/imagenet_full_cb_r50_cosine_scheduler_mixup_aug.pth'),
        use_gumbel=False))

work_dir = "./experiments/coco/cb_r50_fpn_2x_coco"