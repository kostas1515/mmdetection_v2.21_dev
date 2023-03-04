_base_ = [
    '../atss_r50_fpn_1x_coco.py'
]


model = dict(
    bbox_head=dict(
        loss_cls=dict(type="GumbelFocalLoss",num_classes=80,gamma=2,alpha=0.25),
        init_cfg=dict(type='Normal',layer='Conv2d',std=0.01,override=dict(type='Normal',name='atss_cls',std=0.01,bias_prob=0.08))),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.0001,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

work_dir='./experiments/coco/atss_r50_fpn_1x_coco_gumbel_g2_a025/'
# work_dir='./experiments/test/'
# data = dict(samples_per_gpu=5)

# lr_config = dict(warmup_iters=1000,warmup_ratio=0.001)