_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

model = dict(bbox_head=dict(loss_cls=dict(type="GumbelFocalLoss",num_classes=80,gamma=2.0,alpha=0.5),
                                         init_cfg=dict(type='Normal',layer='Conv2d',std=0.01,override=dict(type='Normal',name='retina_cls',std=0.01,bias_prob=0.08))
                                         ))

work_dir='./experiments/coco/retinanet_r50_fpn_2x_coco_gumbel_g2_a05/'
# work_dir='./experiments/test/'
# data = dict(samples_per_gpu=4)

lr_config = dict(warmup_iters=2000,warmup_ratio=0.001)