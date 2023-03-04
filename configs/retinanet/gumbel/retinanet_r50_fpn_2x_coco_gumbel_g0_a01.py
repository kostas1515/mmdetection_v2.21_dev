_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

model = dict(bbox_head=dict(loss_cls=dict(type="GumbelFocalLoss",num_classes=80,gamma=0,alpha=0.1),
                                         init_cfg=dict(type='Normal',layer='Conv2d',std=0.01,override=dict(type='Normal',name='retina_cls',std=0.01,bias_prob=0.08))
                                         ))

work_dir='./experiments/coco/retinanet_r50_fpn_2x_coco_gumbel_g0_a01_schedule/'
# work_dir='./experiments/test/'
# fp16 = dict(loss_scale=512.)
# load_from = './experiments/coco/retinanet_r50_fpn_2x_coco_gumbel_g0_a01/epoch_1.pth'
# data = dict(samples_per_gpu=4)

lr_config = dict(warmup_iters=1000,warmup_ratio=0.001,step=[20,23])