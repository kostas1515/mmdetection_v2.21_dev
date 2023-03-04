_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

model = dict(bbox_head=dict(loss_cls=dict(type="GumbelFocalLoss",use_extra_cls_out_channels=1,variant='unified',gamma=0.0,alpha=0.25),
                                         init_cfg=dict(type='Normal',layer='Conv2d',std=0.01,override=dict(type='Normal',name='retina_cls',std=0.01,bias_prob=0.06))
                                         ))

# work_dir='./experiments/coco/retinanet_r50_fpn_1x_coco_unified_g0_a025/'
work_dir='./experiments/test/'
# data = dict(samples_per_gpu=4)

# lr_config = dict(warmup_iters=1000,warmup_ratio=0.001)