_base_ = [
    '../atss_r50_fpn_2x_lvis_v1.py'
]


model = dict(
    bbox_head=dict(
        loss_cls=dict(type="GumbelFocalLoss",num_classes=1203,gamma=0,alpha=0.1),
        init_cfg=dict(type='Normal',layer='Conv2d',std=0.01,override=dict(type='Normal',name='atss_cls',std=0.01,bias_prob=0.06))))

work_dir='./experiments/lvis_oneshot/atss_r50_fpn_2x_lvis_v1_gumbel_g0_a01/'
# work_dir='./experiments/test/'
# data = dict(samples_per_gpu=5)

lr_config = dict(warmup_iters=1000,warmup_ratio=0.001)
fp16 = dict(loss_scale=512.)