_base_ = [
    '../atss_r50_fpn_1x_lvis_v0.5.py'
]


model = dict(
    bbox_head=dict(
        loss_cls=dict(type="GumbelFocalLoss",num_classes=1230,gamma=1,alpha=0.5),
        init_cfg=dict(type='Normal',layer='Conv2d',std=0.01,override=dict(type='Normal',name='atss_cls',std=0.01,bias_prob=0.05))))

work_dir='./experiments/lvis_oneshot/atss_r50_fpn_1x_lvis_v0.5_gumbel_g1_a05/'
# data = dict(samples_per_gpu=5)

lr_config = dict(warmup_iters=1000,warmup_ratio=0.001)
fp16 = dict(loss_scale=512.)