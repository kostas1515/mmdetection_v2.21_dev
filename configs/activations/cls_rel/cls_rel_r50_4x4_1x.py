_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

data = dict(train=dict(oversample_thr=0.0),samples_per_gpu=4)

# model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='normal'),
#                                          init_cfg = dict(type='Constant',val=0.01, bias=-3.45, override=dict(name='fc_cls')))))

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="Icloglog",activation='cls_rel',temperature=1.0),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-3, override=dict(name='fc_cls')))))

work_dir='./experiments/test/'
# work_dir='./experiments/cls_rel_r50_4x4_1x_gumbel_sigm_normal/'


fp16 = dict(loss_scale=512.)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])