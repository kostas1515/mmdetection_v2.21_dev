_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

data = dict(train=dict(oversample_thr=0.0))

model = dict(
    backbone=dict(
        type='SE_ResNeXt',
        depth=50,
        use_gumbel=True))


work_dir='./experiments/test/'