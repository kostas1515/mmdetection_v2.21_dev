_base_ = './droploss_swin_1x_rfs.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
work_dir = 'experiments/droploss_swin_s_1x_rfs_softmax_attn'
