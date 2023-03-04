_base_ = './droploss_swin_1x_rfs_gumbel_attn.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 54, 2],
        embed_dims=128,
        num_heads=[4, 8, 16, 32],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    roi_head=dict(mask_head=dict(
             num_convs=4,
             num_classes=1203,
             predictor_cfg=dict(type='NormedConv2d'))),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5))
work_dir = 'experiments/test'
data = dict(
    samples_per_gpu=1)
# resume_from = 'experiments/droploss_swin_s_1x_rfs_gumbel_attn/latest.pth'
fp16 = dict(loss_scale='dynamic') 
