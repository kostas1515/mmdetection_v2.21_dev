_base_ = './droploss_swin_1x_rfs_gumbel_attn.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    roi_head=dict(mask_head=dict(
             num_classes=1203,
             predictor_cfg=dict(type='NormedConv2d',learnable_temp=True))))
work_dir = 'experiments/droploss_swin_s_1x_rfs_gumbel_attn_lr_cosnorm'
# work_dir = 'experiments/test/'

# resume_from = 'experiments/droploss_swin_s_1x_rfs_gumbel_attn/latest.pth'
fp16 = dict(loss_scale=512.) 
