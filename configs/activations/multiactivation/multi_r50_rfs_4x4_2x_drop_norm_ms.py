_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v1.py'
]
# fp16 = dict(loss_scale=512.)
# data=dict(samples_per_gpu=8)

model = dict(type='MaskScoringRCNN',
             roi_head=dict(type='MaskScoringRoIHead',
                           mask_iou_head=dict(
                            type='MaskIoUHead',
                            num_convs=4,
                            num_fcs=2,
                            roi_feat_size=14,
                            in_channels=256,
                            conv_out_channels=256,
                            fc_out_channels=1024,
                            num_classes=1203),
                           bbox_head=dict(type='DisentangledMultiActivationBBoxHead',
                                            weight_similarity_loss=None,
                                            loss_cls=dict(type="MultiActivation",loss_cls='droploss',class_heads=2),
                                            init_cfg = dict(type='Normal', std=0.01, bias_prob=0.06, override=dict(name='fc_cls'))),
                            mask_head=dict(
                                upsample_cfg=dict(
                                    type='carafe',
                                    scale_factor=2,
                                    up_kernel=5,
                                    up_group=1,
                                    encoder_kernel=3,
                                    encoder_dilation=1,
                                    compressed_channels=64),
                                predictor_cfg=dict(type='NormedConv2d', tempearture=20))),
        train_cfg=dict(rcnn=dict(mask_thr_binary=0.5)),
        test_cfg = dict(rcnn=dict(nms=dict(type='nms', iou_threshold=0.3),mask_thr_binary=0.4)))

work_dir='./experiments/activations/multi_r50_rfs_4x4_2x_drop_norm_ms_gs/'
# auto_scale_lr = dict(enable=True, base_batch_size=16)
# work_dir='./experiments/test/'

