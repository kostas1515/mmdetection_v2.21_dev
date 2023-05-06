_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v1.py'
]
# fp16 = dict(loss_scale=512.)
# data=dict(samples_per_gpu=8)

model = dict(roi_head=dict(bbox_head=dict(type='DisentangledMultiActivationBBoxHead',
                                            weight_similarity_loss=None,
                                            loss_cls=dict(type="MultiActivation",loss_cls='droploss',class_heads=2),
                                            init_cfg = dict(type='Normal', std=0.01, bias_prob=0.06, override=dict(name='fc_cls'))),
                        mask_head=dict(predictor_cfg=dict(type='NormedConv2d', tempearture=20))),
        test_cfg = dict(rcnn=dict(nms=dict(type='nms', iou_threshold=0.3))))

work_dir='./experiments/activations/multi_r50_rfs_4x4_2x_dropnorm_gumbel_normal_2h/'
# auto_scale_lr = dict(enable=True, base_batch_size=16)
# work_dir='./experiments/test/'

